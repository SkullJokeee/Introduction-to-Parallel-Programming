#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#include "ivfpq_gpu.h"

#define NPROBE 15
#define K_PQ 256
#define M 4
#define TOP_K 10
#define P 300
#define TILE_SIZE 16
#define BATCH_SIZE 500
#define NUM_STREAMS 2

size_t query_n = 0;
size_t gt_d = 0, vecdim = 0;
float* query;
int* gt;

struct SearchResult {
    float recall;
    int64_t latency;
};

template<typename T>
T* LoadData(std::string data_path, size_t& n, size_t& d){
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i * d * sz), d * sz);
    }
    fin.close();
    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
    return data;
}

// 反transpose IVF codebook，恢复行主序以适配 GPU 访问
void reverse_transpose_codebook(float* codebook, int nlist, int vecdim){
    int t_cb = (nlist / 4) * 4;
    float* temp_cb = new float[t_cb * vecdim];
    std::memcpy(temp_cb, codebook, t_cb * vecdim * sizeof(float));
    for(int g = 0; g < t_cb / 4; g++){
        for(int d = 0; d < vecdim; d++){
            for(int r = 0; r < 4; r++){
                int c = g * 4 + r;
                codebook[c * vecdim + d] = temp_cb[g * 4 * vecdim + d * 4 + r];
            }
        }
    }
    delete[] temp_cb;
}

// CPU粗选，为当前batch选出距离最近的nprobe个簇
void cpu_coarse_batch(const float* query_batch, const float* ivf_cb, int nlist, int vecdim, int batch_size, int* top_clusters){
    for(int q = 0; q < batch_size; q++){
        const float* q_vec = query_batch + q * vecdim;
        int* q_top = top_clusters + q * NPROBE;
        std::vector<std::pair<float, int>> temp_dists(nlist);
        for(int c = 0; c < nlist; c++){
            const float* center = ivf_cb + c * vecdim;
            float ip = 0.0f;
            for(int d = 0; d < vecdim; d++){
                ip += q_vec[d] * center[d];
            }
            temp_dists[c] = {1.0f - ip, c};
        }
        std::partial_sort(temp_dists.begin(), temp_dists.begin() + NPROBE, temp_dists.end());
        for(int i = 0; i < NPROBE; i++){
            q_top[i] = temp_dists[i].second;
        }
    }
}

struct StreamResource {
    cudaStream_t stream;
    float* d_query_batch;
    int* d_top_clusters;
    float* d_luts;
    float* d_cluster_dists;
    float* d_top_vals;
    uint32_t* d_top_idxs;
    uint32_t* d_candidates;
    float* d_final_vals;
    uint32_t* d_final_idxs;
    int* h_top_clusters;
    uint32_t* h_final_idxs_batch;
};

// GPU精确重排
void gpu_fine_batch(int batch_size, float* d_query_batch, float* d_pq_cbs, uint32_t* d_offset, const uint32_t* h_offset, uint32_t* d_lst, uint8_t* d_base_pq, float* d_base, int* d_top_clusters, int* h_top_clusters, size_t sub_d, size_t max_cluster_size, StreamResource& res, uint32_t* h_final_idxs_batch){
    cudaStream_t stream = res.stream;
    int threads1D = 256;
    int blocks1D = (batch_size + threads1D - 1) / threads1D;

    cudaMemcpyAsync(d_top_clusters, h_top_clusters, batch_size * NPROBE * sizeof(int), cudaMemcpyHostToDevice, stream);

    dim3 lut_grid(batch_size, NPROBE);
    build_lut_kernel<<<lut_grid, 256, 0, stream>>>(d_query_batch, d_pq_cbs, res.d_luts, d_top_clusters, batch_size, NPROBE, M, K_PQ, sub_d);

    init_topP_kernel<<<blocks1D, threads1D, 0, stream>>>(res.d_top_vals, res.d_top_idxs, batch_size, P);

    for(int c = 0; c < NPROBE; c++){
        uint32_t cluster_size = 0;
        for(int q = 0; q < batch_size; q++){
            int cluster_idx = h_top_clusters[q * NPROBE + c];
            uint32_t sz = h_offset[cluster_idx + 1] - h_offset[cluster_idx];
            if(sz > cluster_size){
                cluster_size = sz;
            }
        }

        if(cluster_size == 0){
            continue;
        }

        dim3 adc_grid(batch_size, (cluster_size + threads1D - 1) / threads1D);
        adc_lookup_kernel<<<adc_grid, threads1D, 0, stream>>>(d_base_pq, res.d_luts, res.d_cluster_dists, d_offset, d_top_clusters, batch_size, NPROBE, M, K_PQ, max_cluster_size, c);

        int merge_blocks = (batch_size + threads1D - 1) / threads1D;
        merge_clusters_kernel<<<merge_blocks, threads1D, 0, stream>>>(res.d_cluster_dists, d_lst, d_offset, d_top_clusters, res.d_top_vals, res.d_top_idxs, batch_size, NPROBE, P, max_cluster_size, c);
    }

    cudaMemcpyAsync(res.d_candidates, res.d_top_idxs, batch_size * P * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);

    init_topK_kernel<<<blocks1D, threads1D, 0, stream>>>(res.d_final_vals, res.d_final_idxs, batch_size, TOP_K);
    rerank_kernel<<<blocks1D, threads1D, 0, stream>>>(d_base, d_query_batch, res.d_candidates, res.d_final_vals, res.d_final_idxs, batch_size, P, TOP_K, vecdim);

    cudaMemcpyAsync(h_final_idxs_batch, res.d_final_idxs, batch_size * TOP_K * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
}

int main(int argc, char* argv[]){
    std::string data_path = "anndata/";
    std::string files_path = "files/";

    size_t cb_n = 0, cb_dim = 0;
    size_t pq_cb_n = 0, pq_cb_dim = 0;
    size_t offset_n = 0, offset_d = 0;
    size_t list_n = 0, list_d = 0;
    size_t base_pq_n = 0, base_pq_d = 0;
    size_t base_n = 0, base_dim = 0;

    query = LoadData<float>(data_path + "DEEP100K.query.fbin", query_n, vecdim);
    gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", query_n, gt_d);

    float* ivf_cb = LoadData<float>(files_path + "ivfpq_ivf_cb.bin", cb_n, cb_dim);
    reverse_transpose_codebook(ivf_cb, cb_n, cb_dim);
    float* pq_cbs = LoadData<float>(files_path + "ivfpq_pq_cb.bin", pq_cb_n, pq_cb_dim);
    uint32_t* offset = LoadData<uint32_t>(files_path + "ivfpq_offset.bin", offset_n, offset_d);
    uint32_t* lst = LoadData<uint32_t>(files_path + "ivfpq_list.bin", list_n, list_d);
    uint8_t* base_pq = LoadData<uint8_t>(files_path + "ivfpq_base.bin", base_pq_n, base_pq_d);
    float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_n, base_dim);

    query_n = 2000;
    size_t nlist = cb_n;
    size_t sub_d = vecdim / M;

    // 计算最大倒排列表长度
    size_t max_cluster_size = 0;
    for(size_t i = 0; i < nlist; ++i){
        uint32_t sz = offset[i + 1] - offset[i];
        if(sz > max_cluster_size){
            max_cluster_size = sz;
        }
    }

    std::vector<SearchResult> results(query_n);

    float *d_ivf_cb, *d_pq_cbs, *d_base;
    uint8_t *d_base_pq;
    uint32_t *d_offset, *d_lst;

    // 上传 IVF/PQ 索引与 base 到 GPU
    cudaMalloc(&d_ivf_cb, nlist * vecdim * sizeof(float));
    cudaMemcpy(d_ivf_cb, ivf_cb, nlist * vecdim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pq_cbs, pq_cb_n * pq_cb_dim * sizeof(float));
    cudaMemcpy(d_pq_cbs, pq_cbs, pq_cb_n * pq_cb_dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_offset, offset_n * offset_d * sizeof(uint32_t));
    cudaMemcpy(d_offset, offset, offset_n * offset_d * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_lst, list_n * list_d * sizeof(uint32_t));
    cudaMemcpy(d_lst, lst, list_n * list_d * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_base_pq, base_pq_n * base_pq_d * sizeof(uint8_t));
    cudaMemcpy(d_base_pq, base_pq, base_pq_n * base_pq_d * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_base, base_n * base_dim * sizeof(float));
    cudaMemcpy(d_base, base, base_n * base_dim * sizeof(float), cudaMemcpyHostToDevice);

    // 创建 CUDA 流与每流资源
    StreamResource res[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreate(&res[i].stream);
        cudaMalloc(&res[i].d_query_batch, BATCH_SIZE * vecdim * sizeof(float));
        cudaMalloc(&res[i].d_top_clusters, BATCH_SIZE * NPROBE * sizeof(int));
        cudaMalloc(&res[i].d_luts, BATCH_SIZE * NPROBE * M * K_PQ * sizeof(float));
        cudaMalloc(&res[i].d_cluster_dists, BATCH_SIZE * max_cluster_size * sizeof(float));
        cudaMalloc(&res[i].d_top_vals, BATCH_SIZE * P * sizeof(float));
        cudaMalloc(&res[i].d_top_idxs, BATCH_SIZE * P * sizeof(uint32_t));
        cudaMalloc(&res[i].d_candidates, BATCH_SIZE * P * sizeof(uint32_t));
        cudaMalloc(&res[i].d_final_vals, BATCH_SIZE * TOP_K * sizeof(float));
        cudaMalloc(&res[i].d_final_idxs, BATCH_SIZE * TOP_K * sizeof(uint32_t));
        res[i].h_top_clusters = new int[BATCH_SIZE * NPROBE];
        res[i].h_final_idxs_batch = new uint32_t[BATCH_SIZE * TOP_K];
    }

    int num_batches = (query_n + BATCH_SIZE - 1) / BATCH_SIZE;
    uint32_t* h_final_idxs = new uint32_t[query_n * TOP_K];

    auto total_start = std::chrono::high_resolution_clock::now();

    // 按 batch 提交 CPU 粗选 + GPU 精排流水线
    for(int b = 0; b < num_batches; b++){
        int batch_start = b * BATCH_SIZE;
        int batch_size = std::min(BATCH_SIZE, (int)query_n - batch_start);
        int stream_id = b % NUM_STREAMS;
        StreamResource& r = res[stream_id];

        cpu_coarse_batch(query + batch_start * vecdim, ivf_cb, nlist, vecdim, batch_size, r.h_top_clusters);

        cudaMemcpyAsync(r.d_query_batch, query + batch_start * vecdim, batch_size * vecdim * sizeof(float), cudaMemcpyHostToDevice, r.stream);

        gpu_fine_batch(batch_size, r.d_query_batch, d_pq_cbs, d_offset, offset, d_lst, d_base_pq, d_base, r.d_top_clusters, r.h_top_clusters, sub_d, max_cluster_size, r, r.h_final_idxs_batch);

        cudaMemcpyAsync(h_final_idxs + batch_start * TOP_K, r.h_final_idxs_batch, batch_size * TOP_K * sizeof(uint32_t), cudaMemcpyHostToHost, r.stream);
    }

    // 同步所有流并统计召回率
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamSynchronize(res[i].stream);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    int64_t total_latency = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    int64_t avg_latency = total_latency / query_n;

    for(size_t i = 0; i < query_n; i++){
        std::set<uint32_t> gtset;
        for(int j = 0; j < TOP_K; j++){
            gtset.insert(gt[j + i * gt_d]);
        }
        size_t acc = 0;
        for(int j = 0; j < TOP_K; j++){
            if(gtset.find(h_final_idxs[i * TOP_K + j]) != gtset.end()){
                acc++;
            }
        }
        results[i] = {(float)acc / TOP_K, avg_latency};
    }

    float avg_recall = 0, avg_lat = 0;
    for(size_t i = 0; i < query_n; i++){
        avg_recall += results[i].recall;
        avg_lat += results[i].latency;
    }

    std::cout << "average recall: " << avg_recall / query_n << "\n";
    std::cout << "average single query latency (us): " << avg_lat / query_n << "\n";
    std::cout << "total time for all queries (us): " << total_latency << "\n";

    delete[] h_final_idxs;
    delete[] query;
    delete[] gt;
    delete[] ivf_cb;
    delete[] pq_cbs;
    delete[] offset;
    delete[] lst;
    delete[] base_pq;
    delete[] base;

    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamDestroy(res[i].stream);
        cudaFree(res[i].d_query_batch);
        cudaFree(res[i].d_top_clusters);
        cudaFree(res[i].d_luts);
        cudaFree(res[i].d_cluster_dists);
        cudaFree(res[i].d_top_vals);
        cudaFree(res[i].d_top_idxs);
        cudaFree(res[i].d_candidates);
        cudaFree(res[i].d_final_vals);
        cudaFree(res[i].d_final_idxs);
        delete[] res[i].h_top_clusters;
        delete[] res[i].h_final_idxs_batch;
    }

    cudaFree(d_ivf_cb);
    cudaFree(d_pq_cbs);
    cudaFree(d_offset);
    cudaFree(d_lst);
    cudaFree(d_base_pq);
    cudaFree(d_base);

    return 0;
}
