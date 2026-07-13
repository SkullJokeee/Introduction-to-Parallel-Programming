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

    float *d_query, *d_ivf_cb, *d_pq_cbs, *d_base;
    uint8_t *d_base_pq;
    uint32_t *d_offset, *d_lst;

    // 上传 IVF/PQ 索引与 base 到 GPU
    cudaMalloc(&d_query, query_n * vecdim * sizeof(float));
    cudaMemcpy(d_query, query, query_n * vecdim * sizeof(float), cudaMemcpyHostToDevice);

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

    float* d_coarse_dist;
    int* d_top_clusters;
    cudaMalloc(&d_coarse_dist, query_n * nlist * sizeof(float));
    cudaMalloc(&d_top_clusters, query_n * NPROBE * sizeof(int));

    dim3 coarse_grid((nlist + TILE_SIZE - 1) / TILE_SIZE, (query_n + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    int threads1D = 256;
    int blocks1D = (query_n + threads1D - 1) / threads1D;

    auto total_start = std::chrono::high_resolution_clock::now();

    // 粗选：计算查询-簇距离并选出 nprobe
    coarse_sgemm_kernel<<<coarse_grid, block>>>(d_ivf_cb, d_query, d_coarse_dist, nlist, query_n, vecdim);
    coarse_select_kernel<<<blocks1D, threads1D>>>(d_coarse_dist, d_top_clusters, query_n, nlist, NPROBE);

    int* h_top_clusters = new int[query_n * NPROBE];
    cudaMemcpy(h_top_clusters, d_top_clusters, query_n * NPROBE * sizeof(int), cudaMemcpyDeviceToHost);

    float* d_luts;
    cudaMalloc(&d_luts, query_n * NPROBE * M * K_PQ * sizeof(float));

    dim3 lut_grid(query_n, NPROBE);
    int lut_threads = 256;
    build_lut_kernel<<<lut_grid, lut_threads>>>(d_query, d_pq_cbs, d_luts, d_top_clusters, query_n, NPROBE, M, K_PQ, sub_d);

    float* d_cluster_dists;
    cudaMalloc(&d_cluster_dists, query_n * max_cluster_size * sizeof(float));

    float* d_top_vals;
    uint32_t* d_top_idxs;
    cudaMalloc(&d_top_vals, query_n * P * sizeof(float));
    cudaMalloc(&d_top_idxs, query_n * P * sizeof(uint32_t));

    init_topP_kernel<<<blocks1D, threads1D>>>(d_top_vals, d_top_idxs, query_n, P);

    // 合并到 top-P
    for(int c = 0; c < NPROBE; c++){
        uint32_t cluster_size = 0;
        for(int q = 0; q < query_n; q++){
            int cluster_idx = h_top_clusters[q * NPROBE + c];
            uint32_t sz = offset[cluster_idx + 1] - offset[cluster_idx];
            if(sz > cluster_size){
                cluster_size = sz;
            }
        }

        if(cluster_size == 0){
            continue;
        }

        dim3 adc_grid(query_n, (cluster_size + threads1D - 1) / threads1D);
        adc_lookup_kernel<<<adc_grid, threads1D>>>(d_base_pq, d_luts, d_cluster_dists, d_offset, d_top_clusters, query_n, NPROBE, M, K_PQ, max_cluster_size, c);

        int merge_blocks = (query_n + threads1D - 1) / threads1D;
        merge_clusters_kernel<<<merge_blocks, threads1D>>>(d_cluster_dists, d_lst, d_offset, d_top_clusters, d_top_vals, d_top_idxs, query_n, NPROBE, P, max_cluster_size, c);
    }

    float* h_top_vals = new float[query_n * P];
    uint32_t* h_top_idxs = new uint32_t[query_n * P];
    cudaMemcpy(h_top_vals, d_top_vals, query_n * P * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_idxs, d_top_idxs, query_n * P * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    float* d_final_vals;
    uint32_t* d_final_idxs;
    cudaMalloc(&d_final_vals, query_n * TOP_K * sizeof(float));
    cudaMalloc(&d_final_idxs, query_n * TOP_K * sizeof(uint32_t));

    init_topK_kernel<<<blocks1D, threads1D>>>(d_final_vals, d_final_idxs, query_n, TOP_K);

    uint32_t* d_candidates;
    cudaMalloc(&d_candidates, query_n * P * sizeof(uint32_t));
    cudaMemcpy(d_candidates, h_top_idxs, query_n * P * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    // 精确重排
    rerank_kernel<<<blocks1D, threads1D>>>(d_base, d_query, d_candidates, d_final_vals, d_final_idxs, query_n, P, TOP_K, vecdim);

    cudaDeviceSynchronize();

    uint32_t* h_final_results = new uint32_t[query_n * TOP_K];
    cudaMemcpy(h_final_results, d_final_idxs, query_n * TOP_K * sizeof(uint32_t), cudaMemcpyDeviceToHost);

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
            if(gtset.find(h_final_results[i * TOP_K + j]) != gtset.end()){
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

    delete[] h_top_clusters;
    delete[] h_top_vals;
    delete[] h_top_idxs;
    delete[] h_final_results;
    delete[] query;
    delete[] gt;
    delete[] ivf_cb;
    delete[] pq_cbs;
    delete[] offset;
    delete[] lst;
    delete[] base_pq;
    delete[] base;

    cudaFree(d_query);
    cudaFree(d_ivf_cb);
    cudaFree(d_pq_cbs);
    cudaFree(d_offset);
    cudaFree(d_lst);
    cudaFree(d_base_pq);
    cudaFree(d_base);
    cudaFree(d_coarse_dist);
    cudaFree(d_top_clusters);
    cudaFree(d_luts);
    cudaFree(d_cluster_dists);
    cudaFree(d_top_vals);
    cudaFree(d_top_idxs);
    cudaFree(d_candidates);
    cudaFree(d_final_vals);
    cudaFree(d_final_idxs);

    return 0;
}
