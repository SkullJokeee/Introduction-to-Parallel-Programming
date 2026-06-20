#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#include "ivf_gpu.h"

#define group

#define BATCH_SIZE 256

size_t test_number = 0, base_number = 0;
size_t test_gt_d = 0, vecdim = 0;
float* test_query;
int* test_gt;

struct SearchResult{
    float recall;
    int64_t latency;
};

struct QueryInfo{
    int orig_idx;
    int top1_cluster;
};

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d){
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
    return data;
}

int main(int argc, char *argv[]){
    std::string data_path = "anndata/";  
    
    size_t cb_n = 0, cb_dim = 0;
    size_t offset_n = 0, offset_d = 0;
    size_t list_n = 0, list_d = 0;
    size_t ivf_base_n = 0, ivf_base_dim = 0;

    test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);

    float* codebook_ivf = LoadData<float>("files/ivf_codebook.bin", cb_n, cb_dim);
    uint32_t* offset_ivf = LoadData<uint32_t>("files/ivf_offset.bin", offset_n, offset_d);
    uint32_t* list_ivf = LoadData<uint32_t>("files/ivf_ivflist.bin", list_n, list_d);
    float* base_ivf = LoadData<float>("files/ivf_base.bin", ivf_base_n, ivf_base_dim);

    test_number = 2000;
    base_number = list_n; 
    std::vector<SearchResult> results(test_number);
    
    float *d_query, *d_codebook_ivf, *d_base_ivf;
    uint32_t *d_offset_ivf, *d_list_ivf;

    cudaMalloc(&d_query, test_number * vecdim * sizeof(float));
    cudaMemcpy(d_query, test_query, test_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_codebook_ivf, cb_n * cb_dim * sizeof(float));
    cudaMalloc(&d_offset_ivf, offset_n * offset_d * sizeof(uint32_t));
    cudaMalloc(&d_list_ivf, list_n * list_d * sizeof(uint32_t));
    cudaMalloc(&d_base_ivf, ivf_base_n * ivf_base_dim * sizeof(float));

    cudaMemcpy(d_codebook_ivf, codebook_ivf, cb_n * cb_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset_ivf, offset_ivf, offset_n * offset_d * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_list_ivf, list_ivf, list_n * list_d * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_ivf, base_ivf, ivf_base_n * ivf_base_dim * sizeof(float), cudaMemcpyHostToDevice);

    uint32_t max_cluster_size = 0;
    for(int i = 0; i < cb_n; ++i){
        uint32_t sz = offset_ivf[i + 1] - offset_ivf[i];
        if(sz > max_cluster_size) max_cluster_size = sz;
    }

    float* d_coarse_dist;
    int* d_top_clusters;
    cudaMalloc(&d_coarse_dist, test_number * cb_n * sizeof(float));
    cudaMalloc(&d_top_clusters, test_number * nprobe * sizeof(int));

    dim3 coarse_grid((cb_n + tile_size - 1) / tile_size, (test_number + tile_size - 1) / tile_size);
    dim3 block(tile_size, tile_size);
    int threadsPerBlock1D = 256;
    int blocksPerGrid1D = (test_number + threadsPerBlock1D - 1) / threadsPerBlock1D;

    auto total_start_time = std::chrono::high_resolution_clock::now();

    // 粗量化计算
    generic_sgemm_kernel<<<coarse_grid, block>>>(d_codebook_ivf, d_query, d_coarse_dist, cb_n, test_number, vecdim);
    coarse_select_kernel<<<blocksPerGrid1D, threadsPerBlock1D>>>(d_coarse_dist, d_top_clusters, test_number, cb_n);

    int* h_top_clusters = new int[test_number * nprobe];
    cudaMemcpy(h_top_clusters, d_top_clusters, test_number * nprobe * sizeof(int), cudaMemcpyDeviceToHost);

    int* res_idx = new int[test_number * k];

    #ifdef group
        std::vector<QueryInfo> q_info(test_number);
        for(int i = 0; i < test_number; ++i){
            q_info[i].orig_idx = i;
            q_info[i].top1_cluster = h_top_clusters[i * nprobe];
        }
        
        std::sort(q_info.begin(), q_info.end(), [](const QueryInfo& a, const QueryInfo& b){
            return a.top1_cluster < b.top1_cluster;
        });

        float* h_sorted_query = new float[test_number * vecdim];
        for(int i = 0; i < test_number; ++i){
            int orig = q_info[i].orig_idx;
            memcpy(h_sorted_query + i * vecdim, test_query + orig * vecdim, vecdim * sizeof(float));
        }

        float* d_sorted_query;
        cudaMalloc(&d_sorted_query, test_number * vecdim * sizeof(float));
        cudaMemcpy(d_sorted_query, h_sorted_query, test_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);

        int num_batches = (test_number + BATCH_SIZE - 1) / BATCH_SIZE;
        bool* h_valid_mask = new bool[num_batches * cb_n]();
        
        for(int i = 0; i < test_number; ++i){
            int orig = q_info[i].orig_idx;
            int batch_id = i / BATCH_SIZE;
            for(int p = 0; p < nprobe; ++p){
                int c = h_top_clusters[orig * nprobe + p];
                h_valid_mask[batch_id * cb_n + c] = true;
            }
        }

        int* h_base_to_cluster = new int[base_number];
        for(int c = 0; c < cb_n; ++c){
            for(int i = offset_ivf[c]; i < offset_ivf[c + 1]; ++i){
                h_base_to_cluster[i] = c;
            }
        }

        bool* d_valid_mask;
        int* d_base_to_cluster;
        cudaMalloc(&d_valid_mask, num_batches * cb_n * sizeof(bool));
        cudaMalloc(&d_base_to_cluster, base_number * sizeof(int));
        cudaMemcpy(d_valid_mask, h_valid_mask, num_batches * cb_n * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_base_to_cluster, h_base_to_cluster, base_number * sizeof(int), cudaMemcpyHostToDevice);

        float* d_masked_dist;
        int* d_sorted_res_idx;
        cudaMalloc(&d_masked_dist, test_number * base_number * sizeof(float));
        cudaMalloc(&d_sorted_res_idx, test_number * k * sizeof(int));

        dim3 masked_grid((base_number + tile_size - 1) / tile_size, (test_number + tile_size - 1) / tile_size);
        
        masked_tiled_sgemm_kernel<<<masked_grid, block>>>(
            d_base_ivf, d_sorted_query, d_masked_dist, 
            base_number, test_number, vecdim, 
            d_base_to_cluster, d_valid_mask, BATCH_SIZE, cb_n
        );

        ivf_masked_select_kernel<<<blocksPerGrid1D, threadsPerBlock1D>>>(
            d_masked_dist, d_list_ivf, d_sorted_res_idx, base_number, test_number
        );

        cudaDeviceSynchronize();

        int* h_sorted_res_idx = new int[test_number * k];
        cudaMemcpy(h_sorted_res_idx, d_sorted_res_idx, test_number * k * sizeof(int), cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < test_number; ++i){
            int orig_query_idx = q_info[i].orig_idx;
            memcpy(res_idx + orig_query_idx * k, h_sorted_res_idx + i * k, k * sizeof(int));
        }

        delete[] h_sorted_query;
        delete[] h_valid_mask;
        delete[] h_base_to_cluster;
        delete[] h_sorted_res_idx;

        cudaFree(d_sorted_query);
        cudaFree(d_valid_mask);
        cudaFree(d_base_to_cluster);
        cudaFree(d_masked_dist);
        cudaFree(d_sorted_res_idx);

    #else
        float* d_cluster_dist;
        float* d_global_top_val;
        int* d_global_top_idx;

        cudaMalloc(&d_cluster_dist, test_number * max_cluster_size * sizeof(float));
        cudaMalloc(&d_global_top_val, test_number * k * sizeof(float));
        cudaMalloc(&d_global_top_idx, test_number * k * sizeof(int));

        init_topk_kernel<<<blocksPerGrid1D, threadsPerBlock1D>>>(d_global_top_val, d_global_top_idx, test_number);

        std::vector<int> unique_clusters;
        for(int i = 0; i < test_number * nprobe; ++i){
            unique_clusters.push_back(h_top_clusters[i]);
        }
        std::sort(unique_clusters.begin(), unique_clusters.end());
        unique_clusters.erase(std::unique(unique_clusters.begin(), unique_clusters.end()), unique_clusters.end());

        for(int cluster_id : unique_clusters){
            uint32_t c_start = offset_ivf[cluster_id];
            uint32_t c_end = offset_ivf[cluster_id + 1];
            uint32_t c_size = c_end - c_start;

            if(c_size == 0) continue;

            dim3 fine_grid((c_size + tile_size - 1) / tile_size, (test_number + tile_size - 1) / tile_size);
            
            generic_sgemm_kernel<<<fine_grid, block>>>(
                d_base_ivf + c_start * vecdim, 
                d_query, 
                d_cluster_dist, 
                c_size, test_number, vecdim
            );

            update_global_topk_kernel<<<blocksPerGrid1D, threadsPerBlock1D>>>(
                d_cluster_dist, d_list_ivf, c_start, d_global_top_val, d_global_top_idx, test_number, c_size
            );
        }

        cudaDeviceSynchronize();
        cudaMemcpy(res_idx, d_global_top_idx, test_number * k * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_cluster_dist);
        cudaFree(d_global_top_val);
        cudaFree(d_global_top_idx);

    #endif

    auto total_end_time = std::chrono::high_resolution_clock::now();
    int64_t total_latency = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time).count();
    int64_t avg_simulated_latency = total_latency / test_number;
    
    for(int i = 0; i < test_number; i++){
        std::set<uint32_t> gtset;
        for(int j = 0; j < k; j++){
            gtset.insert(test_gt[j + i * test_gt_d]);
        }
        size_t acc = 0;
        for(int j = 0; j < k; j++){
            if(gtset.find(res_idx[i * k + j]) != gtset.end()){
                acc++;
            }
        }
        results[i] = {(float)acc / k, avg_simulated_latency};
    }
    
    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; i++){
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }
    
    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average single query latency (us): " << avg_latency / test_number << "\n";
    std::cout << "total time for all queries (us): " << total_latency << "\n"; 
    
    delete[] res_idx;
    delete[] h_top_clusters;
    delete[] test_query;
    delete[] test_gt;
    delete[] codebook_ivf;
    delete[] offset_ivf;
    delete[] list_ivf;
    delete[] base_ivf;

    cudaFree(d_query);
    cudaFree(d_codebook_ivf);
    cudaFree(d_offset_ivf);
    cudaFree(d_list_ivf);
    cudaFree(d_base_ivf);
    cudaFree(d_coarse_dist);
    cudaFree(d_top_clusters);
    
    return 0;
}