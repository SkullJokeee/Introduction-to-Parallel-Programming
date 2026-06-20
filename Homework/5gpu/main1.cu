#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <queue>
#include <cuda_runtime.h>

#define tile

#define k 10
#define tile_size 16

size_t test_number = 0, base_number = 0;
size_t test_gt_d = 0, vecdim = 0;
float* test_query;
int* test_gt;
float* base;

struct SearchResult{
    float recall;
    int64_t latency;
};

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; i++){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";

    return data;
}


__global__ void sgemm_kernel(const float* base, const float* query, float* dist, int N, int M, int D){
    int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(q_idx < M && b_idx < N){
        float sum = 0.0f;
        for(int d = 0; d < D; d++){
            sum += query[q_idx * D + d] * base[b_idx * D + d];
        }
        dist[q_idx * N + b_idx] = sum;
    }
}

// 经过 Tiling 优化的矩阵乘法核
__global__ void tiled_sgemm_kernel(const float* base, const float* query, float* dist, int N, int M, int D){
    __shared__ float s_query[tile_size][tile_size + 1];
    __shared__ float s_base[tile_size][tile_size + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * tile_size + ty; // query 索引
    int col = bx * tile_size + tx; // base 索引

    float sum = 0.0f;
    int num_tiles = (D + tile_size - 1) / tile_size;

    for(int t = 0; t < num_tiles; t++){
        // 加载 query 到共享内存
        if(row < M && (t * tile_size + tx) < D){
            s_query[ty][tx] = query[row * D + t * tile_size + tx];
        }else{
            s_query[ty][tx] = 0.0f;
        }

        // 加载 base 到共享内存
        int b_row = bx * tile_size + ty;
        int b_col = t * tile_size + tx;
        if(b_row < N && b_col < D){
            s_base[ty][tx] = base[b_row * D + b_col];
        }else{
            s_base[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 点积计算
        for(int step = 0; step < tile_size; step++){
            sum += s_query[ty][step] * s_base[tx][step];
        }

        __syncthreads();
    }

    if(row < M && col < N){
        dist[row * N + col] = sum;
    }
}

// 寄存器级 Top-K 提取核
__global__ void select_kernel(const float* d_dist, int* d_res_idx, int base_number, int test_number){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(q_idx >= test_number){
        return;
    }
    
    const float* q_dist = d_dist + q_idx * base_number;
    int* q_res = d_res_idx + q_idx * k;
    
    float top_val[k];
    int top_idx[k];
    
    for(int i = 0; i < k; i++){
        top_val[i] = -1e9f;
        top_idx[i] = -1;
    }
    
    for(int j = 0; j < base_number; j++){
        float val = q_dist[j];
        if(val > top_val[k - 1]){
            int p = k - 1;
            while(p > 0 && val > top_val[p - 1]){
                top_val[p] = top_val[p - 1];
                top_idx[p] = top_idx[p - 1];
                p--;
            }
            top_val[p] = val;
            top_idx[p] = j;
        }
    }
    
    for(int i = 0; i < k; i++){
        q_res[i] = top_idx[i];
    }
}

int main(int argc, char *argv[]){
    std::string data_path = "anndata/";  
    
    size_t cb_n = 0, cb_dim = 0;
    size_t offset_n = 0, offset_d = 0;
    size_t list_n = 0, list_d = 0;

    test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 2000;
    std::vector<SearchResult> results;
    results.resize(test_number);
    
    float* d_base;
    float* d_query;
    float* d_dist;
    int* d_res_idx;
    
    cudaMalloc(&d_base, base_number * vecdim * sizeof(float));
    cudaMalloc(&d_query, test_number * vecdim * sizeof(float));
    cudaMalloc(&d_dist, base_number * test_number * sizeof(float));
    cudaMalloc(&d_res_idx, test_number * k * sizeof(int));
    
    cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, test_query, test_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlockGEMM(tile_size, tile_size);
    dim3 blocksPerGridGEMM((base_number + tile_size - 1) / tile_size, (test_number + tile_size - 1) / tile_size);
        
    int threadsPerBlockSelect = 256;
    int blocksPerGridSelect = (test_number + threadsPerBlockSelect - 1) / threadsPerBlockSelect;
        
    auto total_start_time = std::chrono::high_resolution_clock::now();

    #ifdef tile
        tiled_sgemm_kernel<<<blocksPerGridGEMM, threadsPerBlockGEMM>>>(d_base, d_query, d_dist, base_number, test_number, vecdim);
        cudaDeviceSynchronize();

        select_kernel<<<blocksPerGridSelect, threadsPerBlockSelect>>>(d_dist, d_res_idx, base_number, test_number);
        cudaDeviceSynchronize();

    #else
        sgemm_kernel<<<blocksPerGridGEMM, threadsPerBlockGEMM>>>(d_base, d_query, d_dist, base_number, test_number, vecdim);
        cudaDeviceSynchronize();

        select_kernel<<<blocksPerGridSelect, threadsPerBlockSelect>>>(d_dist, d_res_idx, base_number, test_number);
        cudaDeviceSynchronize();

    #endif
    
    int* res_idx = new int[test_number * k];
    cudaMemcpy(res_idx, d_res_idx, test_number * k * sizeof(int), cudaMemcpyDeviceToHost);
    auto total_end_time = std::chrono::high_resolution_clock::now();

    int64_t total_latency = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time).count();
    int64_t avg_simulated_latency = total_latency / test_number;
    
    for(int i = 0; i < test_number; i++){
        std::set<uint32_t> gtset;
        for(int j = 0; j < k; j++){
            int t = test_gt[j + i * test_gt_d];
            gtset.insert(t);
        }
        size_t acc = 0;
        for(int j = 0; j < k; j++){
            int x = res_idx[i * k + j];
            if(gtset.find(x) != gtset.end()){
                acc++;
            }
        }
        float recall = (float)acc / k;
        results[i] = {recall, avg_simulated_latency};
    }
    
    float avg_recall = 0, avg_latency = 0;
    
    for(int i = 0; i < test_number; i++){
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }
    
    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average single query latency (us): " << avg_latency / test_number << "\n";
    std::cout << "total time for all queries (us): " << total_latency << "\n"; 
    std::cout << "average time per query (us): " << total_latency / test_number << "\n";
    
    delete[] res_idx;
    
    cudaFree(d_base);
    cudaFree(d_query);
    cudaFree(d_dist);
    cudaFree(d_res_idx);

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
}