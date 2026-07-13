#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

#define nprobe 40
#define k 10
#define tile_size 16

__global__ void generic_sgemm_kernel(const float* base, const float* query, float* dist, int N, int M, int D){
    __shared__ float s_query[tile_size][tile_size + 1];
    __shared__ float s_base[tile_size][tile_size + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    float sum = 0.0f;
    int num_tiles = (D + tile_size - 1) / tile_size;

    for(int t = 0; t < num_tiles; t++){
        if(row < M && (t * tile_size + tx) < D){
            s_query[ty][tx] = query[row * D + t * tile_size + tx];
        }else{
            s_query[ty][tx] = 0.0f;
        }

        int b_row = bx * tile_size + ty;
        int b_col = t * tile_size + tx;
        if(b_row < N && b_col < D){
            s_base[ty][tx] = base[b_row * D + b_col];
        }else{
            s_base[ty][tx] = 0.0f;
        }

        __syncthreads();
        for(int step = 0; step < tile_size; step++){
            sum += s_query[ty][step] * s_base[tx][step];
        }
        __syncthreads();
    }

    if(row < M && col < N){
        dist[row * N + col] = sum;
    }
}

// 粗量化选择核：找出每个 Query 最近的 nprobe 个簇
__global__ void coarse_select_kernel(const float* dist, int* top_clusters, int M, int Cb){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= M){
        return;
    }

    const float* q_dist = dist + q_idx * Cb;
    int* q_res = top_clusters + q_idx * nprobe;
    
    float top_val[nprobe];
    int top_idx[nprobe];
    for(int i = 0; i < nprobe; i++){
        top_val[i] = -1e9f;
        top_idx[i] = -1;
    }
    
    for(int j = 0; j < Cb; j++){
        float val = q_dist[j];
        if(val > top_val[nprobe - 1]){
            int p = nprobe - 1;
            while(p > 0 && val > top_val[p - 1]){
                top_val[p] = top_val[p - 1];
                top_idx[p] = top_idx[p - 1];
                p--;
            }
            top_val[p] = val;
            top_idx[p] = j;
        }
    }

    for(int i = 0; i < nprobe; i++){
        q_res[i] = top_idx[i];
    }
}

__global__ void init_topk_kernel(float* global_top_val, int* global_top_idx, int M){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= M) return;
    for(int i = 0; i < k; i++){
        global_top_val[q_idx * k + i] = -1e9f;
        global_top_idx[q_idx * k + i] = -1;
    }
}

__global__ void update_global_topk_kernel(const float* cluster_dist, const uint32_t* ivflist, uint32_t cluster_offset, float* global_top_val, int* global_top_idx, int M, int Nc){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= M){
        return;
    }

    const float* c_dist = cluster_dist + q_idx * Nc;
    float* top_val = global_top_val + q_idx * k;
    int* top_idx = global_top_idx + q_idx * k;

    float reg_val[k];
    int reg_idx[k];
    for(int i = 0; i < k; i++){
        reg_val[i] = top_val[i];
        reg_idx[i] = top_idx[i];
    }

    for(int j = 0; j < Nc; j++){
        float val = c_dist[j];
        int orig_idx = ivflist[cluster_offset + j];
        if(val > reg_val[k - 1]){
            int p = k - 1;
            while(p > 0 && val > reg_val[p - 1]){
                reg_val[p] = reg_val[p - 1];
                reg_idx[p] = reg_idx[p - 1];
                p--;
            }
            reg_val[p] = val;
            reg_idx[p] = orig_idx;
        }
    }

    for(int i = 0; i < k; i++){
        top_val[i] = reg_val[i];
        top_idx[i] = reg_idx[i];
    }
}

__global__ void masked_tiled_sgemm_kernel(const float* base, const float* query, float* dist, int N, int M, int D, const int* base_to_cluster, const bool* valid_mask, int batch_size, int cb_n){
    __shared__ bool skip_block;
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Block 级的掩码检查：0号线程去查表，整个 Block 共享结果
    if(tx == 0 && ty == 0){
        // 获取当前 Block 负责的 Base 矩阵列边界，并映射到它所属的簇 ID
        int c1 = base_to_cluster[min(bx * tile_size, N - 1)];
        int c2 = base_to_cluster[min(bx * tile_size + tile_size - 1, N - 1)];
        
        // 获取当前 Block 负责的 Query 矩阵行所属的 Batch ID
        int batch_id = (by * tile_size) / batch_size;
        
        skip_block = !(valid_mask[batch_id * cb_n + c1] || valid_mask[batch_id * cb_n + c2]);
    }
    __syncthreads();

    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    if(skip_block){
        if(row < M && col < N) dist[row * N + col] = -1e9f;
        return;
    }

    __shared__ float s_query[tile_size][tile_size + 1];
    __shared__ float s_base[tile_size][tile_size + 1];

    float sum = 0.0f;
    int num_tiles = (D + tile_size - 1) / tile_size;

    for(int t = 0; t < num_tiles; t++){
        if(row < M && (t * tile_size + tx) < D) s_query[ty][tx] = query[row * D + t * tile_size + tx];
        else s_query[ty][tx] = 0.0f;

        int b_row = bx * tile_size + ty;
        int b_col = t * tile_size + tx;
        if(b_row < N && b_col < D) s_base[ty][tx] = base[b_row * D + b_col];
        else s_base[ty][tx] = 0.0f;

        __syncthreads();
        for(int step = 0; step < tile_size; ++step){
            sum += s_query[ty][step] * s_base[tx][step];
        }
        __syncthreads();
    }

    if(row < M && col < N) dist[row * N + col] = sum;
}

__global__ void ivf_masked_select_kernel(const float* d_dist, const uint32_t* ivflist, int* d_res_idx, int base_number, int test_number){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= test_number) return;

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
            top_idx[p] = ivflist[j]; // 直接查表
        }
    }

    for(int i = 0; i < k; i++){
        q_res[i] = top_idx[i];
    }
}