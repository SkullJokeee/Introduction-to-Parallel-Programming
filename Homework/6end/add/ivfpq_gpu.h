#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// 计算所有查询与 IVF 聚类中心的距离
__global__ void coarse_sgemm_kernel(const float* codebook, const float* query, float* dist, int nlist, int num_query, int vecdim){
    __shared__ float s_query[16][17];
    __shared__ float s_codebook[16][17];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float sum = 0.0f;
    int num_tiles = (vecdim + 15) / 16;

    for(int t = 0; t < num_tiles; t++){
        if(row < num_query && (t * 16 + tx) < vecdim){
            s_query[ty][tx] = query[row * vecdim + t * 16 + tx];
        }else{
            s_query[ty][tx] = 0.0f;
        }

        int cb_row = bx * 16 + ty;
        int cb_col = t * 16 + tx;
        if(cb_row < nlist && cb_col < vecdim){
            s_codebook[ty][tx] = codebook[cb_row * vecdim + cb_col];
        }else{
            s_codebook[ty][tx] = 0.0f;
        }

        __syncthreads();
        for(int step = 0; step < 16; step++){
            sum += s_query[ty][step] * s_codebook[tx][step];
        }
        __syncthreads();
    }

    if(row < num_query && col < nlist){
        dist[row * nlist + col] = 1.0f - sum;
    }
}

// 为每个查询选择距离最近的 nprobe 个簇
__global__ void coarse_select_kernel(const float* dist, int* top_clusters, int num_query, int nlist, int nprobe){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= num_query){
        return;
    }

    const float* q_dist = dist + q_idx * nlist;
    int* q_res = top_clusters + q_idx * nprobe;

    float top_val[40];
    int top_idx[40];
    for(int i = 0; i < nprobe; i++){
        top_val[i] = 1e9f;
        top_idx[i] = -1;
    }

    for(int j = 0; j < nlist; j++){
        float val = q_dist[j];
        if(val < top_val[nprobe - 1]){
            int pos = nprobe - 1;
            while(pos > 0 && val < top_val[pos - 1]){
                top_val[pos] = top_val[pos - 1];
                top_idx[pos] = top_idx[pos - 1];
                pos--;
            }
            top_val[pos] = val;
            top_idx[pos] = j;
        }
    }

    for(int i = 0; i < nprobe; i++){
        q_res[i] = top_idx[i];
    }
}

// 为每个查询的选中簇构建 PQ 查找表
__global__ void build_lut_kernel(const float* query, const float* pq_cbs, float* luts, const int* top_clusters, int num_query, int nprobe, int m, int k_pq, int sub_d){
    int q_idx = blockIdx.x;
    int cluster_idx = blockIdx.y;
    int tid = threadIdx.x;

    if(q_idx >= num_query || cluster_idx >= nprobe){
        return;
    }

    int c = top_clusters[q_idx * nprobe + cluster_idx];

    float* lut = luts + (q_idx * nprobe + cluster_idx) * m * k_pq;
    const float* q_seg = query + q_idx * m * sub_d;

    for(int j = tid; j < m * k_pq; j += blockDim.x){
        int sub = j / k_pq;
        int code = j % k_pq;

        const float* center = pq_cbs + (c * m * k_pq + sub * k_pq + code) * sub_d;
        const float* q_sub = q_seg + sub * sub_d;

        float ip = 0.0f;
        for(int d = 0; d < sub_d; d++){
            ip += q_sub[d] * center[d];
        }

        lut[j] = 1.0f - ip;
    }
}

// 在选中簇内 PQ 查表得到近似距离
__global__ void adc_lookup_kernel(const uint8_t* base_pq, const float* luts, float* cluster_dists, const uint32_t* offset, const int* top_clusters, int num_query, int nprobe, int m, int k_pq, int max_cluster_size, int cluster_idx_in_nprobe){
    int q_idx = blockIdx.x;
    int vec_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if(q_idx >= num_query){
        return;
    }

    int c = top_clusters[q_idx * nprobe + cluster_idx_in_nprobe];

    uint32_t start = offset[c];
    uint32_t end = offset[c + 1];
    uint32_t cluster_size = end - start;

    if(vec_idx >= cluster_size){
        return;
    }

    uint32_t i = start + vec_idx;
    const uint8_t* idx = base_pq + i * m;

    const float* lut = luts + (q_idx * nprobe + cluster_idx_in_nprobe) * m * k_pq;
    const float* lut0 = lut;
    const float* lut1 = lut + k_pq;
    const float* lut2 = lut + 2 * k_pq;
    const float* lut3 = lut + 3 * k_pq;

    float d = lut0[idx[0]] + lut1[idx[1]] + lut2[idx[2]] + lut3[idx[3]];
    cluster_dists[q_idx * max_cluster_size + vec_idx] = d;
}

// 合并多簇候选并维护每个查询的 top-P
__global__ void merge_clusters_kernel(const float* cluster_dists, const uint32_t* lst, const uint32_t* offset, const int* top_clusters, float* top_vals, uint32_t* top_idxs, int num_query, int nprobe, int p, int max_cluster_size, int cluster_idx_in_nprobe){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= num_query){
        return;
    }

    float* t_val = top_vals + q_idx * p;
    uint32_t* t_idx = top_idxs + q_idx * p;

    int c = top_clusters[q_idx * nprobe + cluster_idx_in_nprobe];

    uint32_t start = offset[c];
    uint32_t end = offset[c + 1];
    uint32_t cluster_size = end - start;

    const float* c_dist = cluster_dists + q_idx * max_cluster_size;

    for(uint32_t j = 0; j < cluster_size; j++){
        float val = c_dist[j];
        uint32_t orig_idx = lst[start + j];

        if(val < t_val[p - 1]){
            int pos = p - 1;
            while(pos > 0 && val < t_val[pos - 1]){
                t_val[pos] = t_val[pos - 1];
                t_idx[pos] = t_idx[pos - 1];
                pos--;
            }
            t_val[pos] = val;
            t_idx[pos] = orig_idx;
        }
    }
}

// 对 top-P 候选重算精确距离得到 top-K
__global__ void rerank_kernel(const float* base, const float* query, const uint32_t* candidates, float* final_vals, uint32_t* final_idxs, int num_query, int p, int top_k, int vecdim){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= num_query){
        return;
    }

    const float* q = query + q_idx * vecdim;
    const uint32_t* cands = candidates + q_idx * p;
    float* f_val = final_vals + q_idx * top_k;
    uint32_t* f_idx = final_idxs + q_idx * top_k;

    for(int i = 0; i < p; i++){
        uint32_t idx = cands[i];
        if(idx == 0xFFFFFFFFu){
            continue;
        }

        const float* b = base + idx * vecdim;
        float ip = 0.0f;
        for(int d = 0; d < vecdim; d++){
            ip += q[d] * b[d];
        }
        float dist = 1.0f - ip;

        if(dist < f_val[top_k - 1]){
            int pos = top_k - 1;
            while(pos > 0 && dist < f_val[pos - 1]){
                f_val[pos] = f_val[pos - 1];
                f_idx[pos] = f_idx[pos - 1];
                pos--;
            }
            f_val[pos] = dist;
            f_idx[pos] = idx;
        }
    }
}

// 初始化 top-P 队列为极大值
__global__ void init_topP_kernel(float* top_vals, uint32_t* top_idxs, int num_query, int p){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= num_query){
        return;
    }
    for(int i = 0; i < p; i++){
        top_vals[q_idx * p + i] = 1e9f;
        top_idxs[q_idx * p + i] = 0xFFFFFFFFu;
    }
}

// 初始化 top-K 队列为极大值
__global__ void init_topK_kernel(float* final_vals, uint32_t* final_idxs, int num_query, int top_k){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_idx >= num_query){
        return;
    }
    for(int i = 0; i < top_k; i++){
        final_vals[q_idx * top_k + i] = 1e9f;
        final_idxs[q_idx * top_k + i] = 0xFFFFFFFFu;
    }
}
