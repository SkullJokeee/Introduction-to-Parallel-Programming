#pragma once
#include <mpi.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "simd.h"
#include <cstdint>

float* local_base_ivf = nullptr;
uint32_t* local_list_ivf = nullptr;
uint32_t* local_offset_ivf = nullptr;
size_t local_cluster_count = 0;


void ivf_mpi_setup(float*& base_ivf, uint32_t*& list_ivf, uint32_t*& offset_ivf, size_t cb_n, size_t ivf_dim, int rank, int nprocs){
    local_cluster_count = cb_n / nprocs + (rank < (int)(cb_n % nprocs) ? 1 : 0);
    local_offset_ivf = new uint32_t[local_cluster_count + 1];
    local_offset_ivf[0] = 0;

    size_t local_data_cnt = 0;

    if(rank == 0){
        std::vector<size_t> p_data_cnt(nprocs, 0);
        std::vector<size_t> p_cluster_idx(nprocs, 0);

        for(size_t c = 0; c < cb_n; c++){
            int p = c % nprocs;
            
            size_t c_size = offset_ivf[c + 1] - offset_ivf[c];
            
            if(p == 0){
                local_offset_ivf[p_cluster_idx[0] + 1] = local_offset_ivf[p_cluster_idx[0]] + c_size;
                p_cluster_idx[0]++;
                
                local_data_cnt += c_size;
            }
            else{
                p_data_cnt[p] += c_size;
            }
        }
        
        for(int p = 1; p < nprocs; p++){
            int int_cnt = (int)p_data_cnt[p];
            MPI_Send(&int_cnt, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
        }

    }
    else{
        int int_cnt = 0;
        MPI_Recv(&int_cnt, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        local_data_cnt = int_cnt;
    }
    
    local_list_ivf = new uint32_t[local_data_cnt];
    local_base_ivf = new float[local_data_cnt * ivf_dim];
    
    if(rank == 0){
        std::vector<size_t> p_cluster_idx(nprocs, 0);
        size_t r0_data_ptr = 0;
    
        for(size_t c = 0; c < cb_n; c++){
            int p = c % nprocs;
            size_t c_start = offset_ivf[c];
            size_t c_size = offset_ivf[c + 1] - c_start;
    
            if(p == 0){
                std::memcpy(local_list_ivf + r0_data_ptr, list_ivf + c_start, c_size * sizeof(uint32_t));
                std::memcpy(local_base_ivf + r0_data_ptr * ivf_dim, base_ivf + c_start * ivf_dim, c_size * ivf_dim * sizeof(float));
                r0_data_ptr += c_size;
                p_cluster_idx[0]++;
            }
            else{
                
                int c_size_int = (int)c_size;
                MPI_Send(&c_size_int, 1, MPI_INT, p, 2, MPI_COMM_WORLD);
                
                if(c_size > 0){
                    MPI_Send(list_ivf + c_start, c_size_int, MPI_UINT32_T, p, 3, MPI_COMM_WORLD);
                    MPI_Send(base_ivf + c_start * ivf_dim, c_size_int * (int)ivf_dim, MPI_FLOAT, p, 4, MPI_COMM_WORLD);
                }
            }
        }
    }
    else{
        size_t curr_data_ptr = 0;
        
        for(size_t c_idx = 0; c_idx < local_cluster_count; c_idx++){
            int c_size_int = 0;
            MPI_Recv(&c_size_int, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
            local_offset_ivf[c_idx + 1] = local_offset_ivf[c_idx] + c_size_int;
        
            if(c_size_int > 0){
                MPI_Recv(local_list_ivf + curr_data_ptr, c_size_int, MPI_UINT32_T, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(local_base_ivf + curr_data_ptr * ivf_dim, c_size_int * (int)ivf_dim, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                curr_data_ptr += c_size_int;
            }
        }
    }
}

// 各进程轮询计算部分聚类中心距离，Allgather归并后搜索本地所属聚类top-k
std::priority_queue<std::pair<float, uint32_t>> ivf_mpi_search(const float* query, size_t cb_n, size_t ivf_dim, size_t k, const float* codebook_ivf, int rank, int nprocs){
    size_t nprobe = 20;
    nprobe = std::min(nprobe, cb_n);
    std::vector<float> query_buf(ivf_dim);
    if(rank == 0){
        std::memcpy(query_buf.data(), query, ivf_dim * sizeof(float));
    }
    MPI_Bcast(query_buf.data(), ivf_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    std::vector<float32x4_t> v_query(ivf_dim);
    for(size_t d = 0; d < ivf_dim; ++d){
        v_query[d] = vdupq_n_f32(query_buf[d]);
    }
    std::priority_queue<std::pair<float, uint32_t>> centroid_q;
    size_t total_blocks = cb_n / 4;
    for(size_t b = rank; b < total_blocks; b += nprocs){
        size_t i = b * 4;
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);
        const float* block = codebook_ivf + i * ivf_dim;
        for(size_t d = 0; d < ivf_dim; d += 4){
            float32x4_t c_vec1 = vld1q_f32(block + d * 4);
            float32x4_t c_vec2 = vld1q_f32(block + (d+1) * 4);
            float32x4_t c_vec3 = vld1q_f32(block + (d+2) * 4);
            float32x4_t c_vec4 = vld1q_f32(block + (d+3) * 4);
            sum1 = vmlaq_f32(sum1, v_query[d], c_vec1);
            sum2 = vmlaq_f32(sum2, v_query[d+1], c_vec2);
            sum3 = vmlaq_f32(sum3, v_query[d+2], c_vec3);
            sum4 = vmlaq_f32(sum4, v_query[d+3], c_vec4);
        }
        float32x4_t sum12 = vaddq_f32(sum1, sum2);
        float32x4_t sum34 = vaddq_f32(sum3, sum4);
        float32x4_t sum = vaddq_f32(sum12, sum34);
        float dis_array[4];
        vst1q_f32(dis_array, sum);
        float d1 = 1.0f - dis_array[0];
        float d2 = 1.0f - dis_array[1];
        float d3 = 1.0f - dis_array[2];
        float d4 = 1.0f - dis_array[3];
        if(centroid_q.size() < nprobe){
            centroid_q.push({d1, (uint32_t)i});
        }else if(d1 < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({d1, (uint32_t)i});
        }
        if(centroid_q.size() < nprobe){
            centroid_q.push({d2, (uint32_t)(i+1)});
        }else if(d2 < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({d2, (uint32_t)(i+1)});
        }
        if(centroid_q.size() < nprobe){
            centroid_q.push({d3, (uint32_t)(i+2)});
        }else if(d3 < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({d3, (uint32_t)(i+2)});
        }
        if(centroid_q.size() < nprobe){
            centroid_q.push({d4, (uint32_t)(i+3)});
        }else if(d4 < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({d4, (uint32_t)(i+3)});
        }
    }
    std::vector<float> local_dis(nprobe, 1e30f);
    std::vector<uint32_t> local_ids(nprobe, 0xFFFFFFFF);
    int idx = (int)nprobe - 1;
    while(!centroid_q.empty() && idx >= 0){
        local_dis[idx] = centroid_q.top().first;
        local_ids[idx] = centroid_q.top().second;
        centroid_q.pop();
        idx--;
    }
    std::vector<float> all_dis(nprocs * nprobe);
    std::vector<uint32_t> all_ids(nprocs * nprobe);
    MPI_Allgather(local_dis.data(), (int)nprobe, MPI_FLOAT, all_dis.data(), (int)nprobe, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(local_ids.data(), (int)nprobe, MPI_UINT32_T, all_ids.data(), (int)nprobe, MPI_UINT32_T, MPI_COMM_WORLD);
    for(size_t i = 0; i < nprocs * nprobe; i++){
        if(all_ids[i] == 0xFFFFFFFF) continue;
        if(centroid_q.size() < nprobe){
            centroid_q.push({all_dis[i], all_ids[i]});
        }else if(all_dis[i] < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({all_dis[i], all_ids[i]});
        }
    }
    std::priority_queue<std::pair<float, uint32_t>> local_rst;
    while(!centroid_q.empty()){
        uint32_t global_cid = centroid_q.top().second;
        centroid_q.pop();
        if(global_cid % nprocs != rank){
            continue;
        }
        uint32_t local_cid = global_cid / nprocs;
        uint32_t start = local_offset_ivf[local_cid];
        uint32_t end = local_offset_ivf[local_cid + 1];
        uint32_t last = start + ((end - start) / 4) * 4;
        for(uint32_t i = start; i < last; i += 4){
            __builtin_prefetch(local_base_ivf + (i + 8) * ivf_dim, 0, 1);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);
            const float* block = local_base_ivf + i * ivf_dim;
            for(size_t d = 0; d < ivf_dim; d += 4){
                float32x4_t b_vec1 = vld1q_f32(block + d*4);
                float32x4_t b_vec2 = vld1q_f32(block + (d+1)*4);
                float32x4_t b_vec3 = vld1q_f32(block + (d+2)*4);
                float32x4_t b_vec4 = vld1q_f32(block + (d+3)*4);
                sum1 = vmlaq_f32(sum1, v_query[d], b_vec1);
                sum2 = vmlaq_f32(sum2, v_query[d+1], b_vec2);
                sum3 = vmlaq_f32(sum3, v_query[d+2], b_vec3);
                sum4 = vmlaq_f32(sum4, v_query[d+3], b_vec4);
            }
            float32x4_t sum12 = vaddq_f32(sum1, sum2);
            float32x4_t sum34 = vaddq_f32(sum3, sum4);
            float32x4_t sum = vaddq_f32(sum12, sum34);
            float dis_array[4];
            vst1q_f32(dis_array, sum);
            for(int j = 0; j < 4; j++){
                float dis = 1.0f - dis_array[j];
                uint32_t l = local_list_ivf[i + j];
                if(local_rst.size() < k){
                    local_rst.push({dis, l});
                }else if(dis < local_rst.top().first){
                    local_rst.pop();
                    local_rst.push({dis, l});
                }
            }
        }
        for(uint32_t i = last; i < end; i++){
            const float* current = local_base_ivf + i * ivf_dim;
            float dis = InnerProductSIMDNeon(current, query_buf.data(), ivf_dim);
            uint32_t l = local_list_ivf[i];
            if(local_rst.size() < k){
                local_rst.push({dis, l});
            }else if(dis < local_rst.top().first){
                local_rst.pop();
                local_rst.push({dis, l});
            }
        }
    }
    std::vector<float> send_dis(k, 1e30f);
    std::vector<uint32_t> send_ids(k, 0xFFFFFFFF);
    int local_sz = 0;
    while(!local_rst.empty()){
        if(local_sz < (int)k){
            send_dis[local_sz] = local_rst.top().first;
            send_ids[local_sz] = local_rst.top().second;
            local_sz++;
        }
        local_rst.pop();
    }
    std::vector<float> recv_dis;
    std::vector<uint32_t> recv_ids;
    if(rank == 0){
        recv_dis.resize(nprocs * k);
        recv_ids.resize(nprocs * k);
    }
    MPI_Gather(send_dis.data(), (int)k, MPI_FLOAT, recv_dis.data(), (int)k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(send_ids.data(), (int)k, MPI_UINT32_T, recv_ids.data(), (int)k, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::priority_queue<std::pair<float, uint32_t>> rst_q;
    if(rank == 0){
        for(int i = 0; i < nprocs * (int)k; i++){
            if(recv_ids[i] == 0xFFFFFFFF){
                continue;
            }
            if(rst_q.size() < k){
                rst_q.push({recv_dis[i], recv_ids[i]});
            }else if(recv_dis[i] < rst_q.top().first){
                rst_q.pop();
                rst_q.push({recv_dis[i], recv_ids[i]});
            }
        }
    }
    return rst_q;
}

void ivf_mpi_cleanup(){
    delete[] local_base_ivf;
    delete[] local_list_ivf;
    delete[] local_offset_ivf;
    local_base_ivf = nullptr;
    local_list_ivf = nullptr;
    local_offset_ivf = nullptr;
}