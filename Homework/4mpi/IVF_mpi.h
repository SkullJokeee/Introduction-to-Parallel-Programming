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
size_t local_start_cluster = 0;
size_t local_cluster_count = 0;

// 按聚类块分发base到各进程
void ivf_mpi_setup(float*& base_ivf, uint32_t*& list_ivf, uint32_t*& offset_ivf, size_t cb_n, size_t ivf_dim, int rank, int nprocs){
    size_t base_cnt = cb_n / nprocs;
    size_t rem = cb_n % nprocs;
    local_start_cluster = rank * base_cnt + (rank < (int)rem ? rank : rem);
    local_cluster_count = base_cnt + (rank < (int)rem ? 1 : 0);

    std::vector<int> sendcounts_off(nprocs, 0);
    std::vector<int> displs_off(nprocs, 0);
    std::vector<int> sendcounts_list(nprocs, 0);
    std::vector<int> displs_list(nprocs, 0);
    std::vector<int> sendcounts_base(nprocs, 0);
    std::vector<int> displs_base(nprocs, 0);

    if(rank == 0){
        for(int p = 0; p < nprocs; p++){
            size_t p_start = p * base_cnt + (p < (int)rem ? p : rem);
            size_t p_cnt = base_cnt + (p < (int)rem ? 1 : 0);
            size_t p_data_start = offset_ivf[p_start];
            size_t p_data_end = offset_ivf[p_start + p_cnt];
            size_t p_data_cnt = p_data_end - p_data_start;

            sendcounts_off[p] = (int)(p_cnt + 1);
            displs_off[p] = (int)p_start;

            sendcounts_list[p] = (int)p_data_cnt;
            displs_list[p] = (int)p_data_start;

            sendcounts_base[p] = (int)(p_data_cnt * ivf_dim);
            displs_base[p] = (int)(p_data_start * ivf_dim);
        }
    }

    int local_list_sz = 0;
    MPI_Scatter(rank == 0 ? sendcounts_list.data() : nullptr, 1, MPI_INT, &local_list_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    size_t local_data_cnt = local_list_sz;

    local_offset_ivf = new uint32_t[local_cluster_count + 1];
    local_list_ivf = new uint32_t[local_data_cnt];
    local_base_ivf = new float[local_data_cnt * ivf_dim];

    MPI_Scatterv(offset_ivf, sendcounts_off.data(), displs_off.data(), MPI_UINT32_T, local_offset_ivf, (int)(local_cluster_count + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(list_ivf, sendcounts_list.data(), displs_list.data(), MPI_UINT32_T, local_list_ivf, (int)local_data_cnt, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(base_ivf, sendcounts_base.data(), displs_base.data(), MPI_FLOAT, local_base_ivf, (int)(local_data_cnt * ivf_dim), MPI_FLOAT, 0, MPI_COMM_WORLD);

    uint32_t base_shift = local_offset_ivf[0];
    for(size_t i = 0; i <= local_cluster_count; i++){
        local_offset_ivf[i] -= base_shift;
    }
}

// 各进程 SIMD 计算 query 与部分聚类中心的距离，Allgather 后确定全局最近 nprobe 个聚类，再 SIMD 搜索本地聚类内 top-k
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
    size_t blocks_per_rank = total_blocks / nprocs;
    size_t rem_blocks = total_blocks % nprocs;
    size_t cb_start = (rank * blocks_per_rank + (rank < (int)rem_blocks ? rank : rem_blocks)) * 4;
    size_t cb_end = cb_start + (blocks_per_rank + (rank < (int)rem_blocks ? 1 : 0)) * 4;

    for(size_t i = cb_start; i < cb_end; i += 4){
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
        }
        else if(d1 < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({d1, (uint32_t)i});
        }

        if(centroid_q.size() < nprobe){
            centroid_q.push({d2, (uint32_t)(i+1)});
        }
        else if(d2 < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({d2, (uint32_t)(i+1)});
        }

        if(centroid_q.size() < nprobe){
            centroid_q.push({d3, (uint32_t)(i+2)});
        }
        else if(d3 < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({d3, (uint32_t)(i+2)});
        }

        if(centroid_q.size() < nprobe){
            centroid_q.push({d4, (uint32_t)(i+3)});
        }
        else if(d4 < centroid_q.top().first){
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
        }
        else if(all_dis[i] < centroid_q.top().first){
            centroid_q.pop();
            centroid_q.push({all_dis[i], all_ids[i]});
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> local_rst;

    while(!centroid_q.empty()){
        uint32_t global_cid = centroid_q.top().second;
        centroid_q.pop();

        if(global_cid < local_start_cluster || global_cid >= local_start_cluster + local_cluster_count){
            continue;
        }

        uint32_t local_cid = global_cid - local_start_cluster;
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
                }
                else if(dis < local_rst.top().first){
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
            }
            else if(dis < local_rst.top().first){
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
            }
            else if(recv_dis[i] < rst_q.top().first){
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