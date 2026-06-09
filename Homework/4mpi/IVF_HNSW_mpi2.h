#pragma once
#include <mpi.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include "hnsw.h"

std::vector<hnswlib::HierarchicalNSW<float>*> cluster_hnsw_idx;
std::vector<NEONSpace*> cluster_hnsw_sp;

float* local_base_ivf = nullptr;
uint32_t* local_list_ivf = nullptr;
uint32_t* local_offset_ivf = nullptr;
size_t local_cluster_count = 0;

void ivf_hnsw_mpi_setup(const float* base_ivf, const uint32_t* offset_ivf, size_t cb_n, size_t dim, int rank, int nprocs, size_t M = 16, size_t efConstruction = 150){
    extern uint32_t* list_ivf;
    
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
            }else{
                p_data_cnt[p] += c_size;
            }
        }
        for(int p = 1; p < nprocs; p++){
            int int_cnt = (int)p_data_cnt[p];
            MPI_Send(&int_cnt, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
        }
    }else{
        int int_cnt = 0;
        MPI_Recv(&int_cnt, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_data_cnt = int_cnt;
    }

    local_list_ivf = new uint32_t[local_data_cnt];
    local_base_ivf = new float[local_data_cnt * dim];

    if(rank == 0){
        std::vector<size_t> p_cluster_idx(nprocs, 0);
        size_t r0_data_ptr = 0;
        for(size_t c = 0; c < cb_n; c++){
            int p = c % nprocs;
            size_t c_start = offset_ivf[c];
            size_t c_size = offset_ivf[c + 1] - c_start;
            if(p == 0){
                std::memcpy(local_list_ivf + r0_data_ptr, list_ivf + c_start, c_size * sizeof(uint32_t));
                std::memcpy(local_base_ivf + r0_data_ptr * dim, base_ivf + c_start * dim, c_size * dim * sizeof(float));
                r0_data_ptr += c_size;
                p_cluster_idx[0]++;
            }else{
                int c_size_int = (int)c_size;
                MPI_Send(&c_size_int, 1, MPI_INT, p, 2, MPI_COMM_WORLD);
                if(c_size > 0){
                    MPI_Send(list_ivf + c_start, c_size_int, MPI_UINT32_T, p, 3, MPI_COMM_WORLD);
                    MPI_Send(base_ivf + c_start * dim, c_size_int * (int)dim, MPI_FLOAT, p, 4, MPI_COMM_WORLD);
                }
            }
        }
    }else{
        size_t curr_data_ptr = 0;
        for(size_t c_idx = 0; c_idx < local_cluster_count; c_idx++){
            int c_size_int = 0;
            MPI_Recv(&c_size_int, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_offset_ivf[c_idx + 1] = local_offset_ivf[c_idx] + c_size_int;
            if(c_size_int > 0){
                MPI_Recv(local_list_ivf + curr_data_ptr, c_size_int, MPI_UINT32_T, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(local_base_ivf + curr_data_ptr * dim, c_size_int * (int)dim, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                curr_data_ptr += c_size_int;
            }
        }
    }

    cluster_hnsw_idx.resize(local_cluster_count, nullptr);
    cluster_hnsw_sp.resize(local_cluster_count, nullptr);
    
    for(size_t i = 0; i < local_cluster_count; i++){
        size_t start = local_offset_ivf[i];
        size_t end = local_offset_ivf[i + 1];
        size_t n = end - start;
        if(n == 0) continue;
        
        NEONSpace* sp = new NEONSpace(dim);
        hnswlib::HierarchicalNSW<float>* alg = new hnswlib::HierarchicalNSW<float>(sp, n, M, efConstruction);
        std::vector<float> tmp(dim);
        size_t t = (n / 4) * 4;
        for(size_t j = start; j < end; j++){
            size_t offset = j - start;
            if(offset < t){
                size_t b = offset / 4;
                size_t lane = offset % 4;
                for(size_t d = 0; d < dim; d++){
                    tmp[d] = local_base_ivf[start * dim + b * 4 * dim + d * 4 + lane];
                }
                alg->addPoint(tmp.data(), offset);
            }else{
                alg->addPoint(local_base_ivf + j * dim, offset);
            }
        }
        cluster_hnsw_sp[i] = sp;
        cluster_hnsw_idx[i] = alg;
    }
}

std::priority_queue<std::pair<float, uint32_t>> ivf_hnsw_mpi_search(const float* query, size_t cb_n, size_t dim, size_t k, int rank, int nprocs){
    std::vector<float> query_buf(dim);
    if(rank == 0) std::copy(query, query + dim, query_buf.data());
    MPI_Bcast(query_buf.data(), (int)dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    std::priority_queue<std::pair<float, uint32_t>> local_rst;
    for(size_t i = 0; i < local_cluster_count; i++){
        if(cluster_hnsw_idx[i] == nullptr) continue;
        auto ret = cluster_hnsw_idx[i]->searchKnn(query_buf.data(), k);
        while(!ret.empty()){
            auto top = ret.top();
            uint32_t label = local_list_ivf[local_offset_ivf[i] + top.second];
            if(local_rst.size() < k) local_rst.push({top.first, label});
            else if(top.first < local_rst.top().first){
                local_rst.pop();
                local_rst.push({top.first, label});
            }
            ret.pop();
        }
    }

    std::vector<float> send_dis(k, 1e30f);
    std::vector<uint32_t> send_ids(k, 0xFFFFFFFF);
    int idx = (int)k - 1;
    while(!local_rst.empty() && idx >= 0){
        send_dis[idx] = local_rst.top().first;
        send_ids[idx] = local_rst.top().second;
        local_rst.pop();
        idx--;
    }

    std::vector<float> all_dis(nprocs * k);
    std::vector<uint32_t> all_ids(nprocs * k);
    MPI_Gather(send_dis.data(), (int)k, MPI_FLOAT, all_dis.data(), (int)k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(send_ids.data(), (int)k, MPI_UINT32_T, all_ids.data(), (int)k, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::priority_queue<std::pair<float, uint32_t>> final_q;
    if(rank == 0){
        for(int i = 0; i < nprocs * (int)k; i++){
            if(all_ids[i] == 0xFFFFFFFF) continue;
            if(final_q.size() < k) final_q.push({all_dis[i], all_ids[i]});
            else if(all_dis[i] < final_q.top().first){
                final_q.pop();
                final_q.push({all_dis[i], all_ids[i]});
            }
        }
    }
    return final_q;
}

void ivf_hnsw_mpi_cleanup(){
    for(size_t i = 0; i < cluster_hnsw_idx.size(); i++){
        if(cluster_hnsw_idx[i]) delete cluster_hnsw_idx[i];
        if(cluster_hnsw_sp[i]) delete cluster_hnsw_sp[i];
    }
    delete[] local_base_ivf;
    delete[] local_list_ivf;
    delete[] local_offset_ivf;
    local_base_ivf = nullptr;
    local_list_ivf = nullptr;
    local_offset_ivf = nullptr;
}