#pragma once
#include <mpi.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <omp.h>
#include "hnsw.h"

std::vector<hnswlib::HierarchicalNSW<float>*> cluster_hnsw_idx;
std::vector<NEONSpace*> cluster_hnsw_sp;

float* local_base_ivf = nullptr;
uint32_t* local_list_ivf = nullptr;
uint32_t* local_offset_ivf = nullptr;
size_t local_start_cluster = 0;
size_t local_cluster_count = 0;

void ivf_hnsw_mpi_setup(const float* base_ivf, const uint32_t* offset_ivf, size_t cb_n, size_t dim, int rank, int nprocs, size_t M = 16, size_t efConstruction = 150){
    extern uint32_t* list_ivf;
    
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
            sendcounts_base[p] = (int)(p_data_cnt * dim);
            displs_base[p] = (int)(p_data_start * dim);
        }
    }

    int local_list_sz = 0;
    MPI_Scatter(rank == 0 ? sendcounts_list.data() : nullptr, 1, MPI_INT, &local_list_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    size_t local_data_cnt = local_list_sz;

    local_offset_ivf = new uint32_t[local_cluster_count + 1];
    local_list_ivf = new uint32_t[local_data_cnt];
    local_base_ivf = new float[local_data_cnt * dim];

    MPI_Scatterv(offset_ivf, sendcounts_off.data(), displs_off.data(), MPI_UINT32_T, 
                 local_offset_ivf, (int)(local_cluster_count + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(list_ivf, sendcounts_list.data(), displs_list.data(), MPI_UINT32_T, 
                 local_list_ivf, (int)local_data_cnt, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(base_ivf, sendcounts_base.data(), displs_base.data(), MPI_FLOAT, 
                 local_base_ivf, (int)(local_data_cnt * dim), MPI_FLOAT, 0, MPI_COMM_WORLD);

    uint32_t base_shift = local_offset_ivf[0];

    #pragma omp parallel for  ////
    for(size_t i = 0; i <= local_cluster_count; i++){
        local_offset_ivf[i] -= base_shift;
    }

    cluster_hnsw_idx.resize(local_cluster_count, nullptr);
    cluster_hnsw_sp.resize(local_cluster_count, nullptr);

    #pragma omp parallel for schedule(dynamic, 1)
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

// OpenMP 并行搜索本地各聚类 HNSW，再 Gather 到 0 号进程归并全局 top-k
std::priority_queue<std::pair<float, uint32_t>> ivf_hnsw_mpi_search(const float* query, size_t cb_n, size_t dim, size_t k, int rank, int nprocs){
    std::vector<float> query_buf(dim);
    if(rank == 0) std::copy(query, query + dim, query_buf.data());
    MPI_Bcast(query_buf.data(), (int)dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    std::priority_queue<std::pair<float, uint32_t>> local_rst;
    
    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, uint32_t>> local_rst_thread;
        
        #pragma omp for schedule(dynamic, 1) nowait
        for(size_t i = 0; i < local_cluster_count; i++){
            if(cluster_hnsw_idx[i] == nullptr) continue;
            
            auto ret = cluster_hnsw_idx[i]->searchKnn(query_buf.data(), k);
            while(!ret.empty()){
                auto top = ret.top();
                uint32_t label = local_list_ivf[local_offset_ivf[i] + top.second];
                if(local_rst_thread.size() < k) local_rst_thread.push({top.first, label});
                else if(top.first < local_rst_thread.top().first){
                    local_rst_thread.pop();
                    local_rst_thread.push({top.first, label});
                }
                ret.pop();
            }
        }
        
        #pragma omp critical
        {
            while(!local_rst_thread.empty()){
                if(local_rst.size() < k) local_rst.push(local_rst_thread.top());
                else if(local_rst_thread.top().first < local_rst.top().first){
                    local_rst.pop();
                    local_rst.push(local_rst_thread.top());
                }
                local_rst_thread.pop();
            }
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