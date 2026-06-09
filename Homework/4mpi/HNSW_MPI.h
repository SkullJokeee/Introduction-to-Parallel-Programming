#pragma once
#include <mpi.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cstring>
#include "hnsw.h"

hnswlib::HierarchicalNSW<float>* local_hnsw_idx = nullptr;
NEONSpace* local_hnsw_sp = nullptr;
float* local_base = nullptr;

void hnsw_mpi_setup(const float* global_base, size_t base_number, size_t vecdim, int rank, int nprocs){
    size_t base_cnt = base_number / nprocs;
    size_t rem = base_number % nprocs;
    
    // 计算当前进程负责的数据范围
    size_t local_start = rank * base_cnt + (rank < (int)rem ? rank : rem);
    size_t local_n = base_cnt + (rank < (int)rem ? 1 : 0);

    std::vector<int> sendcounts(nprocs, 0);
    std::vector<int> displs(nprocs, 0);

    if(rank == 0){
        for(int p = 0; p < nprocs; p++){
            size_t p_start = p * base_cnt + (p < (int)rem ? p : rem);
            size_t p_cnt = base_cnt + (p < (int)rem ? 1 : 0);
            sendcounts[p] = (int)(p_cnt * vecdim);
            displs[p] = (int)(p_start * vecdim);
        }
    }

    local_base = new float[local_n * vecdim];

    // 分发至各个节点
    MPI_Scatterv(global_base, sendcounts.data(), displs.data(), MPI_FLOAT, 
                 local_base, (int)(local_n * vecdim), MPI_FLOAT, 0, MPI_COMM_WORLD);

    local_hnsw_sp = new NEONSpace(vecdim);
    
    size_t M = 16;
    size_t efConstruction = 150;
    local_hnsw_idx = new hnswlib::HierarchicalNSW<float>(local_hnsw_sp, local_n, M, efConstruction);

    // 本地HNSW树
    for(size_t i = 0; i < local_n; i++){
        local_hnsw_idx->addPoint(local_base + i * vecdim, local_start + i);
    }
    
    local_hnsw_idx->setEf(200); 
}

std::priority_queue<std::pair<float, uint32_t>> hnsw_mpi_search(const float* query, size_t vecdim, size_t k, int rank, int nprocs) {
    std::vector<float> query_buf(vecdim);
    if (rank == 0){
        std::memcpy(query_buf.data(), query, vecdim * sizeof(float));
    }
    // 0号进程下发当前需要检索的query
    MPI_Bcast(query_buf.data(), (int)vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto ret = local_hnsw_idx->searchKnn(query_buf.data(), k);

    std::vector<float> local_dis(k, 1e30f);
    std::vector<uint32_t> local_ids(k, 0xFFFFFFFF);
    int idx = 0;
    while (!ret.empty()){
        if (idx < (int)k){
            local_dis[idx] = ret.top().first;
            local_ids[idx] = ret.top().second;
            idx++;
        }
        ret.pop();
    }

    std::vector<float> recv_dis;
    std::vector<uint32_t> recv_ids;
    if(rank == 0){
        recv_dis.resize(nprocs * k);
        recv_ids.resize(nprocs * k);
    }

    // 归并各个节点的 top-K
    MPI_Gather(local_dis.data(), (int)k, MPI_FLOAT, recv_dis.data(), (int)k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_ids.data(), (int)k, MPI_UINT32_T, recv_ids.data(), (int)k, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::priority_queue<std::pair<float, uint32_t>> rst_q;
    
    if(rank == 0){
        for (int i = 0; i < nprocs * (int)k; i++){
            if (recv_ids[i] == 0xFFFFFFFF) continue;
            
            if (rst_q.size() < k){
                rst_q.push({recv_dis[i], recv_ids[i]});
            } else if (recv_dis[i] < rst_q.top().first) {
                rst_q.pop();
                rst_q.push({recv_dis[i], recv_ids[i]});
            }
        }
    }

    return rst_q;
}

void hnsw_mpi_cleanup(){
    if (local_hnsw_idx){
        delete local_hnsw_idx;
        local_hnsw_idx = nullptr;
    }
    if (local_hnsw_sp){
        delete local_hnsw_sp;
        local_hnsw_sp = nullptr;
    }
    if (local_base){
        delete[] local_base;
        local_base = nullptr;
    }
}