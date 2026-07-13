#pragma once
#include <queue>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include "hnswlib/hnswlib/hnswlib.h"

static hnswlib::InnerProductSpace* hnsw_space = nullptr;
static hnswlib::HierarchicalNSW<float>* hnsw_idx = nullptr;

void build_hnsw_index(const float* base, size_t base_number, size_t vecdim, const char* path = "files/hnsw.index", size_t M = 16, size_t efConstruction = 150){
    hnsw_space = new hnswlib::InnerProductSpace(vecdim);
    hnsw_idx = new hnswlib::HierarchicalNSW<float>(hnsw_space, base_number, M, efConstruction);

    hnsw_idx->addPoint(base, 0);
    #pragma omp parallel for
    for(size_t i = 1; i < base_number; ++i){
        hnsw_idx->addPoint(base + i * vecdim, i);
    }

    hnsw_idx->saveIndex(path);
    std::cerr << "HNSW index saved to " << path << "\n";
}

void load_hnsw_index(size_t vecdim, const char* path = "files/hnsw.index", size_t efSearch = 200){
    hnsw_space = new hnswlib::InnerProductSpace(vecdim);
    hnsw_idx = new hnswlib::HierarchicalNSW<float>(hnsw_space, path);
    hnsw_idx->setEf(efSearch);
    std::cerr << "HNSW index loaded from " << path << "\n";
}

void free_hnsw_index(){
    if(hnsw_idx){
        delete hnsw_idx;
        hnsw_idx = nullptr;
    }
    if(hnsw_space){
        delete hnsw_space;
        hnsw_space = nullptr;
    }
}

std::priority_queue<std::pair<float, uint32_t>> hnsw_serial_search(const float* query, size_t vecdim, size_t k){
    auto ret = hnsw_idx->searchKnn(query, k);

    std::vector<std::pair<float, uint32_t>> temp;
    temp.reserve(ret.size());
    while(!ret.empty()){
        auto top = ret.top();
        temp.push_back({top.first, (uint32_t)top.second});
        ret.pop();
    }

    return std::priority_queue<std::pair<float, uint32_t>>(std::less<std::pair<float, uint32_t>>(), std::move(temp));
}
