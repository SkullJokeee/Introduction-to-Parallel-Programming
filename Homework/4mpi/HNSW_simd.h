#pragma once
#include <queue>
#include <fstream>
#include <vector>
#include "hnsw.h"

hnswlib::HierarchicalNSW<float>* idx = nullptr;
NEONSpace* sp = nullptr;

void LoadHNSWData(size_t dim, const char* path = "files/hnsw.index"){
    sp = new NEONSpace(dim);
    idx = new hnswlib::HierarchicalNSW<float>(sp, path);
    idx->setEf(200);
    std::cerr << "HNSW index loaded from " << path << "\n";
}

std::priority_queue<std::pair<float, uint32_t>> hnsw_search(const float* base, const float* query, size_t n, size_t dim, size_t k){
    auto ret = idx->searchKnn(query, k);
    
    std::vector<std::pair<float, uint32_t>> temp_vec;
    temp_vec.reserve(ret.size());
    
    while(!ret.empty()){
        auto top = ret.top();
        temp_vec.push_back({top.first, (uint32_t)top.second});
        ret.pop();
    }

    return std::priority_queue<std::pair<float, uint32_t>>(std::less<std::pair<float, uint32_t>>(), std::move(temp_vec));
}
