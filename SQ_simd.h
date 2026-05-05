#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include "simd.h"
#include "flat_simd.h"

struct SQIndex {
    std::vector<uint8_t> vecBase;
    std::vector<float> mins;
    std::vector<float> maxs;
};

SQIndex build_sq_index(float* base, size_t base_number, size_t vecdim) {
    SQIndex idx;
    idx.vecBase.resize(base_number * vecdim);
    idx.mins.resize(vecdim);
    idx.maxs.resize(vecdim);

    for(int d = 0; d < vecdim; ++d) {
        float min_val = base[d], max_val = base[d];
        for(int i = 0; i < base_number; ++i) {
            float val = base[d + i * vecdim];
            if(val < min_val) min_val = val;
            if(val > max_val) max_val = val;
        }
        idx.mins[d] = min_val;
        idx.maxs[d] = max_val;
    }

    for(int i = 0; i < base_number; ++i) {
        for(int d = 0; d < vecdim; ++d) {
            idx.vecBase[d + i * vecdim] = scalarQuantization(base[d + i * vecdim], idx.mins[d], idx.maxs[d]);
        }
    }
    
    return idx;
}

std::priority_queue<std::pair<float, uint32_t>> sq_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k,const SQIndex& sq_idx){    
    // std::priority_queue<std::pair<float, uint32_t>> q;
    std::priority_queue<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>, std::greater<std::pair<uint32_t, uint32_t>>> q;

    // std::vector<uint8_t> vecBase(base_number * vecdim);
    std::vector<uint8_t> vecQ(vecdim);

    size_t p = 1000;
    p = std::max(p,k);

    // for(int d = 0; d < vecdim; ++d) {
    //     float min = base[d], max = base[d];
    //     for(int i = 0; i < base_number; ++i) {
    //         if(base[d + i*vecdim] < min){
    //             min = base[d + i*vecdim];
    //         }
    //         if(base[d + i*vecdim] > max){
    //             max = base[d + i*vecdim];
    //         }
    //     }
    //     for(int i = 0; i < base_number; ++i) {
    //         vecBase[d + i*vecdim] = scalarQuantization(base[d + i*vecdim], min, max);
    //     }
    //     vecQ[d] = scalarQuantization(query[d], min, max);
    // }

    for(int d = 0; d < vecdim; ++d) {
        vecQ[d] = scalarQuantization(query[d], sq_idx.mins[d], sq_idx.maxs[d]);
    }

    // for(int i = 0; i < base_number; i++) {
    //     uint8_t* arrB = vecBase.data();
    //     uint8_t* arrQ = vecQ.data();
        
    //     const uint8_t* current = arrB + i * vecdim;
        
    //     float dis = 0.0f;
    //     dis = InnerProductSIMDNeon(current, arrQ, vecdim);

    //     if(q.size() < p) {
    //         q.push({dis, i});
    //     }
    //     else {
    //         if(dis < q.top().first) {
    //             q.pop(); 
    //             q.push({dis, i});
    //         }
    //     }
    // }

    uint8_t* arrB = sq_idx.vecBase.data();
    uint8_t* arrQ = vecQ.data();

    for(int i = 0; i < base_number; i++) {
        const uint8_t* current = arrB + i * vecdim;
        uint32_t dot = InnerProductSIMDNeon(current, arrQ, vecdim);

        if(q.size() < p) {
            q.push({dot, i});
        }
        else {
            if(dot > q.top().first) {
                q.pop(); 
                q.push({dot, i});
            }
        }
    }

    std::vector<float> tempRst(p * vecdim);

    for(int i = 0; i < p; i ++){
        int idx = q.top().second;
        for(int j = 0; j < vecdim; j++){
            tempRst[i * vecdim + j] = base[idx * vecdim + j];
        }
        q.pop();
    }

    float* t = tempRst.data();

    return flat_search(t, query, p, vecdim, k);
}

uint8_t scalarQuantization(float f, float min, float max){
    uint8_t m = 255;
    if (max == min){
        return 0;
    }

    float scale = m / (max-min);

    float tRst = std::round((f - min) * scale);
    if (tRst < 0){
        tRst = 0.0f;
    }
    if (tRst > 255){
        tRst = 255.0f;
    }

    uint8_t rst = (uint16_t)tRst;

    return rst;
}