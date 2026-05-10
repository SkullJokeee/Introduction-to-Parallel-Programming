#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include "simd.h"
 #include "flat_simd.h"

struct SQIndex {
    uint8_t* qtdBase;
    std::vector<float> mins;
    std::vector<float> maxs;

    ~SQIndex() {
        free(qtdBase);
    }
};

uint8_t scalarQuantization(float f, float min, float max){
    uint8_t m = 255;
    if (max == min){
        return 0;
    }

    float scale = m / (max-min);

    // float tRst = std::round((f - min) * scale);
    float tRst = (f - min) * scale + 0.5f; //0507
    if (tRst < 0){
        tRst = 0.0f;
    }
    if (tRst > 255){
        tRst = 255.0f;
    }

    uint8_t rst = (uint16_t)tRst;

    return rst;
}

SQIndex build_sq_index(float* base, size_t base_number, size_t vecdim){
    SQIndex idx;
    idx.qtdBase = align<uint8_t>(base_number * vecdim);
    idx.mins.resize(vecdim);
    idx.maxs.resize(vecdim);

    for(int d = 0; d < vecdim; ++d){
        float min = base[d], max = base[d];
        for(int i = 0; i < base_number; ++i){
            float val = base[d + i * vecdim];
            if(val < min){
                min = val;
            }
            if(val > max){
                max = val;
            }
        }
        idx.mins[d] = min;
        idx.maxs[d] = max;
    }


    for(int i = 0; i < base_number; ++i){
        for(int d = 0; d < vecdim; ++d){
            idx.qtdBase[d + i * vecdim] = scalarQuantization(base[d + i * vecdim], idx.mins[d], idx.maxs[d]);
        }
    }
    
    return idx;
}

std::priority_queue<std::pair<float, uint32_t>> sq_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k,const SQIndex& sq_idx){    
    
    std::priority_queue<std::pair<uint32_t, uint32_t>> q;
    // std::vector<std::pair<uint32_t, uint32_t>> v(base_number);

    std::vector<uint8_t> vecQ(vecdim);

    size_t p = 15;
    p = std::max(p,k);
    p = std::min(p,base_number);

    for(int d = 0; d < vecdim; ++d){
        vecQ[d] = scalarQuantization(query[d], sq_idx.mins[d], sq_idx.maxs[d]);
    }

    uint8_t* arrQ = vecQ.data();

    // for(int i = 0; i < base_number; i++){
        // const uint8_t* current = sq_idx.qtdBase + i * vecdim;
        // uint32_t dis = EucDisSIMDNeon(current, arrQ, vecdim);

    //     v[i] = {dis, i};
    // }

    // std::nth_element(v.begin(), v.begin() + p, v.end()); // 快速选择

    for(int i = 0; i < base_number; i++){
        // __builtin_prefetch(sq_idx.qtdBase + (i + 4) * vecdim, 0, 1);

        const uint8_t* current = sq_idx.qtdBase + i * vecdim;
        uint32_t dis = EucDisSIMDNeon(current, arrQ, vecdim);
        if(q.size() < p) {
            q.push({dis, i});
        }
        else {
            if(dis < q.top().first) {
                q.pop();
                q.push({dis, i});
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> rst_q;

    for(int i = 0; i < p; i ++){
        // uint32_t idx = v[i].second;
        uint32_t idx = q.top().second;
        q.pop();

        const float* current = base + idx * vecdim;
        
        float dis = InnerProductSIMDNeon(current, query, vecdim);

        if(rst_q.size() < k) {
            rst_q.push({dis, idx});
        }
        else {
            if(dis < rst_q.top().first) {
                rst_q.pop(); 
                rst_q.push({dis, idx});
            }
        }
    }

    return rst_q;

}
