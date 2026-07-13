#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include "common_serial.h"

std::priority_queue<std::pair<float, uint32_t>> pq_serial_search(float* base, float* query, size_t cb_n, size_t pq_n, size_t vecdim, size_t cb_dim, size_t pq_dim, size_t k, const uint8_t* base_pq, const float* codebook_pq){
    std::priority_queue<std::pair<float, uint32_t>> q;

    size_t p = 4000;
    p = std::max(p, k);
    p = std::min(p, pq_n);

    std::vector<float> lut(cb_n);
    for(size_t j = 0; j < pq_dim; ++j){
        const float* segment = query + j * cb_dim;
        for(size_t i = 0; i < 256; ++i){
            const float* center = codebook_pq + (j * 256 + i) * cb_dim;
            float ip = scalar_ip(segment, center, cb_dim);
            lut[j * 256 + i] = 1.0f - ip;
        }
    }

    const float* lut0 = lut.data();
    const float* lut1 = lut.data() + 256;
    const float* lut2 = lut.data() + 512;
    const float* lut3 = lut.data() + 768;

    for(size_t i = 0; i < pq_n; ++i){
        const uint8_t* idx = base_pq + i * pq_dim;
        float d = lut0[idx[0]] + lut1[idx[1]] + lut2[idx[2]] + lut3[idx[3]];

        if(q.size() < p){
            q.push({d, (uint32_t)i});
        }else if(d < q.top().first){
            q.pop();
            q.push({d, (uint32_t)i});
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> rst;
    while(!q.empty()){
        uint32_t idx = q.top().second;
        q.pop();

        const float* current = base + idx * vecdim;
        float dis = 1.0f - scalar_ip(current, query, vecdim);

        if(rst.size() < k){
            rst.push({dis, idx});
        }else if(dis < rst.top().first){
            rst.pop();
            rst.push({dis, idx});
        }
    }

    return rst;
}
