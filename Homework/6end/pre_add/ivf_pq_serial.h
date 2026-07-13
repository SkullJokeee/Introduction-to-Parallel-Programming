#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "common_serial.h"

std::priority_queue<std::pair<float, uint32_t>> ivf_pq_serial_search(float* base, float* query, size_t nlist, size_t n, size_t vecdim, size_t m, size_t k_pq, size_t top_k, const float* ivf_cb, const float* pq_cbs, const uint32_t* offset, const uint32_t* lst, const uint8_t* base_pq){
    size_t nprobe = 10;
    nprobe = std::min(nprobe, nlist);
    size_t sub_d = vecdim / m;

    std::priority_queue<std::pair<float, uint32_t>> ivf_q;
    for(size_t i = 0; i < nlist; ++i){
        float dis = 1.0f - scalar_ip_centroid(query, ivf_cb, i, vecdim);
        if(ivf_q.size() < nprobe){
            ivf_q.push({dis, (uint32_t)i});
        }else if(dis < ivf_q.top().first){
            ivf_q.pop();
            ivf_q.push({dis, (uint32_t)i});
        }
    }

    size_t p = 200;
    p = std::max(p, top_k);
    std::priority_queue<std::pair<float, uint32_t>> pq_q;

    while(!ivf_q.empty()){
        uint32_t idx = ivf_q.top().second;
        ivf_q.pop();

        std::vector<float> lut(m * k_pq);
        for(size_t j = 0; j < m; ++j){
            const float* segment = query + j * sub_d;
            for(size_t i = 0; i < k_pq; ++i){
                const float* center = pq_cbs + (idx * m * k_pq + j * k_pq + i) * sub_d;
                float ip = scalar_ip(segment, center, sub_d);
                lut[j * k_pq + i] = 1.0f - ip;
            }
        }

        const float* lut0 = lut.data();
        const float* lut1 = lut.data() + k_pq;
        const float* lut2 = lut.data() + 2 * k_pq;
        const float* lut3 = lut.data() + 3 * k_pq;

        uint32_t start = offset[idx];
        uint32_t end = offset[idx + 1];

        for(uint32_t i = start; i < end; ++i){
            const uint8_t* code = base_pq + i * m;
            float d = lut0[code[0]] + lut1[code[1]] + lut2[code[2]] + lut3[code[3]];
            uint32_t actual_id = lst[i];

            if(pq_q.size() < p){
                pq_q.push({d, actual_id});
            }else if(d < pq_q.top().first){
                pq_q.pop();
                pq_q.push({d, actual_id});
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> rst;
    while(!pq_q.empty()){
        uint32_t actual_id = pq_q.top().second;
        pq_q.pop();

        const float* current = base + actual_id * vecdim;
        float dis = 1.0f - scalar_ip(current, query, vecdim);

        if(rst.size() < top_k){
            rst.push({dis, actual_id});
        }else if(dis < rst.top().first){
            rst.pop();
            rst.push({dis, actual_id});
        }
    }

    return rst;
}
