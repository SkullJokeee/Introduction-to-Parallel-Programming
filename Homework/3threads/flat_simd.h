#pragma once
#include <queue>
#include <vector>
#include <arm_neon.h>
#include "simd.h"

std::priority_queue<std::pair<float, uint32_t>> flat_simd_search(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){
    std::priority_queue<std::pair<float, uint32_t>> q;
    std::vector<float> dis(base_number);

    for(int i = 0; i < base_number; i += 4){
        dis[i] = InnerProductSIMDNeon(base + i * vecdim, query, vecdim);
        dis[i+1] = InnerProductSIMDNeon(base + (i + 1) * vecdim, query, vecdim);
        dis[i+2] = InnerProductSIMDNeon(base + (i + 2) * vecdim, query, vecdim);
        dis[i+3] = InnerProductSIMDNeon(base + (i + 3) * vecdim, query, vecdim);
    }

    for(int i = 0; i < base_number; i++){
        if(q.size() < k) {
            q.push({dis[i], i});
        }
        else {
            if(dis[i] < q.top().first) {
                q.pop();
                q.push({dis[i], i});
            }
        }
    }

    return q;
}