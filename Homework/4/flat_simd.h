#pragma once
#include <queue>
#include <cassert>
#include <arm_neon.h>
#include "simd.h"

std::priority_queue<std::pair<float, uint32_t>> flat_simd_search(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){

    std::priority_queue<std::pair<float, uint32_t>> q;

    for(int i = 0; i < base_number; i++) {
        const float* current = base + i * vecdim;
        
        float dis = 0.0f;
        dis = InnerProductSIMDNeon(current, query, vecdim);

        if(q.size() < k) {
            q.push({dis, i});
        }
        else {
            if(dis < q.top().first) {
                q.pop(); 
                q.push({dis, i});
            }
        }
    }

    return q;
}