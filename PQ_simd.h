#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include "simd.h"
#include "flat_simd.h"

std::priority_queue<std::pair<float, uint32_t>> pq_adc_search(float* base, float* query, size_t cb_n, size_t pq_n, size_t vecdim, size_t cb_dim, size_t pq_dim, size_t k,const uint8_t* base_pq, const float* codebook_pq){    
    
    std::priority_queue<std::pair<float, uint32_t>> q;

    size_t p = 1000;
    p = std::max(p,k);
    p = 0;

    std::vector<float> lut(cb_n);

    for(int j = 0; j < pq_dim; j++){ // pq_dim:4
        const float* query_segment = query + j * cb_dim;
        for(int i = 0; i < 256; i++){
            const float* centroid = codebook_pq + (j * 256 + i) * cb_dim;
            lut[j * 256 + i] = InnerProductSIMDNeon(centroid, query_segment, cb_dim);
        }
    }

    std::vector<float> dis(pq_n);
    
    for(int i = 0; i < pq_n; i++){
        for(int j = 0 ; j < pq_dim; j++){
            uint8_t idx = base_pq[j + i * pq_dim];
            dis[i] += lut[j * 256 + idx];
        }
    }

    for(int i = 0; i < pq_n; i++){
        if(q.size() < p) {
            q.push({dis[i], i});
        }
        else {
            if(dis[i] < q.top().first) {
                q.pop(); 
                q.push({dis[i], i});
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

    return flat_simd_search(t, query, p, vecdim, k);
}
