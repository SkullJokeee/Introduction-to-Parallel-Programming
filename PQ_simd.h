#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include "simd.h"
#include "flat_simd.h"

std::priority_queue<std::pair<float, uint32_t>> pq_adc_search(float* base, float* query, size_t cb_n, size_t pq_n, size_t vecdim, size_t cb_dim, size_t pq_dim, size_t k,const uint8_t* base_pq, const float* codebook_pq){    
    
    // std::vector<std::pair<float, uint32_t>> v(pq_n);
    std::priority_queue<std::pair<float, uint32_t>> q;///

    size_t p = 3000;
    p = std::max(p,k);
    p = std::min(p,pq_n);

    float* lut = align<float>(cb_n);

    for(int j = 0; j < pq_dim; j++){
        const float* segment = query + j * cb_dim;
        
        for(int i = 0; i < 256; i += 4){
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);

            const float* c1 = codebook_pq + (j*256 + i) * cb_dim;
            const float* c2 = codebook_pq + (j*256 + i + 1) * cb_dim;
            const float* c3 = codebook_pq + (j*256 + i + 2) * cb_dim;
            const float* c4 = codebook_pq + (j*256 + i + 3) * cb_dim;

            for(int d = 0; d < cb_dim; d += 4){
                float32x4_t q_vec = vld1q_f32(segment + d);
                
                float32x4_t c1_vec = vld1q_f32(c1 + d);
                float32x4_t c2_vec = vld1q_f32(c2 + d);
                float32x4_t c3_vec = vld1q_f32(c3 + d);
                float32x4_t c4_vec = vld1q_f32(c4 + d);
                
                sum1 = vmlaq_f32(sum1, q_vec, c1_vec);
                sum2 = vmlaq_f32(sum2, q_vec, c2_vec);
                sum3 = vmlaq_f32(sum3, q_vec, c3_vec);
                sum4 = vmlaq_f32(sum4, q_vec, c4_vec);
            }
            
            float d1 = vaddvq_f32(sum1);
            float d2 = vaddvq_f32(sum2);
            float d3 = vaddvq_f32(sum3);
            float d4 = vaddvq_f32(sum4);

            lut[j*256 + i] = 1.0f - d1;
            lut[j*256 + i + 1] = 1.0f - d2;
            lut[j*256 + i + 2] = 1.0f - d3;
            lut[j*256 + i + 3] = 1.0f - d4;
        }
    }

    std::vector<float> dis(pq_n);

    const float* lut0 = lut;
    const float* lut1 = lut + 256;
    const float* lut2 = lut + 512;
    const float* lut3 = lut + 768;

    for(int i = 0; i < pq_n; i+=4){
        __builtin_prefetch(base_pq + i * 4 + 128, 0, 1); ///
        const uint8_t* idx1 = base_pq + i * 4;
        const uint8_t* idx2 = base_pq + (i + 1) * 4;
        const uint8_t* idx3 = base_pq + (i + 2) * 4;
        const uint8_t* idx4 = base_pq + (i + 3) * 4;
        // float d1 = lut[idx1[0]] + lut[256 + idx1[1]] + lut[512 + idx1[2]] + lut[768 + idx1[3]];
        // float d2 = lut[idx2[0]] + lut[256 + idx2[1]] + lut[512 + idx2[2]] + lut[768 + idx2[3]];
        // float d3 = lut[idx3[0]] + lut[256 + idx3[1]] + lut[512 + idx3[2]] + lut[768 + idx3[3]];
        // float d4 = lut[idx4[0]] + lut[256 + idx4[1]] + lut[512 + idx4[2]] + lut[768 + idx4[3]];
        float d1 = lut0[idx1[0]] + lut1[idx1[1]] + lut2[idx1[2]] + lut3[idx1[3]];
        float d2 = lut0[idx2[0]] + lut1[idx2[1]] + lut2[idx2[2]] + lut3[idx2[3]];
        float d3 = lut0[idx3[0]] + lut1[idx3[1]] + lut2[idx3[2]] + lut3[idx3[3]];
        float d4 = lut0[idx4[0]] + lut1[idx4[1]] + lut2[idx4[2]] + lut3[idx4[3]];
        
        dis[i] = d1;
        dis[i+1] = d2;
        dis[i+2] = d3;
        dis[i+3] = d4;
    }

    // for(int i = 0; i < pq_n; i++){
    //     v[i] = {dis[i], i};
    // }

    // std::nth_element(v.begin(), v.begin() + p, v.end()); 

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

    std::priority_queue<std::pair<float, uint32_t>> rst_q;

    for(int i = 0; i < p; i ++){
        // uint32_t idx = v[i].second;
        uint32_t idx = q.top().second;
        q.pop();

        const float* current = base + idx * vecdim;
        
        float dis = InnerProductSIMDNeon(current, query, vecdim);

        if(rst_q.size() < k){
            rst_q.push({dis, idx});
        }
        else {
            if(dis < rst_q.top().first){
                rst_q.pop(); 
                rst_q.push({dis, idx});
            }
        }
    }

    free(lut);

    return rst_q;
}
