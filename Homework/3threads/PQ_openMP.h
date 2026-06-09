#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "simd.h"
#include "flat_simd.h"

std::priority_queue<std::pair<float, uint32_t>> pq_search(float* base, float* query, size_t cb_n, size_t pq_n, size_t vecdim, size_t cb_dim, size_t pq_dim, size_t k, const uint8_t* base_pq, const float* codebook_pq) {
    size_t p = 4000;
    p = std::max(p, k);
    p = std::min(p, pq_n);
    
    float* lut = align<float>(cb_n);
    
    // 计算查询向量到PQ码本的距离，生成查找表lut
    for(int j=0; j<pq_dim; ++j){
        const float* segment = query + j * cb_dim;
    
        for(int i=0; i<256; i+=4){
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);
    
            const float* c1 = codebook_pq + (j*256 + i) * cb_dim; ////
            const float* c2 = codebook_pq + (j*256 + i + 1) * cb_dim;
            const float* c3 = codebook_pq + (j*256 + i + 2) * cb_dim;
            const float* c4 = codebook_pq + (j*256 + i + 3) * cb_dim;
    
            for(int d=0; d<cb_dim; d+=4){
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
            lut[j*256 + i] = 1.0f - vaddvq_f32(sum1);
            lut[j*256 + i + 1] = 1.0f - vaddvq_f32(sum2);
            lut[j*256 + i + 2] = 1.0f - vaddvq_f32(sum3);
            lut[j*256 + i + 3] = 1.0f - vaddvq_f32(sum4);

        }
    }

    const float* lut0 = lut;
    const float* lut1 = lut + 256;
    const float* lut2 = lut + 512;
    const float* lut3 = lut + 768;
    
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    // 根据PQ编码从lut中查表计算所有向量的近似距离
    {
        std::priority_queue<std::pair<float, uint32_t>> temp_q;
        for(int i=0; i<pq_n; i+=4){
            __builtin_prefetch(base_pq + i*4 + 128, 0, 1);
            const uint8_t* idx1 = base_pq + i * 4;
            const uint8_t* idx2 = base_pq + (i+1) * 4;
            const uint8_t* idx3 = base_pq + (i+2) * 4;
            const uint8_t* idx4 = base_pq + (i+3) * 4;

            float d1 = lut0[idx1[0]] + lut1[idx1[1]] + lut2[idx1[2]] + lut3[idx1[3]];
            float d2 = lut0[idx2[0]] + lut1[idx2[1]] + lut2[idx2[2]] + lut3[idx2[3]];
            float d3 = lut0[idx3[0]] + lut1[idx3[1]] + lut2[idx3[2]] + lut3[idx3[3]];
            float d4 = lut0[idx4[0]] + lut1[idx4[1]] + lut2[idx4[2]] + lut3[idx4[3]];
            
            for(int j=0; j<4; ++j){
                float dis = (j==0?d1:(j==1?d2:(j==2?d3:d4)));
                uint32_t id = i + j;
                if(temp_q.size() < p){
                    temp_q.push({dis, id});
                }
                else if(dis < temp_q.top().first){
                    temp_q.pop();
                    temp_q.push({dis, id});
                }

            }
        }

        // #pragma omp critical
        {
            while(!temp_q.empty()){
                auto pr = temp_q.top();
                temp_q.pop();
                if(q.size() < p){
                    q.push(pr);
                }
                else if(pr.first < q.top().first){
                    q.pop();
                    q.push(pr);
                }
            }
        }

    }
    
    std::vector<uint32_t> temp_p;
    while(!q.empty()){
        temp_p.push_back(q.top().second);
        q.pop();
    }
    
    std::priority_queue<std::pair<float, uint32_t>> rst_q;
    // 对候选向量重新计算精确距离，得到最终结果
    {
        std::priority_queue<std::pair<float, uint32_t>> temp_rst;
        for(size_t i=0; i<temp_p.size(); ++i){
            uint32_t idx = temp_p[i];
            const float* current = base + idx * vecdim;
            float dis = InnerProductSIMDNeon(current, query, vecdim);
            if(temp_rst.size() < k){
                temp_rst.push({dis, idx});
            }
            else if(dis < temp_rst.top().first){
                temp_rst.pop();
                temp_rst.push({dis, idx});
            }
        }
        // #pragma omp critical
        {
            while(!temp_rst.empty()){
                auto pr = temp_rst.top();
                temp_rst.pop();
                
                if(rst_q.size() < k){
                    rst_q.push(pr);
                }
                else if(pr.first < rst_q.top().first){
                    rst_q.pop();
                    rst_q.push(pr);
                }
            }
        }
    }

    free(lut);

    return rst_q;
}