#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "simd.h"
#include "flat_simd.h"

std::priority_queue<std::pair<float, uint32_t>> ivf_pq_search(float* base, float* query, size_t nlist, size_t n, size_t vecdim, size_t m, size_t k_pq, size_t top_k, const float* ivf_cb, const float* pq_cbs, const uint32_t* offset, const uint32_t* lst, const uint8_t* base_pq) {
    size_t nprobe = 20;
    nprobe = std::min(nprobe, nlist);
    size_t sub_d = vecdim / m;
    std::vector<float32x4_t> v_query(vecdim);
    
    for(size_t d=0; d<vecdim; ++d){
        v_query[d] = vdupq_n_f32(query[d]);
    }
    
    std::priority_queue<std::pair<float, uint32_t>> q_ivf;
    
    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, uint32_t>> temp_q;
        #pragma omp for nowait
        for(int i=0; i<nlist; i+=4){
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);
            
            const float* block = ivf_cb + i * vecdim;
            
            for(int d=0; d<vecdim; d+=4){
                float32x4_t c_vec1 = vld1q_f32(block + d*4);
                float32x4_t c_vec2 = vld1q_f32(block + (d+1)*4);
                float32x4_t c_vec3 = vld1q_f32(block + (d+2)*4);
                float32x4_t c_vec4 = vld1q_f32(block + (d+3)*4);
            
                sum1 = vmlaq_f32(sum1, v_query[d], c_vec1);
                sum2 = vmlaq_f32(sum2, v_query[d+1], c_vec2);
                sum3 = vmlaq_f32(sum3, v_query[d+2], c_vec3);
                sum4 = vmlaq_f32(sum4, v_query[d+3], c_vec4);
            }

            float32x4_t sum12 = vaddq_f32(sum1, sum2);
            float32x4_t sum34 = vaddq_f32(sum3, sum4);
            float32x4_t sum = vaddq_f32(sum12, sum34);
            
            float dis_array[4];
            vst1q_f32(dis_array, sum);
            
            for(int j=0; j<4; ++j){
                float d_val = 1.0f - dis_array[j];
                if(temp_q.size() < nprobe){
                    temp_q.push({d_val, i+j});
                }
                else if(d_val < temp_q.top().first){
                    temp_q.pop();
                    temp_q.push({d_val, i+j});
                }

            }
        }
        #pragma omp critical
        {
            while(!temp_q.empty()){
                auto pr = temp_q.top();
                temp_q.pop();

                if(q_ivf.size() < nprobe){
                    q_ivf.push(pr);
                }
                else if(pr.first < q_ivf.top().first){
                    q_ivf.pop();
                    q_ivf.push(pr);
                }
            }
        }
    }

    std::vector<uint32_t> temp_p;

    while(!q_ivf.empty()){

        temp_p.push_back(q_ivf.top().second);
        q_ivf.pop();
    }

    std::priority_queue<std::pair<float, uint32_t>> rst_q;
    
    size_t p = 500; 
    p = std::max(p, top_k);
    std::priority_queue<std::pair<float, uint32_t>> q_pq;
    
    #pragma omp parallel
    {
        float* lut = align<float>(m * k_pq);
        std::priority_queue<std::pair<float, uint32_t>> temp_rst;

        #pragma omp for nowait schedule(dynamic, 1)
        for(size_t idx=0; idx<temp_p.size(); idx++){
            uint32_t cluster_idx = temp_p[idx];
            for(int j=0; j<m; ++j){
                const float* segment = query + j * sub_d;
                for(int i=0; i<k_pq; i+=4){
                    float32x4_t sum1 = vdupq_n_f32(0.0f);
                    float32x4_t sum2 = vdupq_n_f32(0.0f);
                    float32x4_t sum3 = vdupq_n_f32(0.0f);
                    float32x4_t sum4 = vdupq_n_f32(0.0f);
                    
                    const float* c1 = pq_cbs + (cluster_idx * m * k_pq * sub_d) + (j * k_pq + i) * sub_d;
                    const float* c2 = pq_cbs + (cluster_idx * m * k_pq * sub_d) + (j * k_pq + i + 1) * sub_d;
                    const float* c3 = pq_cbs + (cluster_idx * m * k_pq * sub_d) + (j * k_pq + i + 2) * sub_d;
                    const float* c4 = pq_cbs + (cluster_idx * m * k_pq * sub_d) + (j * k_pq + i + 3) * sub_d;
                    
                    for(int d=0; d<sub_d; d+=4){
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
                    
                    lut[j*k_pq + i] = 1.0f - vaddvq_f32(sum1);
                    lut[j*k_pq + i + 1] = 1.0f - vaddvq_f32(sum2);
                    lut[j*k_pq + i + 2] = 1.0f - vaddvq_f32(sum3);
                    lut[j*k_pq + i + 3] = 1.0f - vaddvq_f32(sum4);
                }
            }
            const float* lut0 = lut;
            const float* lut1 = lut + k_pq;
            const float* lut2 = lut + 2 * k_pq;
            const float* lut3 = lut + 3 * k_pq;
            
            uint32_t start = offset[cluster_idx];
            uint32_t end = offset[cluster_idx+1];
            
            for(uint32_t i=start; i<end; i++){
                const uint8_t* idx_pq = base_pq + i * m;
                float d = lut0[idx_pq[0]] + lut1[idx_pq[1]] + lut2[idx_pq[2]] + lut3[idx_pq[3]];
                uint32_t actual_id = lst[i];
            
                if(temp_rst.size() < p){
                    temp_rst.push({d, actual_id});
                }
                else if(d < temp_rst.top().first){
                    temp_rst.pop();
                    temp_rst.push({d, actual_id});
                }
            }
        }

        free(lut);
        
        #pragma omp critical
        {
            while(!temp_rst.empty()){
                auto pr = temp_rst.top();
                temp_rst.pop();

                if(q_pq.size() < p){
                    q_pq.push(pr);
                }
                else if(pr.first < q_pq.top().first){
                    q_pq.pop();
                    q_pq.push(pr);
                }
            }
        }
    }

    while(!q_pq.empty()) {
        uint32_t actual_id = q_pq.top().second;
        q_pq.pop();

        const float* current = base + actual_id * vecdim;
        float dis = InnerProductSIMDNeon(current, query, vecdim);

        if(rst_q.size() < top_k){
            rst_q.push({dis, actual_id});
        }
        else if(dis < rst_q.top().first){
            rst_q.pop(); 
            rst_q.push({dis, actual_id});
        }
    }
    
    return rst_q;
}