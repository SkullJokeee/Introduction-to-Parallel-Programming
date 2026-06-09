#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "simd.h"

std::priority_queue<std::pair<float, uint32_t>> ivf_search(const float* base, const float* query, size_t cb_n, size_t ivf_n, size_t cb_dim, size_t ivf_dim, size_t k, const float* base_ivf, const float* codebook_ivf, const uint32_t* list_ivf, const uint32_t* offset_ivf) {
    size_t nprobe = 25;
    nprobe = std::min(nprobe, cb_n);
    std::vector<float32x4_t> v_query(ivf_dim);

    for(size_t d=0; d<ivf_dim; ++d){
        v_query[d] = vdupq_n_f32(query[d]);
    }
    
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, uint32_t>> temp_q;
        
        #pragma omp for nowait
        for(int i=0; i<cb_n; i+=4){
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);
            
            const float* block = codebook_ivf + i * cb_dim;
            for(int d=0; d<cb_dim; d+=4){
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
                if(q.size() < nprobe){
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
    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, uint32_t>> temp_rst;
        #pragma omp for nowait schedule(dynamic, 1)
        for(size_t idx=0; idx<temp_p.size(); idx++){
            
            uint32_t cluster_idx = temp_p[idx];
            uint32_t start = offset_ivf[cluster_idx];
            uint32_t end = offset_ivf[cluster_idx+1];
            uint32_t last = start + ((end - start)/4)*4;
            
            for(uint32_t i=start; i<last; i+=4){
                __builtin_prefetch(base_ivf + (i+8)*ivf_dim, 0, 1);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);
                float32x4_t sum4 = vdupq_n_f32(0.0f);
                const float* block = base_ivf + i * ivf_dim;
                for(int d=0; d<ivf_dim; d+=4){
                    float32x4_t b_vec1 = vld1q_f32(block + d*4);
                    float32x4_t b_vec2 = vld1q_f32(block + (d+1)*4);
                    float32x4_t b_vec3 = vld1q_f32(block + (d+2)*4);
                    float32x4_t b_vec4 = vld1q_f32(block + (d+3)*4);
                    sum1 = vmlaq_f32(sum1, v_query[d], b_vec1);
                    sum2 = vmlaq_f32(sum2, v_query[d+1], b_vec2);
                    sum3 = vmlaq_f32(sum3, v_query[d+2], b_vec3);
                    sum4 = vmlaq_f32(sum4, v_query[d+3], b_vec4);
                }
                float32x4_t sum12 = vaddq_f32(sum1, sum2);
                float32x4_t sum34 = vaddq_f32(sum3, sum4);
                float32x4_t sum = vaddq_f32(sum12, sum34);
                float dis_array[4];
                vst1q_f32(dis_array, sum);
                for(int j=0; j<4; ++j){
                    float dis = 1.0f - dis_array[j];
                    uint32_t l = list_ivf[i+j];
                    if(temp_rst.size() < k){
                        temp_rst.push({dis, l});
                    }
                    else if(dis < temp_rst.top().first){
                        temp_rst.pop();
                        temp_rst.push({dis, l});
                    }
                }
            
            }
            
            for(uint32_t i=last; i<end; i++){
                const float* current = base_ivf + i * ivf_dim;
                float dis = InnerProductSIMDNeon(current, query, ivf_dim);
                uint32_t l = list_ivf[i];
                
                if(temp_rst.size() < k){
                    temp_rst.push({dis, l});
                }
                else if(dis < temp_rst.top().first){
                    temp_rst.pop();
                    temp_rst.push({dis, l});
                }
            }

        }
        
        #pragma omp critical
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

    return rst_q;
}