#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include "simd.h"
#include "flat_simd.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include <pthread.h>
#include <atomic>

struct ThreadPool;
struct PoolArg;

extern float* test_query;
extern int* test_gt;
extern float* base;
extern size_t test_number;
extern size_t base_number;
extern size_t test_gt_d;
extern size_t vecdim;
extern const size_t k;
extern size_t cb2_n;
extern size_t cb2_dim;
extern size_t ivf_dim;
extern uint32_t* offset_ivf;
extern uint32_t* list_ivf;
extern float* base_ivf;
extern float* codebook_ivf;

void* thread(void* p){
    PoolArg* arg = (PoolArg*)p;
    ThreadPool* pool = arg->pool;
    int tid = arg->id;
    int job = 0;
    
    while(1){
        pthread_mutex_lock(&pool->lock);
        while(pool->current_job == job && !pool->isStop){
            pthread_cond_wait(&pool->start_cond, &pool->lock);
        }
        if(pool->isStop){
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }
        job = pool->current_job;
        int current_phase = pool->phase;
        pthread_mutex_unlock(&pool->lock);
        
        if(current_phase == 1){ 
            std::vector<float32x4_t> v_query(cb2_dim); 
            
            for(size_t d = 0; d < cb2_dim; ++d){
                v_query[d] = vdupq_n_f32(pool->query[d]);
            }
            
            // 各线程并行计算查询向量到IVF聚类中心的距离
            while(1){
                int i = pool->task_idx.fetch_add(4);
                if(i >= pool->task_max) break;
                
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);
                float32x4_t sum4 = vdupq_n_f32(0.0f);
                
                const float* block = codebook_ivf + i * cb2_dim;
                
                for(int d = 0; d < cb2_dim; d += 4){
                    float32x4_t c_vec1 = vld1q_f32(block + d * 4);
                    float32x4_t c_vec2 = vld1q_f32(block + (d+1) * 4);
                    float32x4_t c_vec3 = vld1q_f32(block + (d+2) * 4);
                    float32x4_t c_vec4 = vld1q_f32(block + (d+3) * 4);
                    
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
                
                pool->cb_dis[i] = 1.0f - dis_array[0];
                pool->cb_dis[i+1] = 1.0f - dis_array[1];
                pool->cb_dis[i+2] = 1.0f - dis_array[2];
                pool->cb_dis[i+3] = 1.0f - dis_array[3];
            }
        }
        else if(current_phase == 2){ 
            std::vector<std::priority_queue<std::pair<float, uint32_t>>> cluster_queues;
            std::vector<float32x4_t> v_query(ivf_dim);
            
            for(size_t d = 0; d < ivf_dim; ++d){
                v_query[d] = vdupq_n_f32(pool->query[d]);
            }
            
            // 各线程在选中的簇内计算精确距离并维护局部top-k
            while(1){
                int task_id = pool->task_idx.fetch_add(1);
                if(task_id >= pool->task_max) break;
                
                uint32_t cluster_idx = pool->active_clusters[task_id];
                uint32_t start = offset_ivf[cluster_idx];
                uint32_t end = offset_ivf[cluster_idx + 1];
                uint32_t last = start + ((end - start) / 4) * 4;
                
                std::priority_queue<std::pair<float, uint32_t>> cluster_q;
                
                for(int i = start; i < last; i += 4){
                    __builtin_prefetch(base_ivf + (i + 8) * ivf_dim, 0, 1);
                    
                    float32x4_t sum1 = vdupq_n_f32(0.0f);
                    float32x4_t sum2 = vdupq_n_f32(0.0f);
                    float32x4_t sum3 = vdupq_n_f32(0.0f);
                    float32x4_t sum4 = vdupq_n_f32(0.0f);
                    
                    const float* block = base_ivf + i * ivf_dim;
                    
                    for(int d = 0; d < ivf_dim; d += 4){
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
                    
                    for(int j = 0; j < 4; j++){
                        float dis = 1.0f - dis_array[j];
                        uint32_t l = list_ivf[i + j];
                        if(cluster_q.size() < k){
                            cluster_q.push({dis, l});
                        }
                        else if(dis < cluster_q.top().first){
                            cluster_q.pop(); 
                            cluster_q.push({dis, l});
                        }
                    }
                }
                
                for(uint32_t i = last; i < end; i++){
                    const float* current = base_ivf + i * ivf_dim;
                    float dis = InnerProductSIMDNeon(current, pool->query, ivf_dim);
                    uint32_t l = list_ivf[i];
                    if(cluster_q.size() < k){
                        cluster_q.push({dis, l});
                    }
                    else if(dis < cluster_q.top().first){
                        cluster_q.pop(); 
                        cluster_q.push({dis, l});
                    }
                }
                
                cluster_queues.push_back(cluster_q);
            }
            
            std::priority_queue<std::pair<float, uint32_t>> local_q;
            for(auto& q : cluster_queues){
                while(!q.empty()){
                    auto P = q.top();
                    q.pop();
                    if(local_q.size() < k){
                        local_q.push(P);
                    }
                    else if(P.first < local_q.top().first){
                        local_q.pop();
                        local_q.push(P);
                    }
                }
            }
            pool->thread_rst[tid] = local_q;
        }
        
        pthread_mutex_lock(&pool->lock);
        pool->active_threads -= 1;
        if(pool->active_threads == 0){
            pthread_cond_broadcast(&pool->done_cond);
        }
        pthread_mutex_unlock(&pool->lock);
    }
    
    return NULL;
}

std::priority_queue<std::pair<float, uint32_t>> ivf_search(ThreadPool* pool){    
    size_t nprobe = 25;
    nprobe = std::min(nprobe, cb2_n);
    
    pool->run_phase(1, cb2_n);
    
    std::priority_queue<std::pair<float, uint32_t>> q;
    for(int i = 0; i < cb2_n; i++){
        if(q.size() < nprobe){
            q.push({pool->cb_dis[i], i});
        } 
        else if(pool->cb_dis[i] < q.top().first){
            q.pop(); 
            q.push({pool->cb_dis[i], i}); 
        }
    }
    
    pool->active_clusters.clear();
    
    while(!q.empty()){
        pool->active_clusters.push_back(q.top().second);
        q.pop();
    }
    
    pool->run_phase(2, pool->active_clusters.size());
    
    std::priority_queue<std::pair<float, uint32_t>> rst_q;
    // 汇总各线程结果得到最终top-k
    for(int i = 0; i < THREAD_N; i++){
        auto& local_q = pool->thread_rst[i];
        while(!local_q.empty()){
            auto P = local_q.top();
            local_q.pop();
            if(rst_q.size() < k){
                rst_q.push(P);
            }
            else if(P.first < rst_q.top().first){
                rst_q.pop();
                rst_q.push(P);
            }
        }
    }
    
    return rst_q;
}
