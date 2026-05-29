#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <pthread.h>
#include <atomic>
#include "simd.h"
#include "flat_simd.h"

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
extern size_t pq_dim;
extern size_t cb_dim;

struct ThreadPool;
struct PoolArg;

size_t P = 500; ////

void* thread(void* thread_p){
    PoolArg* arg = (PoolArg*)thread_p;
    ThreadPool* pool = arg->pool;
    int tid = arg->id;
    int job = 0;
    
    size_t sub_d = cb_dim;
    size_t k_pq = 256;
    size_t m = pq_dim;
    
    float* lut = (float*)aligned_alloc(16, m * k_pq * sizeof(float));
    
    while(1){
        pthread_mutex_lock(&pool->lock);
        while(pool->current_job == job && !pool->isStop){
            pthread_cond_wait(&pool->start_cond, &pool->lock);
        }
        if(pool->isStop){
            free(lut);
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }
        job = pool->current_job;
        int current_phase = pool->phase;
        pthread_mutex_unlock(&pool->lock);
        
        if(current_phase == 1){
            std::vector<float32x4_t> v_query(cb2_dim);
            for(int d = 0; d < cb2_dim; ++d){
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
                    float32x4_t c_vec2 = vld1q_f32(block + (d + 1) * 4);
                    float32x4_t c_vec3 = vld1q_f32(block + (d + 2) * 4);
                    float32x4_t c_vec4 = vld1q_f32(block + (d + 3) * 4);
                    
                    sum1 = vmlaq_f32(sum1, v_query[d], c_vec1);
                    sum2 = vmlaq_f32(sum2, v_query[d + 1], c_vec2);
                    sum3 = vmlaq_f32(sum3, v_query[d + 2], c_vec3);
                    sum4 = vmlaq_f32(sum4, v_query[d + 3], c_vec4);
                }
                
                float32x4_t sum12 = vaddq_f32(sum1, sum2);
                float32x4_t sum34 = vaddq_f32(sum3, sum4);
                float32x4_t sum = vaddq_f32(sum12, sum34);
                
                float dis_array[4];
                vst1q_f32(dis_array, sum);
                
                pool->cb_dis[i] = 1.0f - dis_array[0];
                pool->cb_dis[i + 1] = 1.0f - dis_array[1];
                pool->cb_dis[i + 2] = 1.0f - dis_array[2];
                pool->cb_dis[i + 3] = 1.0f - dis_array[3];
            }
        }
        else if(current_phase == 2){
            // 各线程在选中簇内计算PQ查找表lut并查表得到近似距离
            const float* pq_cbs = pool->pq_cb;
            const float* lut0 = lut;
            const float* lut1 = lut + k_pq;
            const float* lut2 = lut + 2 * k_pq;
            const float* lut3 = lut + 3 * k_pq;
            size_t p = P / 5; ////
            
            std::vector<std::priority_queue<std::pair<float, uint32_t>>> cluster_queues;
            
            while(1){
                int task_id = pool->task_idx.fetch_add(1);
                if(task_id >= pool->task_max) break;
                
                uint32_t cluster_idx = pool->active_clusters[task_id];
                uint32_t start = offset_ivf[cluster_idx];
                uint32_t end = offset_ivf[cluster_idx + 1];
                uint32_t last = start + ((end - start) / 4) * 4;
                
                std::priority_queue<std::pair<float, uint32_t>> cluster_q;
                
                for(int j = 0; j < m; j++){
                    const float* segment = pool->query + j * sub_d;
                    
                    for(int i = 0; i < k_pq; i += 4){

                        float32x4_t sum1 = vdupq_n_f32(0.0f);
                        float32x4_t sum2 = vdupq_n_f32(0.0f);
                        float32x4_t sum3 = vdupq_n_f32(0.0f);
                        float32x4_t sum4 = vdupq_n_f32(0.0f);
                        const float* c1 = pq_cbs + (cluster_idx * m * k_pq * sub_d) + (j * k_pq + i) * sub_d;
                        const float* c2 = pq_cbs + (cluster_idx * m * k_pq * sub_d) + (j * k_pq + i + 1) * sub_d;
                        const float* c3 = pq_cbs + (cluster_idx * m * k_pq * sub_d) + (j * k_pq + i + 2) * sub_d;
                        const float* c4 = pq_cbs + (cluster_idx * m * k_pq * sub_d) + (j * k_pq + i + 3) * sub_d;
                        
                        for(int d = 0; d < sub_d; d += 4){
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
                        
                        lut[j * k_pq + i] = 1.0f - d1;
                        lut[j * k_pq + i + 1] = 1.0f - d2;
                        lut[j * k_pq + i + 2] = 1.0f - d3;
                        lut[j * k_pq + i + 3] = 1.0f - d4;

                    }
                }
                
                for(int i = start; i < last; i++){
                    const uint8_t* idx = pool->base_pq + i * m;
                    float d = lut0[idx[0]] + lut1[idx[1]] + lut2[idx[2]] + lut3[idx[3]];
                    uint32_t id = list_ivf[i];
                    if(cluster_q.size() < p){
                        cluster_q.push({d, id});
                    }
                    else if(d < cluster_q.top().first){
                        cluster_q.pop();
                        cluster_q.push({d, id});
                    }
                }
                
                for(int i = last; i < end; i++){
                    const uint8_t* idx = pool->base_pq + i * m;
                    float d = lut0[idx[0]] + lut1[idx[1]] + lut2[idx[2]] + lut3[idx[3]];
                    
                    uint32_t id = list_ivf[i];
                    
                    if(cluster_q.size() < P){
                        cluster_q.push({d, id});
                    }
                    else if(d < cluster_q.top().first){
                        cluster_q.pop();
                        cluster_q.push({d, id});
                    }
                }
                
                cluster_queues.push_back(cluster_q);
            }
            
            std::priority_queue<std::pair<float, uint32_t>> local_q;
            for(auto& q : cluster_queues){
                while(!q.empty()){
                    auto pr = q.top();
                    q.pop();
                    if(local_q.size() < P){
                        local_q.push(pr);
                    }
                    else if(pr.first < local_q.top().first){
                        local_q.pop();
                        local_q.push(pr);
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

std::priority_queue<std::pair<float, uint32_t>> ivf_pq_search(ThreadPool* pool){
    size_t nprobe = 20;
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
    
    std::priority_queue<std::pair<float, uint32_t>> global_r;
    
    for(int i = 0; i < THREAD_N; i++){
        auto& temp_q = pool->thread_rst[i]; 
        while(!temp_q.empty()){
            auto pr = temp_q.top();
            temp_q.pop();
            if(global_r.size() < P){
                global_r.push(pr);
            }
            else if(pr.first < global_r.top().first){
                global_r.pop();
                global_r.push(pr);
            }
        }
    }
    
    std::priority_queue<std::pair<float, uint32_t>> rst_q;
    
    // 对候选向量重新计算精确距离，得到最终结果
    while(!global_r.empty()){
        uint32_t actual_id = global_r.top().second;
        global_r.pop();
        const float* current = base + actual_id * vecdim;
        float dis = InnerProductSIMDNeon(current, pool->query, vecdim);
        
        if(rst_q.size() < k){
            rst_q.push({dis, actual_id});
        }
        else if(dis < rst_q.top().first){
            rst_q.pop();
            rst_q.push({dis, actual_id});
        }
    }
    
    return rst_q;
}
