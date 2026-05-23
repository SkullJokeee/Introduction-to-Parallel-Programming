#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include "simd.h"
#include "flat_simd.h"
#include <pthread.h>

struct ThreadPool;

extern float* test_query;
extern int* test_gt;
extern float* base;
extern float* codebook_pq;
extern uint8_t* base_pq;
extern size_t test_number = 0, base_number = 0;
extern size_t test_gt_d = 0, vecdim = 0;
extern size_t pq_n = 0, cb_n = 0;
extern size_t pq_dim = 0, cb_dim = 0;
extern const size_t k = 10;

void* thread(void* p){
    ThreadPool* pool = (ThreadPool*)p;
    while(1){
        Task t;

        pthread_mutex_lock(&pool->queue_lock);
        while(pool->tasks.empty()){
            pthread_cond_wait(&pool->queue_cond, &pool->queue_lock);
        }
        if(pool->tasks.empty() && !isStop){
            pthread_mutex_unlock(&pool->queue_lock);
            break;
        }

        t = pool->tasks.front(); 
        pool->tasks.pop();
        pthread_mutex_unlock(&pool->queue_lock);
        
        // if(t.type == 0){ // 构建lut
            for(int j = 0; j < pq_dim; j++){
                const float* segment = pool->query + j * cb_dim;
                
                for(int i = t.start; i < t.end; i += 4){ // 类中心集合并行
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

                    pool->lut[j*256 + i] = 1.0f - d1;
                    pool->lut[j*256 + i + 1] = 1.0f - d2;
                    pool->lut[j*256 + i + 2] = 1.0f - d3;
                    pool->lut[j*256 + i + 3] = 1.0f - d4;
                }
            }

        // }
        // else if(t.type == 1){ // 查找
        // }
        
        pthread_mutex_lock(&pool->done_lock);

        pool->num -= 1;
        if(pool->num == 0){
            pthread_cond_broadcast(&pool->done_cond);
        }
        pthread_mutex_unlock(&pool->done_lock);

        // if(pool->isStop && pool->tasks.empty()){
        //     break;
        // }

    }

    return NULL;
}

std::priority_queue<std::pair<float, uint32_t>> pq_adc_search(ThreadPool* pool){    
    
    std::priority_queue<std::pair<float, uint32_t>> q;

    size_t p = 4000;
    p = std::max(p,k);
    p = std::min(p,pq_n);


    // 计算查询向量到类中心的距离并存入lut（通过四个子向量分别计算后求和）
    int size = 256 / 16;

    for(int i = 0; i < 16; i++){
        Task t;
        t.type = 0;
        t.start = i * size;
        t.end = (i + 1) * size;

        pool->num += 1;
        pool->tasks.push(t);
    }

    pthread_mutex_lock(&pool->done_lock);
    while(pool->num > 0){
        pthread_cond_wait(&pool->done_cond, &pool->done_lock);
    }
    pthread_mutex_unlock(&pool->done_lock);

    // pthread_mutex_lock(&pool.queue_lock);
    // pool.stop = true;
    // pthread_cond_broadcast(&pool.queue_cond);
    // pthread_mutex_unlock(&pool.queue_lock);

    std::vector<float> dis(pq_n);

    const float* lut0 = pool->lut;
    const float* lut1 = pool->lut + 256;
    const float* lut2 = pool->lut + 512;
    const float* lut3 = pool->lut + 768;

    // 对所有base向量，根据其索引在lut表中找到其对应的类中心和查询向量的距离

    for(int i = 0; i < pq_n; i+=4){
        __builtin_prefetch(base_pq + i * 4 + 128, 0, 1);
        const uint8_t* idx1 = base_pq + i * 4;
        const uint8_t* idx2 = base_pq + (i + 1) * 4;
        const uint8_t* idx3 = base_pq + (i + 2) * 4;
        const uint8_t* idx4 = base_pq + (i + 3) * 4;
        float d1 = lut0[idx1[0]] + lut1[idx1[1]] + lut2[idx1[2]] + lut3[idx1[3]];
        float d2 = lut0[idx2[0]] + lut1[idx2[1]] + lut2[idx2[2]] + lut3[idx2[3]];
        float d3 = lut0[idx3[0]] + lut1[idx3[1]] + lut2[idx3[2]] + lut3[idx3[3]];
        float d4 = lut0[idx4[0]] + lut1[idx4[1]] + lut2[idx4[2]] + lut3[idx4[3]];
        
        dis[i] = d1;
        dis[i+1] = d2;
        dis[i+2] = d3;
        dis[i+3] = d4;
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

    std::priority_queue<std::pair<float, uint32_t>> rst_q;

    for(int i = 0; i < p; i ++){
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
