#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <pthread.h>
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

struct Task;
struct ThreadPool;

void* ivf_pq_thread(void* p){
    ThreadPool* pool = (ThreadPool*)p;

    while(1){
        Task t;

        pthread_mutex_lock(&pool->queue_lock);
        while(pool->tasks.empty()){

            if(pool->isStop){
                pthread_mutex_unlock(&pool->queue_lock);
                return NULL;
            }
            pthread_cond_wait(&pool->queue_cond, &pool->queue_lock);
        }

        t = pool->tasks.front();
        pool->tasks.pop();
        pthread_mutex_unlock(&pool->queue_lock);

        if(t.type == 0){
            std::vector<float32x4_t> v_query(cb2_dim);

            for(int d = 0; d < cb2_dim; ++d){
                v_query[d] = vdupq_n_f32(pool->query[d]);

            }

            for(int i = t.start; i < t.end; i += 4){

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
        else if(t.type == 1){
            std::priority_queue<std::pair<float, uint32_t>> temp_q;
            uint32_t start = t.start;
            uint32_t end = t.end;
            uint32_t cluster_idx = t.id;
            uint32_t last = start + ((end - start) / 4) * 4;

            size_t sub_d = cb_dim;
            size_t k_pq = 256;
            size_t m = pq_dim;

            float* lut = pool->lut;
            const float* pq_cbs = pool->pq_cb;

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

            const float* lut0 = lut;
            const float* lut1 = lut + k_pq;
            const float* lut2 = lut + 2 * k_pq;
            const float* lut3 = lut + 3 * k_pq;

            for(int i = start; i < last; i++){
                const uint8_t* idx = pool->base_pq + i * m;

                float d = lut0[idx[0]] + lut1[idx[1]] + lut2[idx[2]] + lut3[idx[3]];
                uint32_t id = list_ivf[i];

                if(temp_q.size() < k){
                    temp_q.push({d, id});
                }
                else if(d < temp_q.top().first){
                    temp_q.pop();
                    temp_q.push({d, id});
                }
            }

            for(int i = last; i < end; i++){

                const uint8_t* idx = pool->base_pq + i * m;
                float d = lut0[idx[0]] + lut1[idx[1]] + lut2[idx[2]] + lut3[idx[3]];
                uint32_t id = list_ivf[i];

                if(temp_q.size() < k){
                    temp_q.push({d, id});
                }
                else if(d < temp_q.top().first){
                    temp_q.pop();
                    temp_q.push({d, id});
                }
            }

            pool->task_rst[cluster_idx] = temp_q;
        }

        pthread_mutex_lock(&pool->done_lock);
        pool->num -= 1;
        if(pool->num == 0){
            pthread_cond_broadcast(&pool->done_cond);
        }
        pthread_mutex_unlock(&pool->done_lock);

    }

    return NULL;
}

std::priority_queue<std::pair<float, uint32_t>> ivf_pq_search(ThreadPool* pool){

    size_t nprobe = 20;
    nprobe = std::min(nprobe, cb2_n);

    int size = cb2_n / 8;

    for(int i = 0; i < 8; i++){
        Task t;
        t.type = 0;
        t.start = i * size;
        t.end = (i + 1) * size;

        pthread_mutex_lock(&pool->queue_lock);
        pool->num += 1;
        pool->tasks.push(t);
        pthread_cond_signal(&pool->queue_cond);
        pthread_mutex_unlock(&pool->queue_lock);
    }

    pthread_mutex_lock(&pool->done_lock);
    while(pool->num > 0){
        pthread_cond_wait(&pool->done_cond, &pool->done_lock);

    }
    pthread_mutex_unlock(&pool->done_lock);

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

    std::priority_queue<std::pair<float, uint32_t>> rst_q;

    while(!q.empty()){
        uint32_t cluster_idx = q.top().second;
        q.pop();

        Task t;
        t.type = 1;
        t.start = offset_ivf[cluster_idx];
        t.end = offset_ivf[cluster_idx + 1];
        t.id = cluster_idx;

        pthread_mutex_lock(&pool->queue_lock);
        pool->num += 1;
        pool->tasks.push(t);
        pthread_cond_signal(&pool->queue_cond);
        pthread_mutex_unlock(&pool->queue_lock);

    }

    pthread_mutex_lock(&pool->done_lock);
    while(pool->num > 0){
        pthread_cond_wait(&pool->done_cond, &pool->done_lock);
    }
    pthread_mutex_unlock(&pool->done_lock);

    for(int i = 0; i < cb2_n; i++){
        auto& temp_q = pool->task_rst[i]; ////
        while(!temp_q.empty()){
            auto pr = temp_q.top();
            temp_q.pop();

            if(rst_q.size() < k){
                rst_q.push(pr);
            }
            else if(pr.first < rst_q.top().first){
                rst_q.pop();
                rst_q.push(pr);
            }
        }
    }

    return rst_q;
}
