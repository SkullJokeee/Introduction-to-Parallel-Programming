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

struct Task;
struct ThreadPool;

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

void LoadIvfData(std::string path, size_t nlist, size_t n, size_t d, uint32_t*& offset, uint32_t*& ivflist, float*& base){
    std::ifstream fin;

    fin.open(path + "ivf_offset.bin", std::ios::in | std::ios::binary);
    offset = new uint32_t[nlist + 1];
    fin.read((char*)offset, (nlist + 1) * sizeof(uint32_t));
    fin.close();

    fin.open(path + "ivf_ivflist.bin", std::ios::in | std::ios::binary);
    ivflist = new uint32_t[n];
    fin.read((char*)ivflist, n * sizeof(uint32_t));
    fin.close();

    fin.open(path + "ivf_base.bin", std::ios::in | std::ios::binary);
    base = new float[n * d];
    fin.read((char*)base, n * d * sizeof(float));
    fin.close();
}

void* ivf_thread(void* p){
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
            
            for (size_t d = 0; d < cb2_dim; ++d) {
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
        else if(t.type == 1){ 
            std::priority_queue<std::pair<float, uint32_t>> local_q;
            uint32_t start = t.start;
            uint32_t end = t.end;
            uint32_t last = start + ((end - start) / 4) * 4;

            std::vector<float32x4_t> v_query(ivf_dim);
            for (size_t d = 0; d < ivf_dim; ++d) {
                v_query[d] = vdupq_n_f32(pool->query[d]);
            }

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

                    if(local_q.size() < k){
                        local_q.push({dis, l});
                    }
                    else if(dis < local_q.top().first){
                        local_q.pop(); 
                        local_q.push({dis, l});
                    }
                }
            }

            for(uint32_t i = last; i < end; i++){
                const float* current = base_ivf + i * ivf_dim;

                float dis = InnerProductSIMDNeon(current, pool->query, ivf_dim);
                uint32_t l = list_ivf[i];

                if(local_q.size() < k){
                    local_q.push({dis, l});
                }
                else if(dis < local_q.top().first){
                    local_q.pop(); 
                    local_q.push({dis, l});
                }
            }
            
            pool->task_rst[t.id] = local_q;
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

std::priority_queue<std::pair<float, uint32_t>> ivf_search(ThreadPool* pool){    
    size_t nprobe = 20;
    nprobe = std::min(nprobe, cb2_n);

    int size = cb2_n / 8; 

    for(int i = 0; i < 8; i++){
        Task t;

        t.type = 0;
        t.start = i * size;
        t.end = (i + 1) * size;

        pthread_mutex_lock(&pool->done_lock); ////
        pool->num += 1;
        pthread_mutex_unlock(&pool->done_lock);

        pthread_mutex_lock(&pool->queue_lock);
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

    int task_id = 0;
    while(!q.empty()){
        uint32_t idx = q.top().second;
        q.pop();

        Task t;
        t.type = 1;
        t.start = offset_ivf[idx];
        t.end = offset_ivf[idx + 1];
        t.id = task_id++;

        pthread_mutex_lock(&pool->done_lock);
        pool->num += 1;
        pthread_mutex_unlock(&pool->done_lock);

        pthread_mutex_lock(&pool->queue_lock);
        pool->tasks.push(t);
        pthread_cond_signal(&pool->queue_cond);
        pthread_mutex_unlock(&pool->queue_lock);
    }

    pthread_mutex_lock(&pool->done_lock);

    while(pool->num > 0){
        pthread_cond_wait(&pool->done_cond, &pool->done_lock);
    }
    pthread_mutex_unlock(&pool->done_lock);

    for(int i = 0; i < task_id; i++){
        auto& local_q = pool->task_rst[i];

        while(!local_q.empty()){
            auto p = local_q.top();
            local_q.pop();

            if(rst_q.size() < k){
                rst_q.push(p);
            }
            else if(p.first < rst_q.top().first){
                rst_q.pop();
                rst_q.push(p);
            }

        }
    }

    return rst_q;
}