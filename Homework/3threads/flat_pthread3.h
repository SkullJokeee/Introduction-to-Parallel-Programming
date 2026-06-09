#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <atomic>
#include "simd.h"

struct ThreadPool;
struct PoolArg;

extern float* base;
extern size_t base_number;
extern size_t vecdim;
extern const size_t k;

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
            int size = 1024; 
            
            // 各线程动态领取任务，计算查询向量与base向量的距离
            while(1){
                int i_task = pool->task_idx.fetch_add(1);
                if(i_task >= pool->task_max) break;
                
                int start = i_task * size;
                int end = std::min((int)base_number, (i_task + 1) * size);
                
                for(int i = start; i < end; i++){
                    const float* current = base + i * vecdim;
                    float dis = InnerProductSIMDNeon(current, pool->query, vecdim);
                    
                    if(pool->thread_rst[tid].size() < k){
                        pool->thread_rst[tid].push({dis, i});
                    }
                    
                    else{
                        if(dis < pool->thread_rst[tid].top().first){
                            pool->thread_rst[tid].pop();
                            pool->thread_rst[tid].push({dis, i});
                        }

                    }
                }
            }
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

std::priority_queue<std::pair<float, uint32_t>> flat_pthread_search(ThreadPool* pool){    
    int size = 1024;
    int max_tasks = (base_number + size - 1) / size;
    pool->run_phase(1, max_tasks);
    
    std::priority_queue<std::pair<float, uint32_t>> rst_q;

    // 合并各线程的局部结果得到最终的top-k
    for(int i = 0; i < pool->thread_rst.size(); i++){
    
        while(!pool->thread_rst[i].empty()){
            auto item = pool->thread_rst[i].top();
            pool->thread_rst[i].pop();
            
            if(rst_q.size() < k){
                rst_q.push(item);
            }

            else{
                if(item.first < rst_q.top().first){
                    rst_q.pop(); 
                    rst_q.push(item);
                }
            }

        }
    }
    
    return rst_q;
}