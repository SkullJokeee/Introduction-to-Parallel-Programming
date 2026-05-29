#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "simd.h"

std::priority_queue<std::pair<float, uint32_t>> flat_openmp_search(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){
    int max_threads = omp_get_max_threads();
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> temp_q(max_threads);

    // int p = 20;
    // 使用OpenMP并行计算查询向量与所有base向量的距离

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 2048)
        for(int i = 0; i < base_number; i++){
            
            const float* current = base + i * vecdim;
            float dis = InnerProductSIMDNeon(current, query, vecdim);

            if(temp_q[tid].size() < k){
                temp_q[tid].push({dis, i});
            }
            else{
             
                if(dis < temp_q[tid].top().first){
                    temp_q[tid].pop();
                    temp_q[tid].push({dis, i});
                }
            }

        }
    }

    std::priority_queue<std::pair<float, uint32_t>> rst_q;
    
    // 合并各线程的局部结果得到最终的top-k
    for(int i = 0; i < max_threads; i++){
    
        while(!temp_q[i].empty()){
            auto item = temp_q[i].top();
            temp_q[i].pop();
            
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