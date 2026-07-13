#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "common_serial.h"

std::priority_queue<std::pair<float, uint32_t>> ivf_serial_search(const float* base, const float* query, size_t cb_n, size_t ivf_n, size_t cb_dim, size_t ivf_dim, size_t k, const float* base_ivf, const float* codebook_ivf, const uint32_t* list_ivf, const uint32_t* offset_ivf){
    const size_t nprobe = 30;
    const size_t probe_n = std::min(nprobe, cb_n);

    std::priority_queue<std::pair<float, uint32_t>> center_q;
    for(size_t i = 0; i < cb_n; ++i){
        float dis = 1.0f - scalar_ip_centroid(query, codebook_ivf, i, cb_dim);
        if(center_q.size() < probe_n){
            center_q.push({dis, i});
        }else if(dis < center_q.top().first){
            center_q.pop();
            center_q.push({dis, i});
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> rst;
    while(!center_q.empty()){
        uint32_t idx = center_q.top().second;
        center_q.pop();

        uint32_t start = offset_ivf[idx];
        uint32_t end = offset_ivf[idx + 1];
        uint32_t t = start + ((end - start) / 4) * 4;

        for(uint32_t i = start; i < t; ++i){
            float ip = 0.0f;
            for(size_t d = 0; d < ivf_dim; ++d){
                ip += query[d] * ivf_base_get_transposed(base_ivf, ivf_dim, start, i, d);
            }
            float dis = 1.0f - ip;
            uint32_t base_id = list_ivf[i];

            if(rst.size() < k){
                rst.push({dis, base_id});
            }else if(dis < rst.top().first){
                rst.pop();
                rst.push({dis, base_id});
            }
        }

        for(uint32_t i = t; i < end; ++i){
            float ip = 0.0f;
            for(size_t d = 0; d < ivf_dim; ++d){
                ip += query[d] * ivf_base_get_standard(base_ivf, ivf_dim, i, d);
            }
            float dis = 1.0f - ip;
            uint32_t base_id = list_ivf[i];

            if(rst.size() < k){
                rst.push({dis, base_id});
            }else if(dis < rst.top().first){
                rst.pop();
                rst.push({dis, base_id});
            }
        }
    }

    return rst;
}
