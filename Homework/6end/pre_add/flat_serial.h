#pragma once
#include <queue>
#include <cstdint>

std::priority_queue<std::pair<float, uint32_t>> flat_serial_search(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){
    std::priority_queue<std::pair<float, uint32_t>> q;

    for(size_t i = 0; i < base_number; ++i){
        float ip = 0.0f;
        for(size_t d = 0; d < vecdim; ++d){
            ip += base[i * vecdim + d] * query[d];
        }
        float dis = 1.0f - ip;

        if(q.size() < k){
            q.push({dis, (uint32_t)i});
        }else if(dis < q.top().first){
            q.pop();
            q.push({dis, (uint32_t)i});
        }
    }

    return q;
}
