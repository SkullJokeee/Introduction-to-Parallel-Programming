#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "hnsw.h"

std::vector<hnswlib::HierarchicalNSW<float>*> cluster_hnsw_idx;
std::vector<NEONSpace*> cluster_hnsw_sp;

// 为每个聚类独立建立HNSW索引
void BuildClusterHNSWIndices(const float* base_ivf, const uint32_t* offset_ivf, size_t nlist, size_t dim, size_t M = 16, size_t efConstruction = 150, size_t ef = 200){
    cluster_hnsw_idx.resize(nlist, nullptr);
    cluster_hnsw_sp.resize(nlist, nullptr);
    for(size_t c = 0; c < nlist; c++){
        size_t start = offset_ivf[c];
        size_t end = offset_ivf[c + 1];
        size_t n = end - start;
        if(n == 0) continue;
        NEONSpace* sp = new NEONSpace(dim);
        hnswlib::HierarchicalNSW<float>* alg = new hnswlib::HierarchicalNSW<float>(sp, n, M, efConstruction);
        for(size_t j = start; j < end; j++){
            alg->addPoint(base_ivf + j * dim, j - start);
        }
        cluster_hnsw_sp[c] = sp;
        cluster_hnsw_idx[c] = alg;
        alg->setEf(ef);
    }
}

std::priority_queue<std::pair<float, uint32_t>> ivf_hnsw_search(const float* base, const float* query, size_t cb_n, size_t ivf_n, size_t cb_dim, size_t ivf_dim, size_t k, const float* base_ivf, const float* codebook_ivf, const uint32_t* list_ivf, const uint32_t* offset_ivf){
    size_t nprobe = 20;
    nprobe = std::min(nprobe, cb_n);

    std::vector<float32x4_t> v_query(ivf_dim);
    for(size_t d = 0; d < ivf_dim; ++d){
        v_query[d] = vdupq_n_f32(query[d]);
    }

    std::priority_queue<std::pair<float, uint32_t>> q;

    for(int i = 0; i < (int)cb_n; i += 4){
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);
        const float* block = codebook_ivf + i * cb_dim;

        for(int d = 0; d < (int)cb_dim; d += 4){
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

        float d1 = 1.0f - dis_array[0];
        float d2 = 1.0f - dis_array[1];
        float d3 = 1.0f - dis_array[2];
        float d4 = 1.0f - dis_array[3];

        if(q.size() < nprobe){
            q.push({d1, (uint32_t)i});
        }
        else if(d1 < q.top().first){
            q.pop();
            q.push({d1, (uint32_t)i});
        }
        if(q.size() < nprobe){
            q.push({d2, (uint32_t)(i+1)});
        }
        else if(d2 < q.top().first){
            q.pop();
            q.push({d2, (uint32_t)(i+1)});
        }
        if(q.size() < nprobe){
            q.push({d3, (uint32_t)(i+2)});
        }
        else if(d3 < q.top().first){
            q.pop();
            q.push({d3, (uint32_t)(i+2)});
        }
        if(q.size() < nprobe){
            q.push({d4, (uint32_t)(i+3)});
        }
        else if(d4 < q.top().first){
            q.pop();
            q.push({d4, (uint32_t)(i+3)});
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> rst_q;

    while(!q.empty()){
        uint32_t cid = q.top().second;
        q.pop();
        if(cluster_hnsw_idx[cid] == nullptr){
            continue;
        }
        uint32_t start = offset_ivf[cid];
        auto ret = cluster_hnsw_idx[cid]->searchKnn(query, k);
        while(!ret.empty()){
            auto top = ret.top();
            ret.pop();
            uint32_t label = list_ivf[start + top.second];
            if(rst_q.size() < k){
                rst_q.push({top.first, label});
            }
            else if(top.first < rst_q.top().first){
                rst_q.pop();
                rst_q.push({top.first, label});
            }
        }
    }

    return rst_q;
}
