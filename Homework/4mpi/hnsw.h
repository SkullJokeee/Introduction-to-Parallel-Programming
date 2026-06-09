#pragma once
#include "hnswlib/hnswlib/hnswlib.h"
#include "simd.h"

class NEONSpace : public hnswlib::SpaceInterface<float> {
    size_t dim;
public:
    NEONSpace(size_t d) : dim(d) {}
    // 基于NEON的内积计算
    size_t get_data_size() override{
        return dim * sizeof(float); 
    }
    hnswlib::DISTFUNC<float> get_dist_func() override{
        return [](const void *a, const void *b, const void *qty_ptr) -> float {
            size_t qty = *((size_t*)qty_ptr);
            float rst = InnerProductSIMDNeon((const float*)a, (const float*)b, qty);
            return rst;
        };
    }
    void *get_dist_func_param() override{
        return &dim; 
    }
};
