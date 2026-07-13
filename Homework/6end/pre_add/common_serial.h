#pragma once
#include <cstdint>
#include <cstddef>

inline float scalar_ip(const float* a, const float* b, size_t dim){
    float s = 0.0f;
    for(size_t d = 0; d < dim; ++d){
        s += a[d] * b[d];
    }
    return s;
}

inline float ivf_codebook_get(const float* codebook_ivf, size_t cb_dim, size_t c, size_t d){
    size_t block = (c / 4) * 4;
    size_t j = c % 4;
    return codebook_ivf[block * cb_dim + d * 4 + j];
}

inline float scalar_ip_centroid(const float* query, const float* codebook_ivf, size_t c, size_t cb_dim){
    float s = 0.0f;
    for(size_t d = 0; d < cb_dim; ++d){
        s += query[d] * ivf_codebook_get(codebook_ivf, cb_dim, c, d);
    }
    return s;
}

inline float ivf_base_get_transposed(const float* base_ivf, size_t ivf_dim, size_t start, size_t p, size_t d){
    size_t local = p - start;
    size_t block_local = (local / 4) * 4;
    size_t j = local % 4;
    return base_ivf[(start + block_local) * ivf_dim + d * 4 + j];
}

inline float ivf_base_get_standard(const float* base_ivf, size_t ivf_dim, size_t p, size_t d){
    return base_ivf[p * ivf_dim + d];
}
