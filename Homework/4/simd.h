#pragma once
#include <queue>
#include <cassert>
#include <arm_neon.h>
#include <cstdlib>


struct simd8float32 {  // 用于高效处理8个32位浮点数的并行计算。
    float32x4x2_t data;  // 两个维度，分别储存128位浮点数

    simd8float32() = default;

    explicit simd8float32(const float* x)
        : data{vld1q_f32(x), vld1q_f32(x + 4)} {}
    
    explicit simd8float32(float value)
        : data{vdupq_n_f32(value), vdupq_n_f32(value)} {}

    simd8float32 operator*(const simd8float32& other) const{
        simd8float32 rst;
        rst.data.val[0] = vmulq_f32(this->data.val[0], other.data.val[0]); // 逐元素乘法
        rst.data.val[1] = vmulq_f32(this->data.val[1], other.data.val[1]);
        
        return rst;
    }

    simd8float32 operator+(const simd8float32& other) const{
        simd8float32 rst;
        rst.data.val[0] = vaddq_f32(this->data.val[0], other.data.val[0]); // 逐元素加法
        rst.data.val[1] = vaddq_f32(this->data.val[1], other.data.val[1]);
        
        return rst;
    }

    void innerProduct(simd8float32& a, simd8float32& b){
        this->data.val[0] = vmlaq_f32(this->data.val[0], a.data.val[0], b.data.val[0]);
        this->data.val[1] = vmlaq_f32(this->data.val[1], a.data.val[1], b.data.val[1]);
    }

    void storeu(float* dst){
        vst1q_f32(dst, data.val[0]);
        vst1q_f32(dst + 4, data.val[1]);
    }

    float getSum(){
        float rst = vaddvq_f32(this->data.val[0]);
        rst += vaddvq_f32(this->data.val[1]);
        return rst;
    }

};

float InnerProductSIMDNeon8(const float* b1, const float* b2, size_t vecdim) {
    assert(vecdim % 8 == 0); // 假设维度能被8整除

    simd8float32 sum(0.0); // 8xfloat32全部初始化为0
    for (int i = 0; i < vecdim; i += 8) {
        simd8float32 s1(b1 + i), s2(b2 + i);
        sum.innerProduct(s1, s2);
    }

    float dis = sum.getSum();

    return 1 - dis;
}

float InnerProductSIMDNeon(const float* b1, const float* b2, size_t vecdim) {
    assert(vecdim % 8 == 0); // 假设维度能被8整除

    // b1 = (float*)__builtin_assume_aligned(b1, 32);
    // b2 = (float*)__builtin_assume_aligned(b2, 32);

    assert(vecdim % 32 == 0);
    
    simd8float32 sum1(0.0);
    simd8float32 sum2(0.0);
    simd8float32 sum3(0.0);
    simd8float32 sum4(0.0);

    for (int i = 0; i < vecdim; i += 32) {
        simd8float32 s11(b1 + i), s12(b2 + i);
        simd8float32 s21(b1 + i + 8), s22(b2 + i + 8);
        simd8float32 s31(b1 + i + 16), s32(b2 + i + 16);
        simd8float32 s41(b1 + i + 24), s42(b2 + i + 24);

        sum1.innerProduct(s11, s12);
        sum2.innerProduct(s21, s22);
        sum3.innerProduct(s31, s32);
        sum4.innerProduct(s41, s42);
    }

    float32x4_t vsum1 = vaddq_f32(sum1.data.val[0], sum1.data.val[1]); ////
    float32x4_t vsum2 = vaddq_f32(sum2.data.val[0], sum2.data.val[1]);
    float32x4_t vsum3 = vaddq_f32(sum3.data.val[0], sum3.data.val[1]);
    float32x4_t vsum4 = vaddq_f32(sum4.data.val[0], sum4.data.val[1]);

    float32x4_t vsum5 = vaddq_f32(vsum1, vsum2);
    float32x4_t vsum6 = vaddq_f32(vsum3, vsum4);
    float32x4_t sum = vaddq_f32(vsum5, vsum6);

    return 1 - vaddvq_f32(sum);

}

uint32_t EucDisSIMDNeon(const uint8_t* b1, const uint8_t* b2, size_t vecdim) {
    assert(vecdim % 16 == 0);
    assert(vecdim % 32 == 0);

    // b1 = (uint8_t*)__builtin_assume_aligned(b1, 32);
    // b2 = (uint8_t*)__builtin_assume_aligned(b2, 32);

    uint32x4_t sum1 = vdupq_n_u32(0);
    uint32x4_t sum2 = vdupq_n_u32(0);

    for (int i = 0; i < vecdim; i += 32) { 
        uint8x16_t v11 = vld1q_u8(b1 + i);
        uint8x16_t v12 = vld1q_u8(b2 + i);
        uint8x16_t v21 = vld1q_u8(b1 + i + 16);
        uint8x16_t v22 = vld1q_u8(b2 + i + 16);
        
        uint8x16_t diff1 = vabdq_u8(v11, v12); 
        uint8x16_t diff2 = vabdq_u8(v21, v22); 
        
        uint16x8_t low1 = vmull_u8(vget_low_u8(diff1), vget_low_u8(diff1));
        uint16x8_t high1 = vmull_u8(vget_high_u8(diff1), vget_high_u8(diff1));
        uint16x8_t low2 = vmull_u8(vget_low_u8(diff2), vget_low_u8(diff2));
        uint16x8_t high2 = vmull_u8(vget_high_u8(diff2), vget_high_u8(diff2));
        
        sum1 = vpadalq_u16(sum1, low1);
        sum1 = vpadalq_u16(sum1, high1);
        sum2 = vpadalq_u16(sum2, low2);
        sum2 = vpadalq_u16(sum2, high2);
    }
    
    uint32x4_t sum = sum1 + sum2;
    uint32_t rst = vgetq_lane_u32(sum, 0) + vgetq_lane_u32(sum, 1) + vgetq_lane_u32(sum, 2) + vgetq_lane_u32(sum, 3);

    return rst;

}

template <typename T>
T* align(size_t num){
    size_t alignment = 32;
    size_t num_byte = num * sizeof(T);
    size_t pad = (num_byte + alignment - 1) & (~(alignment - 1)); // 补到最近一个更大的32的倍数
    return (T*)(aligned_alloc(alignment, pad));
}

template <typename T>
T* align(T* element, size_t num){
    size_t alignment = 32;
    size_t num_byte = num * sizeof(T);
    size_t pad = (num_byte + alignment - 1) & (~(alignment - 1));

    T* rst = (T*)aligned_alloc(alignment, pad);
    std::memcpy(rst, element, num_byte);

    return rst;
}