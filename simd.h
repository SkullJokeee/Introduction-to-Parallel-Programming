#pragma once
#include <queue>
#include <cassert>
#include <arm_neon.h>


struct simd8float32 {  // 用于高效处理8个32位浮点数的并行计算。
    float32x4x2_t data;  // 两个维度，分别储存128位浮点数

    simd8float32() = default;

    explicit simd8float32(const float* x) // 结构体的构造函数，接受一个指向float数组的指针，并使用NEON指令将数据加载到SIMD寄存器中
        : data{vld1q_f32(x), vld1q_f32(x + 4)} {} // vld1q_f32是NEON指令，用于加载4个float32到SIMD寄存器中
    
    explicit simd8float32(float value) // 另一个构造函数，接受一个float值，并将其广播到SIMD寄存器中的所有元素
        : data{vdupq_n_f32(value), vdupq_n_f32(value)} {} // vdupq_n_f32是NEON指令，用于将一个float值广播到SIMD寄存器中的所有元素

    // explicit simd8float32(uint8_t value)
    //     : data{vdupq_n_f32(value), vdupq_n_f32(value)} {}

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
        vmlaq_f32(this->data.val[0], a.data.val[0], b.data.val[0]);
        vmlaq_f32(this->data.val[1], a.data.val[1], b.data.val[1]);
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


float InnerProductSIMDNeon(const float* b1, const float* b2, size_t vecdim) { // 计算内积
    assert(vecdim % 8 == 0); // 假设维度能被8整除

    simd8float32 sum(0.0); // 8xfloat32全部初始化为0
    for (int i = 0; i < vecdim; i += 8) {
        // simd8float32 s1(b1 + i), s2(b2 + i);
        // simd8float32 m = s1 * s2;
        // sum += m;

        simd8float32 s1(b1 + i), s2(b2 + i);
        sum.innerProduct(s1, s2);
    }

    // float tmp[8];
    // sum.storeu(tmp);
    // float dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    
    float dis = sum.getSum();

    return 1 - dis;
}

uint32_t InnerProductSIMDNeon(const uint8_t* b1, const uint8_t* b2, size_t vecdim) { // 计算内积
    assert(vecdim % 16 == 0);

    uint32x4_t sum = vdupq_n_u32(0);

    for (int i = 0; i < vecdim; i += 16) { 
        uint8x16_t v1 = vld1q_u8(b1 + i);
        uint8x16_t v2 = vld1q_u8(b2 + i);
        
        uint16x8_t low = vmull_u8(vget_low_u8(v1), vget_low_u8(v2));
        uint16x8_t high = vmull_u8(vget_high_u8(v1), vget_high_u8(v2));
        
        sum = vpadalq_u16(sum, low);
        sum = vpadalq_u16(sum, high);
    }
    
    uint32_t rst = vgetq_lane_u32(sum, 0) + vgetq_lane_u32(sum, 1) + vgetq_lane_u32(sum, 2) + vgetq_lane_u32(sum, 3);

    return rst;
}
