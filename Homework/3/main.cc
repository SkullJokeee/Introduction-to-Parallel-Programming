#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 可以自行添加需要的头文件
#include "simd.h"
#include "SQ_simd.h"
#include "PQ_simd.h"

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    /////////
    size_t n = 100000, d = 96;
    float* mock_test = new float[test_number * d];
    int* mock_test_gt = new int[test_number * d];
    float* mock_base = new float[n * d];
    for(size_t i = 0; i < test_number * d; ++i){
        mock_test[i] = (float)rand() / RAND_MAX;
        mock_test_gt[i] = rand() % n;
    }
    for(size_t i = 0; i < n * d; ++i){
        mock_base[i] = (float)rand() / RAND_MAX;
    }
    test_query = mock_test;
    test_gt = mock_test_gt;
    base = mock_base;
    base_number = n;
    vecdim = 96;
    ////////

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    ////////
    SQIndex sq_idx = build_sq_index(base, base_number, vecdim);

    size_t pq_n = 0, cb_n = 0;
    size_t pq_dim = 0, cb_dim = 0;

    auto codebook_pq = LoadData<float>("files/pq_codebook.bin", cb_n, cb_dim);      // 4*256个24维向量
    auto base_pq = LoadData<uint8_t>("files/pq_base.bin", pq_n, pq_dim);    // base_number个4维向量

    auto aligned_base = align<float>(base, base_number * vecdim);
    auto aligned_query = align<float>(test_query, test_number * vecdim); ////0509
    

    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        // auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k); 
        // auto res = flat_simd_search(base, test_query + i*vecdim, base_number, vecdim, k); 
        // auto res = sq_search(base, test_query + i*vecdim, base_number, vecdim, k, sq_idx);
        // auto res = pq_adc_search(base, test_query + i*vecdim, cb_n, pq_n, vecdim, cb_dim, pq_dim, k, base_pq, codebook_pq);
        // auto res = flat_search(aligned_base, aligned_query + i*vecdim, base_number, vecdim, k); 
        // auto res = flat_simd_search(aligned_base, aligned_query + i*vecdim, base_number, vecdim, k); 
        // auto res = sq_search(aligned_base, aligned_query + i*vecdim, base_number, vecdim, k, sq_idx);
        auto res = pq_adc_search(aligned_base, aligned_query + i*vecdim, cb_n, pq_n, vecdim, cb_dim, pq_dim, k, base_pq, codebook_pq);
        ////////

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    free(aligned_base);
    free(aligned_query);
    return 0;
}
