#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <queue>
#include <cublas_v2.h>
#include <cuda_runtime.h>


#define k 10

size_t test_number = 0, base_number = 0;
size_t test_gt_d = 0, vecdim = 0;
float* test_query;
int* test_gt;
float* base;

struct SearchResult{
    float recall;
    int64_t latency;
};

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

__global__ void select_kernel(const float* d_dist, int* d_res_idx, int base_number, int test_number){
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(q_idx >= test_number){
        return;
    }
    
    const float* q_dist = d_dist + q_idx * base_number;
    int* q_res = d_res_idx + q_idx * k;
    
    float top_val[k];
    int top_idx[k];
    
    for(int i = 0; i < k; ++i){
        top_val[i] = -1e9f;
        top_idx[i] = -1;
    }
    
    for(int j = 0; j < base_number; ++j){
        float val = q_dist[j];
        if(val > top_val[k - 1]){
            int p = k - 1;
            while(p > 0 && val > top_val[p - 1]){
                top_val[p] = top_val[p - 1];
                top_idx[p] = top_idx[p - 1];
                p--;
            }
            top_val[p] = val;
            top_idx[p] = j;
        }
    }
    
    for(int i = 0; i < k; ++i){
        q_res[i] = top_idx[i];
    }
}

int main(int argc, char *argv[]){
    std::string data_path = "anndata/";  
    
    size_t cb_n = 0, cb_dim = 0;
    size_t offset_n = 0, offset_d = 0;
    size_t list_n = 0, list_d = 0;

    test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);


    test_number = 2000;
    std::vector<SearchResult> results;
    results.resize(test_number);
    
    float* d_base;
    float* d_query;
    float* d_dist;
    int* d_res_idx;
    
    cudaMalloc(&d_base, base_number * vecdim * sizeof(float));
    cudaMalloc(&d_query, test_number * vecdim * sizeof(float));
    cudaMalloc(&d_dist, base_number * test_number * sizeof(float));
    cudaMalloc(&d_res_idx, test_number * k * sizeof(int));
    
    cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, test_query, test_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;

        
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, base_number, test_number, vecdim, &alpha, d_base, vecdim, d_query, vecdim, &beta, d_dist, base_number);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (test_number + threadsPerBlock - 1) / threadsPerBlock;
    
    auto total_start_time = std::chrono::high_resolution_clock::now();

    select_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dist, d_res_idx, base_number, test_number);
    cudaDeviceSynchronize();
    cublasDestroy(handle);


    int* res_idx = new int[test_number * k];
    cudaMemcpy(res_idx, d_res_idx, test_number * k * sizeof(int), cudaMemcpyDeviceToHost);
    auto total_end_time = std::chrono::high_resolution_clock::now();

    int64_t total_latency = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time).count();
    int64_t avg_simulated_latency = total_latency / test_number;
    
    for(int i = 0; i < test_number; i++){
        std::set<uint32_t> gtset;
        for(int j = 0; j < k; j++){
            int t = test_gt[j + i * test_gt_d];
            gtset.insert(t);
        }
        size_t acc = 0;
        for(int j = 0; j < k; j++){
            int x = res_idx[i * k + j];
            if(gtset.find(x) != gtset.end()){
                acc++;
            }
        }
        float recall = (float)acc / k;
        results[i] = {recall, avg_simulated_latency};
    }
    
    float avg_recall = 0, avg_latency = 0;
    
    for(int i = 0; i < test_number; i++){
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }
    
    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average single query latency (us): " << avg_latency / test_number << "\n";
    std::cout << "total time for all queries (us): " << total_latency << "\n"; 
    std::cout << "average time per query (us): " << total_latency / test_number << "\n";
    
    delete[] res_idx;
    
    cudaFree(d_base);
    cudaFree(d_query);
    cudaFree(d_dist);
    cudaFree(d_res_idx);

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
}