#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>

#define HNSW_EF_SEARCH 200

using namespace std;

const size_t k = 10;
const size_t thread_n = 8;

float* query = nullptr;
int* gt = nullptr;
float* base = nullptr;
size_t query_n = 0;
size_t base_n = 0;
size_t gt_d = 0;
size_t vecdim = 0;

float* codebook_ivf = nullptr;
uint32_t* offset_ivf = nullptr;
uint32_t* list_ivf = nullptr;
float* base_ivf = nullptr;
float* codebook_pq = nullptr;
uint8_t* base_pq = nullptr;
size_t ivf_cb_n = 128;
size_t ivf_cb_dim = 0;
size_t ivf_dim = 0;
size_t pq_n = 0;
size_t cb_n = 0;
size_t pq_dim = 0;
size_t cb_dim = 0;
const size_t nlist = 128;
const size_t m = 4;
const size_t k_pq = 256;

#if !defined(ALG_FLAT) && !defined(ALG_IVF) && !defined(ALG_PQ) && !defined(ALG_PQ_IVF) && !defined(ALG_IVF_PQ) && !defined(ALG_HNSW)
#define ALG_IVF
#endif

#ifndef MODE_SERIAL
#define MODE_SERIAL
#endif

#ifdef ALG_FLAT
    #include "flat_serial.h"
#endif

#ifdef ALG_IVF
    #include "ivf_serial.h"
#endif

#ifdef ALG_PQ
    #include "pq_serial.h"
#endif

#ifdef ALG_PQ_IVF
    #include "pq_ivf_serial.h"
#endif

#ifdef ALG_IVF_PQ
    #include "ivf_pq_serial.h"
#endif

#ifdef ALG_HNSW
    #include "hnsw_serial.h"
#endif

struct SearchResult {
    float recall;
    int64_t latency;
};

template<typename T>
T *LoadData(string data_path, size_t& n, size_t& d){
    ifstream fin;
    fin.open(data_path, ios::in | ios::binary);
    if(!fin.is_open()){
        cerr << "ERROR: cannot open " << data_path << "\n";
        exit(1);
    }
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    if(!fin){
        cerr << "ERROR: cannot read header from " << data_path << "\n";
        exit(1);
    }
    T* data = new T[n * d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i * d * sz), d * sz);
        if(!fin){
            cerr << "ERROR: cannot read data row " << i << " from " << data_path << "\n";
            exit(1);
        }
    }
    fin.close();

    cerr << "load data " << data_path << "\n";
    cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";

    return data;
}

SearchResult run_one_query(size_t query_idx){
    const unsigned long Converter = 1000 * 1000;
    struct timeval val;
    gettimeofday(&val, NULL);

    priority_queue<pair<float, uint32_t>> res;

    #ifdef ALG_FLAT
        res = flat_serial_search(base, query + query_idx * vecdim, base_n, vecdim, k);
    #endif

    #ifdef ALG_IVF
        res = ivf_serial_search(base, query + query_idx * vecdim, ivf_cb_n, base_n, ivf_cb_dim, ivf_dim, k, base_ivf, codebook_ivf, list_ivf, offset_ivf);
    #endif

    #ifdef ALG_PQ
        res = pq_serial_search(base, query + query_idx * vecdim, cb_n, pq_n, vecdim, cb_dim, pq_dim, k, base_pq, codebook_pq);
    #endif

    #ifdef ALG_PQ_IVF
        res = pq_ivf_serial_search(base, query + query_idx * vecdim, nlist, base_n, vecdim, m, k_pq, k, codebook_ivf, codebook_pq, offset_ivf, list_ivf, base_pq);
    #endif

    #ifdef ALG_IVF_PQ
        res = ivf_pq_serial_search(base, query + query_idx * vecdim, nlist, base_n, vecdim, m, k_pq, k, codebook_ivf, codebook_pq, offset_ivf, list_ivf, base_pq);
    #endif

    #ifdef ALG_HNSW
        res = hnsw_serial_search(query + query_idx * vecdim, vecdim, k);
    #endif

    struct timeval newVal;
    gettimeofday(&newVal, NULL);
    int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

    set<uint32_t> gtset;
    for(size_t j = 0; j < k; ++j){
        gtset.insert(gt[j + query_idx * gt_d]);
    }

    size_t acc = 0;
    while(res.size()){
        uint32_t x = res.top().second;
        if(gtset.find(x) != gtset.end()){
            ++acc;
        }
        res.pop();
    }
    float recall = (float)acc / k;

    return {recall, diff};
}

#if defined(MODE_PTHREAD)
struct ThreadArg {
    size_t start;
    size_t end;
    vector<SearchResult>* results;
};

void* thread_search(void* arg){
    ThreadArg* args = (ThreadArg*)arg;
    for(size_t i = args->start; i < args->end; ++i){
        (*args->results)[i] = run_one_query(i);
    }
    return nullptr;
}
#endif

int main(int argc, char *argv[]){
    string data_path = "anndata/";

    query = LoadData<float>(data_path + "DEEP100K.query.fbin", query_n, vecdim);
    gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", query_n, gt_d);
    base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_n, vecdim);

    query_n = 2000;

    pq_dim = 4;
    cb_dim = vecdim / pq_dim;
    ivf_cb_dim = vecdim;
    ivf_dim = vecdim;

    size_t temp_n = 0, temp_d = 0;

    #ifdef ALG_IVF
        codebook_ivf = LoadData<float>("files/ivf_codebook.bin", ivf_cb_n, ivf_cb_dim);
        offset_ivf = LoadData<uint32_t>("files/ivf_offset.bin", temp_n, temp_d);
        list_ivf   = LoadData<uint32_t>("files/ivf_ivflist.bin", temp_n, temp_d);
        base_ivf   = LoadData<float>("files/ivf_base.bin", temp_n, temp_d);
    #endif

    #ifdef ALG_PQ
        codebook_pq = LoadData<float>("files/pq_codebook.bin", cb_n, cb_dim);
        base_pq = LoadData<uint8_t>("files/pq_base.bin", pq_n, pq_dim);
    #endif

    #ifdef ALG_PQ_IVF
        codebook_ivf = LoadData<float>("files/pqivf_ivf_cb.bin", temp_n, temp_d);
        codebook_pq  = LoadData<float>("files/pqivf_pq_cb.bin", temp_n, temp_d);
        offset_ivf   = LoadData<uint32_t>("files/pqivf_offset.bin", temp_n, temp_d);
        list_ivf     = LoadData<uint32_t>("files/pqivf_list.bin", temp_n, temp_d);
        base_pq      = LoadData<uint8_t>("files/pqivf_base.bin", temp_n, temp_d);
        cb_n = m * k_pq;
        pq_n = base_n;
    #endif

    #ifdef ALG_IVF_PQ
        codebook_ivf = LoadData<float>("files/ivfpq_ivf_cb.bin", temp_n, temp_d);
        codebook_pq  = LoadData<float>("files/ivfpq_pq_cb.bin", temp_n, temp_d);
        offset_ivf   = LoadData<uint32_t>("files/ivfpq_offset.bin", temp_n, temp_d);
        list_ivf     = LoadData<uint32_t>("files/ivfpq_list.bin", temp_n, temp_d);
        base_pq      = LoadData<uint8_t>("files/ivfpq_base.bin", temp_n, temp_d);
    #endif

    #ifdef ALG_HNSW
        ifstream fin("files/hnsw.index", ios::binary);
        if(!fin.good()){
            cerr << "HNSW index not found, building...\n";
            build_hnsw_index(base, base_n, vecdim, "files/hnsw.index");
        }
        fin.close();
        load_hnsw_index(vecdim, "files/hnsw.index", HNSW_EF_SEARCH);
    #endif

    vector<SearchResult> results(query_n);

    #ifdef MODE_OPENMP
        omp_set_num_threads(thread_n);
    #endif

    const unsigned long Converter = 1000 * 1000;
    struct timeval total_start_time;
    gettimeofday(&total_start_time, NULL);

    #ifdef MODE_SERIAL
        for(size_t i = 0; i < query_n; ++i){
            results[i] = run_one_query(i);
        }
    #endif

    #ifdef MODE_OPENMP
        #pragma omp parallel for
        for(size_t i = 0; i < query_n; ++i){
            results[i] = run_one_query(i);
        }
    #endif

    #ifdef MODE_PTHREAD
        vector<pthread_t> threads(thread_n);
        vector<ThreadArg> thread_args(thread_n);
        size_t chunk = query_n / thread_n;

        for(size_t t = 0; t < thread_n; ++t){
            thread_args[t].start = t * chunk;
            thread_args[t].end = (t == thread_n - 1) ? query_n : (t + 1) * chunk;
            thread_args[t].results = &results;
            pthread_create(&threads[t], NULL, thread_search, &thread_args[t]);
        }

        for(size_t t = 0; t < thread_n; ++t){
            pthread_join(threads[t], NULL);
        }
    #endif

    struct timeval total_end_time;
    gettimeofday(&total_end_time, NULL);
    int64_t total_latency = (total_end_time.tv_sec * Converter + total_end_time.tv_usec) - (total_start_time.tv_sec * Converter + total_start_time.tv_usec);

    float avg_recall = 0, avg_latency = 0;
    for(size_t i = 0; i < query_n; ++i){
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    #ifdef ALG_IVF
        delete[] codebook_ivf;
        delete[] offset_ivf;
        delete[] list_ivf;
        delete[] base_ivf;
    #endif

    #ifdef ALG_PQ
        delete[] codebook_pq;
        delete[] base_pq;
    #endif

    #if defined(ALG_PQ_IVF) || defined(ALG_IVF_PQ)
        delete[] codebook_ivf;
        delete[] codebook_pq;
        delete[] offset_ivf;
        delete[] list_ivf;
        delete[] base_pq;
    #endif

    #ifdef ALG_HNSW
        free_hnsw_index();
    #endif

    delete[] query;
    delete[] gt;
    delete[] base;

    cout << "average recall: " << avg_recall / query_n << "\n";
    cout << "average single query latency (us): " << avg_latency / query_n << "\n";
    cout << "total time for all queries (us): " << total_latency << "\n";
    cout << "average time per query (us): " << total_latency / query_n << "\n";

    return 0;
}
