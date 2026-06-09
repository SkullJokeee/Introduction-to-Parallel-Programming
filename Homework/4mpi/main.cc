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
#include "simd.h"
#include <pthread.h>
#include <queue>
#include <atomic>
#include <mpi.h>

// #define simd
// #define Pthread
// #define openMP

// #define mpi
#define mpi2
// #define openMP_mpi

// #define Query

// #define ivf
#define hnsw
// #define ivf_hnsw

using namespace hnswlib;

size_t test_number = 0, base_number = 0;
size_t test_gt_d = 0, vecdim = 0;
size_t pq_n = 0, cb_n = 0;
size_t pq_dim = 0, cb_dim = 0;
const size_t k = 10;
float* test_query;
int* test_gt;
float* base;
float* codebook_pq;
uint8_t* base_pq;
size_t ivf_n = 0;
size_t cb2_n = 128;
size_t cb2_dim = 0;
size_t ivf_dim = 0;
uint32_t* offset_ivf;
uint32_t* list_ivf;
float* base_ivf;
float* codebook_ivf;

const size_t nlist = 128; 
const size_t m = 4;
const size_t k_pq = 256;

#define THREAD_N 8

struct ThreadPool;
struct PoolArg{
    ThreadPool* pool;
    int id;
};

struct ThreadPool{
    std::vector<pthread_t> threads;
    std::vector<PoolArg> args;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t start_cond = PTHREAD_COND_INITIALIZER;
    pthread_cond_t done_cond = PTHREAD_COND_INITIALIZER;

    float* query;
    bool isStop = false;
    int active_threads = 0;
    int phase = 0;
    int current_job = 0;

    std::atomic<int> task_idx;
    int task_max;
    std::vector<uint32_t> active_clusters;

    float* lut = nullptr;
    float* cb_dis = nullptr;
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_rst;

    float* pq_cb = nullptr;
    uint8_t* base_pq = nullptr;

    ThreadPool() : threads(THREAD_N), args(THREAD_N), thread_rst(THREAD_N) {
        for(int i = 0; i < THREAD_N; i++){
            args[i].pool = this;
            args[i].id = i;
        }
    }

    ~ThreadPool() {
        free(lut);
        free(cb_dis);
    }

    void run_phase(int p, int max) {
        for(int i = 0; i < THREAD_N; i++){
            thread_rst[i] = std::priority_queue<std::pair<float, uint32_t>>();
        }
        task_idx.store(0);
        task_max = max;
        pthread_mutex_lock(&lock);
        phase = p;
        active_threads = THREAD_N;
        current_job++;
        pthread_cond_broadcast(&start_cond);
        while(active_threads > 0){
            pthread_cond_wait(&done_cond, &lock);
        }
        phase = 0;
        pthread_mutex_unlock(&lock);
    }
};

#if defined(simd) || defined(Query)
    #include "IVF_simd.h"
#endif

#ifdef hnsw
    #include "HNSW_simd.h"
#endif

#if defined(ivf_hnsw) && defined(simd)
    #include "IVF_HNSW_simd.h"
#endif

#if defined(mpi) && defined(ivf)
    #include "IVF_mpi.h"
#elif defined(mpi2) && defined(ivf)
    #include "IVF_mpi2.h"
#elif defined(openMP_mpi) && defined(ivf)
    #include "IVF_openMP_mpi.h"
#endif

#if defined(mpi) && defined(hnsw)
    #include "HNSW_MPI.h"
#elif defined(mpi2) && defined(hnsw)
    #include "HNSW_mpi2.h"
#elif defined(openMP_mpi) && defined(hnsw)
    #include "HNSWopenMP_mpi.h"
#endif

#if defined(mpi) && defined(ivf_hnsw)
    #include "IVF_HNSW_mpi.h"
#elif defined(mpi2) && defined(ivf_hnsw)
    #include "IVF_HNSW_mpi2.h"
#elif defined(openMP_mpi) && defined(ivf_hnsw)
    #include "IVF_HNSW_openMP_mpi.h"
#endif

#if defined(Pthread) && !defined(Query)
    #ifdef flat
        #include "flat_pthread.h"
    #elif defined(pq)
        #include "PQ_pthread.h"
    #elif defined(ivf)
        #include "IVF_pthread.h"
    #elif defined(pq_ivf)
        #include "PQ_IVF_pthread.h"
    #elif defined(ivf_pq)
        #include "IVF_PQ_pthread.h"
    #endif
#endif

#if defined(openMP) && !defined(Query)
    #ifdef flat
        #include "flat_openMP.h"
    #elif defined(pq)
        #include "PQ_openMP.h"
    #elif defined(ivf)
        #include "IVF_openMP.h"
    #elif defined(pq_ivf)
        #include "PQ_IVF_openMP.h"
    #elif defined(ivf_pq)
        #include "IVF_PQ_openMP.h"
    #endif
#endif

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
    int64_t latency;
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150;
    const int M = 16;

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

#if defined(Pthread) && defined(Query)
    struct ThreadArg{
        int start;
        int end;
        std::vector<SearchResult>* results;
    };

    void* thread_search(void* arg) {
        ThreadArg* args = (ThreadArg*)arg;
        const unsigned long Converter = 1000 * 1000;

        for(int i = args->start; i < args->end; i++) {
            struct timeval val;
            gettimeofday(&val, NULL);

            std::priority_queue<std::pair<float, uint32_t>> res;

            #ifdef flat_scan
                res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k); 
            #endif
            #ifdef flat
                res = flat_simd_search(base, test_query + i*vecdim, base_number, vecdim, k); 
            #endif
            #ifdef sq
                res = sq_search(base, test_query + i*vecdim, base_number, vecdim, k, sq_idx);
            #endif
            #ifdef pq
                res = pq_search(base, test_query + i*vecdim, cb_n, pq_n, vecdim, cb_dim, pq_dim, k, base_pq, codebook_pq);
            #endif
            #ifdef ivf
                res = ivf_search(base, test_query + i*vecdim, cb2_n, base_number, cb2_dim, ivf_dim, k, base_ivf, codebook_ivf, list_ivf, offset_ivf);
            #endif
            #ifdef pq_ivf
                res = pq_ivf_search(base, test_query + i*vecdim, nlist, base_number, vecdim, m, k_pq, k, codebook_ivf, codebook_pq, offset_ivf, list_ivf, base_pq);
            #endif
            #ifdef ivf_pq
                res = ivf_pq_search(base, test_query + i*vecdim, nlist, base_number, vecdim, m, k_pq, k, codebook_ivf, codebook_pq, offset_ivf, list_ivf, base_pq);
            #endif
            #ifdef hnsw
                res = hnsw_search(base, test_query + i*vecdim, base_number, vecdim, k);
            #endif
            #ifdef ivf_hnsw
                res = ivf_hnsw_search(base, test_query + i*vecdim, cb2_n, base_number, cb2_dim, ivf_dim, k, base_ivf, codebook_ivf, list_ivf, offset_ivf);
            #endif

            struct timeval newVal;
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

            std::set<uint32_t> gtset;
            for(int j = 0; j < k; ++j){
                int r = test_gt[j + i * test_gt_d];
                gtset.insert(r);
            }

            size_t acc = 0;
            while (res.size()) {   
                int x = res.top().second;
                if(gtset.find(x) != gtset.end()){
                    ++acc;
                }
                res.pop();
            }
            float recall = (float)acc / k;

            (*args->results)[i] = {recall, diff};
        }
        return nullptr;
    }

#endif

int main(int argc, char *argv[])
{
    #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)
        MPI_Init(&argc, &argv);
        int rank, nprocs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    #endif

    std::string data_path = "/anndata/";  
    test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 2000;
    
    pq_dim = 4;
    cb_dim = vecdim / pq_dim;
    cb2_n = 128;
    cb2_dim = vecdim;

    std::vector<SearchResult> results;
    results.resize(test_number);

    size_t tmp1 = 0, tmp2 = 0;

    #ifdef sq
        SQIndex sq_idx = build_sq_index(base, base_number, vecdim);
    #endif

    #ifdef pq
        codebook_pq = LoadData<float>("files/pq_codebook.bin", cb_n, cb_dim);
        base_pq = LoadData<uint8_t>("files/pq_base.bin", pq_n, pq_dim);
    #endif

    #ifdef ivf
        ivf_n = base_number;
        cb2_n = 0;
        ivf_dim = vecdim;
        cb2_dim = 0;

        #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)   
            if(rank == 0){
                codebook_ivf = LoadData<float>("files/ivf_codebook.bin", cb2_n, cb2_dim);
                offset_ivf = LoadData<uint32_t>("files/ivf_offset.bin", tmp1, tmp2);
                list_ivf   = LoadData<uint32_t>("files/ivf_ivflist.bin", tmp1, tmp2);
                base_ivf   = LoadData<float>("files/ivf_base.bin", tmp1, tmp2);
            }
            MPI_Bcast(&cb2_n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(&cb2_dim, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            if(rank != 0){
                codebook_ivf = new float[cb2_n * cb2_dim];
            }
            MPI_Bcast(codebook_ivf, cb2_n * cb2_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
            ivf_mpi_setup(base_ivf, list_ivf, offset_ivf, cb2_n, ivf_dim, rank, nprocs);
            #else
            codebook_ivf = LoadData<float>("files/ivf_codebook.bin", cb2_n, cb2_dim);
            offset_ivf = LoadData<uint32_t>("files/ivf_offset.bin", tmp1, tmp2);
            list_ivf   = LoadData<uint32_t>("files/ivf_ivflist.bin", tmp1, tmp2);
            base_ivf   = LoadData<float>("files/ivf_base.bin", tmp1, tmp2);
        #endif
    #endif

    #ifdef pq_ivf
        codebook_ivf = LoadData<float>("files/pqivf_ivf_cb.bin", tmp1, tmp2);
        codebook_pq  = LoadData<float>("files/pqivf_pq_cb.bin", tmp1, tmp2);
        offset_ivf   = LoadData<uint32_t>("files/pqivf_offset.bin", tmp1, tmp2);
        list_ivf     = LoadData<uint32_t>("files/pqivf_list.bin", tmp1, tmp2);
        base_pq      = LoadData<uint8_t>("files/pqivf_base.bin", tmp1, tmp2);
    #endif
    
    #ifdef ivf_pq
        codebook_ivf = LoadData<float>("files/ivfpq_ivf_cb.bin", tmp1, tmp2);
        codebook_pq  = LoadData<float>("files/ivfpq_pq_cb.bin", tmp1, tmp2);
        offset_ivf   = LoadData<uint32_t>("files/ivfpq_offset.bin", tmp1, tmp2);
        list_ivf     = LoadData<uint32_t>("files/ivfpq_list.bin", tmp1, tmp2);
        base_pq      = LoadData<uint8_t>("files/ivfpq_base.bin", tmp1, tmp2);
    #endif

    #ifdef hnsw
        #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)
            hnsw_mpi_setup(base, base_number, vecdim, rank, nprocs);
        #else
            std::ifstream fin("files/hnsw.index", std::ios::binary);
            if(!fin.good()){
                build_index(base, base_number, vecdim);
            }
            fin.close();
            LoadHNSWData(vecdim);
        #endif
    #endif

    #ifdef ivf_hnsw
        ivf_n = base_number;
        cb2_n = 0;
        ivf_dim = vecdim;
        cb2_dim = 0;

        #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)   
            if(rank == 0){
                codebook_ivf = LoadData<float>("files/ivf_codebook.bin", cb2_n, cb2_dim);
                offset_ivf = LoadData<uint32_t>("files/ivf_offset.bin", tmp1, tmp2);
                list_ivf   = LoadData<uint32_t>("files/ivf_ivflist.bin", tmp1, tmp2);
                base_ivf   = LoadData<float>("files/ivf_base.bin", tmp1, tmp2);
            }
            MPI_Bcast(&cb2_n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(&cb2_dim, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            if(rank != 0){
                codebook_ivf = new float[cb2_n * cb2_dim];
            }
            MPI_Bcast(codebook_ivf, cb2_n * cb2_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
            ivf_hnsw_mpi_setup(base_ivf, offset_ivf, cb2_n, ivf_dim, rank, nprocs);
        #else
            codebook_ivf = LoadData<float>("files/ivf_codebook.bin", cb2_n, cb2_dim);
            offset_ivf = LoadData<uint32_t>("files/ivf_offset.bin", tmp1, tmp2);
            list_ivf   = LoadData<uint32_t>("files/ivf_ivflist.bin", tmp1, tmp2);
            base_ivf   = LoadData<float>("files/ivf_base.bin", tmp1, tmp2);
            BuildClusterHNSWIndices(base_ivf, offset_ivf, cb2_n, ivf_dim);
        #endif
    #endif
    

    #if defined(openMP) || defined(openMP_mpi)
        omp_set_num_threads(THREAD_N); 
    #endif
    
    struct timeval total_start_time;
    gettimeofday(&total_start_time, NULL);

    #if defined(Pthread) && !defined(Query)
        ThreadPool pool;
        cb_n = m * k_pq;
        pool.lut = align<float>(cb_n);

        pool.cb_dis = align<float>(cb2_n);

        pool.pq_cb = codebook_pq; 
        pool.base_pq = base_pq;
        
        for(int i = 0; i < THREAD_N; i++){
            pthread_create(&pool.threads[i], NULL, thread, &pool.args[i]);
        }

    #endif

    #if defined(Pthread) && defined(Query)
        int num = THREAD_N;
        std::vector<pthread_t> threads(num);
        std::vector<ThreadArg> thread_args(num);
        
        int size = test_number / num;

        for(int i = 0; i < num; i++){
            thread_args[i].start = i * size;
            thread_args[i].end = (i + 1) * size;
            thread_args[i].results = &results;

            pthread_create(&threads[i], NULL, thread_search, &thread_args[i]);
        }

        for(int i = 0; i < num; i++){
            pthread_join(threads[i], NULL);
        }

    #endif
        
    #if defined(openMP) && defined(Query)
        #pragma omp parallel for
    #endif
    #if !(defined(Pthread) && defined(Query))
    for(int i = 0; i < test_number; ++i) {
        #ifdef Pthread
        pool.query = test_query + i*vecdim;
        #endif

        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        std::priority_queue<std::pair<float, uint32_t>> res;

        #ifdef flat_scan
            res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k); 
        #endif

        #ifdef flat
            res = flat_simd_search(base, test_query + i*vecdim, base_number, vecdim, k); 
        #endif

        #ifdef sq
            res = sq_search(base, test_query + i*vecdim, base_number, vecdim, k, sq_idx);
        #endif

        #ifdef pq
            #if defined(simd) || defined(openMP)
                res = pq_search(base, test_query + i*vecdim, cb_n, base_number, vecdim, cb_dim, pq_dim, k, base_pq, codebook_pq);
            #endif
            #ifdef Pthread
                res = pq_search(&pool);
            #endif
        #endif

        #ifdef ivf
            #if defined(simd) || defined(openMP)
                res = ivf_search(base, test_query + i*vecdim, cb2_n, base_number, cb2_dim, ivf_dim, k, base_ivf, codebook_ivf, list_ivf, offset_ivf);
            #endif
            #ifdef Pthread
                res = ivf_search(&pool);
            #endif
            #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)   
                res = ivf_mpi_search(test_query + i*vecdim, cb2_n, ivf_dim, k, codebook_ivf, rank, nprocs);
            #endif
        #endif

        #ifdef ivf_hnsw
            #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)
                res = ivf_hnsw_mpi_search(test_query + i*vecdim, cb2_n, ivf_dim, k, rank, nprocs);
            #else
                res = ivf_hnsw_search(base, test_query + i*vecdim, cb2_n, base_number, cb2_dim, ivf_dim, k, base_ivf, codebook_ivf, list_ivf, offset_ivf);
            #endif
        #endif

        #ifdef pq_ivf
            #if defined(simd) || defined(openMP)
                res = pq_ivf_search(base, test_query + i*vecdim, cb2_n, base_number, vecdim, m, k_pq, k, codebook_ivf, codebook_pq, offset_ivf, list_ivf, base_pq);
            #endif
            #ifdef Pthread
                res = pq_ivf_search(&pool);
            #endif
        #endif
        
        #ifdef ivf_pq
            #if defined(simd) || defined(openMP)
                res = ivf_pq_search(base, test_query + i*vecdim, cb2_n, base_number, vecdim, m, k_pq, k, codebook_ivf, codebook_pq, offset_ivf, list_ivf, base_pq);
            #endif
            #ifdef Pthread
                res = ivf_pq_search(&pool);
            #endif
        #endif

        #ifdef hnsw
            #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)
                res = hnsw_mpi_search(test_query + i*vecdim, vecdim, k, rank, nprocs);
            #else
                res = hnsw_search(base, test_query + i*vecdim, base_number, vecdim, k);
            #endif
        #endif

        #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)
        if(rank == 0){
        #endif
        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while(res.size()){
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
        #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)   
        }
        #endif
    }
    #endif

    #if defined(Pthread) && !defined(Query)
        pthread_mutex_lock(&pool.lock);
        pool.isStop = true;
        pthread_cond_broadcast(&pool.start_cond);
        pthread_mutex_unlock(&pool.lock);

        for(int i = 0; i < THREAD_N; i++){
            pthread_join(pool.threads[i], NULL);
        }
    #endif

    struct timeval total_end_time;
    gettimeofday(&total_end_time, NULL);
    
    const unsigned long Converter = 1000 * 1000;
    int64_t total_latency = (total_end_time.tv_sec * Converter + total_end_time.tv_usec) - (total_start_time.tv_sec * Converter + total_start_time.tv_usec);

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    #ifdef pq
        delete[] codebook_pq;
        delete[] base_pq;
    #endif

    #ifdef ivf
        #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)
            ivf_mpi_cleanup();
        #endif
        delete[] codebook_ivf;
        delete[] offset_ivf;
        delete[] list_ivf;
        delete[] base_ivf;
    #endif

    #ifdef ivf_hnsw
        #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)
            ivf_hnsw_mpi_cleanup();
        #endif
        delete[] codebook_ivf;
        delete[] offset_ivf;
        delete[] list_ivf;
        delete[] base_ivf;
    #endif

    #ifdef hnsw
        #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)
            hnsw_mpi_cleanup();
        #endif
    #endif

    #if defined(pq_ivf) || defined(ivf_pq)
        delete[] codebook_ivf;
        delete[] codebook_pq;
        delete[] offset_ivf;
        delete[] list_ivf;
        delete[] base_pq;
    #endif

    #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)   
        if(rank == 0){
    #endif
    
        std::cout << "average recall: " << avg_recall / test_number << "\n";
        std::cout << "average single query latency (us): " << avg_latency / test_number << "\n";

        std::cout << "total time for all queries (us): " << total_latency << "\n"; 
        std::cout << "average time per query (us): " << total_latency / test_number << "\n";
    
    #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)   
        }
    #endif

    #if defined(mpi) || defined(mpi2) || defined(openMP_mpi)   
        MPI_Finalize();
    #endif

    return 0;
}