//
// Created by shiting on 2025-04-09.
//
#include <iostream>
#include <vector>
#include "omp.h"
#include <random>
#include <iomanip>
#include <algorithm>
#include "chrono"
#include <getopt.h>
#include <cstdlib>

#define RP 10
double k = 0.5; // heat transfer coefficient
double dt = 1.; // time step
double dx = 1.; // grid spacing

void vector_double_define_random(double* phi, int start, int end) {
    std::random_device seed;
    std::mt19937 randomInt(seed());
    std::uniform_real_distribution<> unit(1.0, 10.0);

    for (int i = start; i < end; i++)
        phi[i] = unit(randomInt);

}

double computeMedian(std::vector<long>& nums) {
    if (nums.empty()) {
        return 0.0;  // Handle empty vector case
    }

    // Make a copy to avoid modifying the original vector
    std::vector<long> sorted = nums;
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    if (n % 2 == 0) {  // Even number of elements
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {  // Odd number of elements
        return sorted[n/2];
    }
}

void define_uniform(double* phi, int start, int end){
    for (int i = start; i < end; ++i) {
        phi[i] = i;
    }
}

static inline double heat(double left, double mid, double right)
{
    return mid+(k*dt/dx*dx)*(left-2*mid+right);
}


static inline void heat_part( int size, double* next,
                              double* left,
                              double *mid, double *right, std::chrono::steady_clock::time_point *task_end, long *task_duration)
{
    int tid = omp_get_thread_num();
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    next[0] = heat(left[size-1], mid[0], mid[1]);
    for (int i = 1; i < size-1; ++i)
        next[i] = heat(mid[i-1], mid[i], mid[i+1]);
    next[size-1] = heat(mid[size-2], mid[size-1],
                        right[0]);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    task_end[tid] = end;
    task_duration[tid] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}

//idx does the wrapping here
static inline int idx(int i, int size)
{
    return (i < 0) ? (i + size) % size : i % size;
}

void stencil(int nt, int np, int nx, double *current, double *next, double **U,
             std::chrono::steady_clock::time_point *task_end, std::chrono::steady_clock::time_point *taskwait,
             long *task_duration){
    for (int t = 0; t < nt; t++) {
        for (int i = 0; i < np; ++i) {
#pragma omp task depend(out: next[i*nx]) \
depend(in: current[idx(i-1, np)*nx], \
current[i*nx], current[idx(i+1, np)*nx])
            heat_part(nx, &next[i * nx],
                      &current[idx(i - 1, np) * nx],
                      &current[i * nx],
                      &current[idx(i + 1, np) * nx],
                      task_end, task_duration);
        }
        current = U[(t + 1) % 2];
        next = U[t % 2];
    }
#pragma omp taskwait
    *taskwait = std::chrono::steady_clock::now();
}


int main(int argc, char* argv[]){
    int np = -1, nt = -1, nx = -1;

    // Define long options
    static struct option long_options[] = {
            {"np", required_argument, 0, 'p'},
            {"nt", required_argument, 0, 't'},
            {"nx", required_argument, 0, 'x'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "p:t:x:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p':
                np = std::atoi(optarg);
                break;
            case 't':
                nt = std::atoi(optarg);
                break;
            case 'x':
                nx = std::atoi(optarg);
                break;
            case 'h':
            default:
                std::cout << "Usage: " << argv[0]
                          << " -p <np> -t <nt> -x <nx>\n"
                          << "   or: " << argv[0]
                          << " --np <np> --nt <nt> --nx <nx>\n";
                return 0;
        }
    }

    // Check required values
    if (np < 0 || nt < 0 || nx < 0) {
        std::cerr << "Error: Missing required parameters.\n";
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    double* U[2];

    U[0] = (double *)malloc(np * nx * sizeof(double));
    U[1] = (double *)malloc(np * nx * sizeof(double));

    double* current = U[0];
    double* next = U[1];

    unsigned int n_threads = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 1);
    std::vector<long> sync(RP, 0.0);
    std::vector<long> exec(RP, 0.0);
    std::vector<long> total(RP, 0.0);

    for (int i = 0; i < RP; i++) {
        std::cout << "ROI: " << i << std::endl;

        std::vector<std::chrono::steady_clock::time_point> task_end(n_threads);
        std::vector<long> task_duration(n_threads, 0.0);
        std::vector<long> barrier(n_threads, 0.0);
        std::chrono::steady_clock::time_point taskwait;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        define_uniform(current, 0, np*nx);
        std::fill(next, next + np * nx, 0);

#pragma omp parallel
        {
#pragma omp single
            {
                stencil(nt, np, nx, current, next, U, task_end.data(), &taskwait, task_duration.data());
            }
            int tid = omp_get_thread_num();
            std::chrono::steady_clock::time_point barrier_begin = std::chrono::steady_clock::now();
#pragma omp barrier
            std::chrono::steady_clock::time_point barrier_end = std::chrono::steady_clock::now();
            barrier[tid] = std::chrono::duration_cast<std::chrono::nanoseconds>(barrier_end - barrier_begin).count();
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        //for (int i = 0; i < np*nx; ++i) {
        //    std::cout << U[nt % 2][i] << " ";
        //}
        std::cout << "Time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
        << std::endl;
        total[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        auto task_finish = *std::max_element(task_end.begin(), task_end.end());
        /*for(int i = 0; i < n_threads; i++)
        {
        std::cout << "Thread " << i << " Task exec time: " << std::setprecision (15) << task_duration[i] << std::endl;
        }*/
        std::cout << "Task exec time: " << std::setprecision (15) <<
        std::reduce(task_duration.begin(), task_duration.end()) << std::endl;
        exec[i] = std::reduce(task_duration.begin(), task_duration.end());

        std::cout << "Task wait time: " << std::setprecision (15) <<
        std::chrono::duration_cast<std::chrono::nanoseconds>(taskwait-task_finish).count() << std::endl;

        std::cout << "Barrier waiting time: " << std::setprecision (15) <<
        std::reduce(barrier.begin(), barrier.end()) << std::endl;
        sync[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(taskwait-task_finish).count() +
                std::reduce(barrier.begin(), barrier.end());
    }

    std::cout << "overall: " << std::endl;
    std::cout << "median of synchronization " << computeMedian(sync) << std::endl;
    std::cout << "median of task exec " << computeMedian(exec) << std::endl;
    std::cout << "median of total time " << computeMedian(total) << std::endl;
}