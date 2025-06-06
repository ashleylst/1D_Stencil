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

void define_uniform(double* phi, int start, int end){
    for (int i = start; i < end; ++i) {
        phi[i] = i;
    }
}

double heat(double left, double mid, double right)
{
    return mid+(k*dt/dx*dx)*(left-2*mid+right);
}


void heat_part( int size, double* next,
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
int idx(int i, int size)
{
    return (i < 0) ? (i + size) % size : i % size;
}

void stencil(int nt, int np, int nx, double *current, double *next, double **U,
             std::chrono::steady_clock::time_point *task_end, std::chrono::steady_clock::time_point *taskwait,
             long *taskcreate, long *task_duration){
    for (int t = 0; t < nt; t++) {
        for (int i = 0; i < np; ++i) {
            auto start = std::chrono::steady_clock::now();
#pragma omp task depend(out: next[i*nx]) \
depend(in: current[idx(i-1, np)*nx], \
current[i*nx], current[idx(i+1, np)*nx])
            heat_part(nx, &next[i * nx],
                      &current[idx(i - 1, np) * nx],
                      &current[i * nx],
                      &current[idx(i + 1, np) * nx],
                      task_end, task_duration);
            auto end = std::chrono::steady_clock::now();
            *taskcreate += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        current = U[(t + 1) % 2];
        next = U[t % 2];
    }
#pragma omp taskwait
    *taskwait = std::chrono::steady_clock::now();
}


int main(){
    int np = 1000;
    int nx = 10000;
    int nt = 100;
    double* U[2];

    U[0] = (double *)malloc(np * nx * sizeof(double));
    U[1] = (double *)malloc(np * nx * sizeof(double));

    double* current = U[0];
    double* next = U[1];

    define_uniform(current, 0, np*nx);
    std::fill(next, next + np * nx, 0);

    unsigned int n_threads = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 1);
    std::vector<std::chrono::steady_clock::time_point> task_end(n_threads);
    std::vector<long> task_duration(n_threads, 0.0);
    std::vector<long> barrier(n_threads, 0.0);
    std::chrono::steady_clock::time_point taskwait;
    long taskcreate = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel
    {
#pragma omp single
        {
            stencil(nt, np, nx, current, next, U, task_end.data(), &taskwait, &taskcreate, task_duration.data());
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
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

    auto task_finish = *std::max_element(task_end.begin(), task_end.end());
    std::cout << "Task exec time: " << std::setprecision (15) <<  *std::max_element(task_duration.begin(), task_duration.end()) << std::endl;

    std::cout << "Task wait time: " << std::setprecision (15) << std::chrono::duration_cast<std::chrono::nanoseconds>(taskwait-task_finish).count() << std::endl;

    std::cout << "Task create time: " << std::setprecision (15) << taskcreate << std::endl;

    std::cout << "Barrier waiting time: " << std::setprecision (15) << *std::max_element(barrier.begin(), barrier.end()) << std::endl;
}