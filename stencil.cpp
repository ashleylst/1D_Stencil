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
                double *mid, double *right, double *task_end, double *task_duration)
{
    int tid = omp_get_thread_num();
    double start = omp_get_wtime();

    next[0] = heat(left[size-1], mid[0], mid[1]);
    for (int i = 1; i < size-1; ++i)
        next[i] = heat(mid[i-1], mid[i], mid[i+1]);
    next[size-1] = heat(mid[size-2], mid[size-1],
                        right[0]);

    task_end[tid] = omp_get_wtime();
    task_duration[tid] += task_end[tid] - start;
}

//idx does the wrapping here
int idx(int i, int size)
{
    return (i < 0) ? (i + size) % size : i % size;
}

void stencil(int nt, int np, int nx, double *current, double *next, double **U,
             double *task_end, double *taskwait, double *task_duration){
    for (int t = 0; t < nt; t++) {
        for (int i = 0; i < np; ++i) {
#pragma omp task untied depend(out: next[i*nx]) \
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
    *taskwait = omp_get_wtime();
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
    std::vector<double> task_end(n_threads, 0.0);
    std::vector<double> task_duration(n_threads, 0.0);
    double taskwait = 0.0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel
    {
#pragma omp single
        {
            stencil(nt, np, nx, current, next, U, task_end.data(), &taskwait, task_duration.data());
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //for (int i = 0; i < np*nx; ++i) {
    //    std::cout << U[nt % 2][i] << " ";
    //}
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    double task_finish = *std::max_element(task_end.begin(), task_end.end());
    std::cout << "Task exec time: " << std::setprecision (15) <<  *std::max_element(task_duration.begin(), task_duration.end()) << std::endl;

    std::cout << "Task wait time: " << std::setprecision (15) << taskwait - task_finish << std::endl;
}