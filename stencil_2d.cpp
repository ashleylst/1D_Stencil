//
// Created by shiting on 2025-11-26.
//
//
// Created by shiting on 2025-04-09.
//
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include "chrono"
#include <getopt.h>
#include <cstdlib>

#define RP 1

void vector_double_define_random(double* phi, int start, int end) {
    std::random_device seed;
    std::mt19937 randomInt(seed());
    std::uniform_real_distribution<> unit(1.0, 10.0);

    for (int i = start; i < end; i++)
        phi[i] = unit(randomInt);

}

//idx does the wrapping here
static inline int idx(int row, int col, int N)
{
    return row * N + col;
}


void print_matrix(int N, double *m){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << m[idx(i, j, N)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
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

static inline double calculate(double up, double down, double left, double right)
{
    return 0.25 * (up + down + left + right);
}



static inline void jacobi_kernel(int N, int n, bool end, double* next,
                                 double *up,
                                 double *mid, double *down)
{
    /// top boundary
    if (up != nullptr)
    {
        for (int x = 1; x < N - 1; x++) {
            next[idx(0, x, N)] = calculate(up[idx(n - 1, x, N)], mid[idx(1, x, N)],
                                           mid[idx(0, x - 1, N)], mid[idx(0, x + 1, N)]);
        }
    }
    /// mid
    int y_end = (end && N % n != 0) ? N % n - 1 : n - 1;
    for (int y = 1; y < y_end; y++) {
        for (int x = 1; x < N - 1; x++) {
            next[idx(y, x, N)] = calculate(mid[idx(y-1, x, N)], mid[idx(y+1, x, N)],
                                           mid[idx(y, x-1, N)], mid[idx(y, x+1, N)]);
        }
    }
    /// bottom boundary
    if (down != nullptr)
    {
        for (int x = 1; x < N - 1; x++) {
            next[idx(n-1, x, N)] = calculate(mid[idx(n-2, x, N)], down[idx(0, x, N)],
                                             mid[idx(n-1, x-1, N)], mid[idx(n-1, x+1, N)]);
        }
    }
}

void stencil(int nt, int n, int N, double *current, double *next, double **U){
    int num_block = std::ceil((N + n - 1) / n);
    for (int t = 0; t < nt; t++) {
#pragma omp task depend(out:next[0]) depend(in:current[0], current[n*N])
        jacobi_kernel(N, n, false, &next[0], nullptr, &current[0], &current[n * N]);
        for (int j = 1; j < num_block - 1; j++) {
#pragma omp task depend(out:next[j * n * N]) depend(in:current[(j - 1) * n * N], current[j * n * N], current[(j + 1) * n * N])
            jacobi_kernel(N, n, false, &next[j * n * N],
                          &current[(j - 1) * n * N],
                          &current[j * n * N],
                          &current[(j + 1) * n * N]);
        }
        if (N % n != 1) {
#pragma omp task depend(out:next[(num_block-1) * n * N]) depend(in:current[(num_block-2) * n * N], current[(num_block-1) * n * N])
            jacobi_kernel(N, n, true, &next[(num_block - 1) * n * N], &current[(num_block - 2) * n * N],
                          &current[(num_block - 1) * n * N], nullptr);
        }
        current = U[(t + 1) % 2];
        next = U[t % 2];

        //print_matrix(N, current);
    }

}



int main(int argc, char* argv[]){
    int n = -1, nt = -1, N = -1;

    // Define long options
    static struct option long_options[] = {
            {"n", required_argument, 0, 'p'},
            {"nt", required_argument, 0, 't'},
            {"N", required_argument, 0, 'x'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "p:t:x:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p':
                n = std::atoi(optarg);
                break;
            case 't':
                nt = std::atoi(optarg);
                break;
            case 'x':
                N = std::atoi(optarg);
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
    if (n < 0 || nt < 0 || N < 0) {
        std::cerr << "Error: Missing required parameters.\n";
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    double* U[2];

    U[0] = (double *)malloc(N * N * sizeof(double));
    U[1] = (double *)malloc(N * N * sizeof(double));

    double* current = U[0];
    double* next = U[1];

    unsigned int n_threads = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 1);
    std::vector<long> total(RP, 0.0);

    for (int i = 0; i < RP; i++) {
        std::cout << "ROI: " << i << std::endl;

        std::vector<std::chrono::steady_clock::time_point> task_end(n_threads);

        define_uniform(current, 0, N * N);
        std::fill(next, next + N * N, 0);
        // Set boundaries to 1.0
        for (std::size_t i = 0; i < N; i++) {
            next[i] = current[i] = 1.0;                                // top boundary
            next[i * N] = current[i * N] = 1.0;                        // left boundary
            next[i * N + (N-1)] = current[i * N + (N-1)] = 1.0;        // right boundary
            next[(N-1) * N + i] = current[(N-1) * N + i] = 1.0;        // bottom boundary
        }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel
        {
#pragma omp single
            {
                stencil(nt, n, N, current, next, U);
            }
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
                  << std::endl;
        total[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

    }

    std::cout << "overall: " << std::endl;
    std::cout << "median of total time " << computeMedian(total) << std::endl;

    free(U[0]);
    free(U[1]);
}