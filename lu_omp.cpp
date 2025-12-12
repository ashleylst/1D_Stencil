#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <iostream>
#include <getopt.h>
#include <chrono>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <cmath>

#define DIVCEIL(a, b) (((a) + (b) - 1) / (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

class PerfManager {

    // control and ack fifo from perf
    int ctl_fd = -1;
    int ack_fd = -1;

    // if perf is enabled
    bool enable = false;

    // commands and acks to/from perf
    static constexpr const char* enable_cmd = "enable";
    static constexpr const char* disable_cmd = "disable";
    static constexpr const char* ack_cmd = "ack\n";

    // send command to perf via fifo and confirm ack
    void send_command(const char* command) {
        if (enable) {
            write(ctl_fd, command, strlen(command));
            char ack[5];
            read(ack_fd, ack, 5);
            assert(strcmp(ack, ack_cmd) == 0);
        }
    }

public:

    PerfManager() {
        // setup fifo file descriptors
        char* ctl_fd_env = std::getenv("PERF_CTL_FD");
        char* ack_fd_env = std::getenv("PERF_ACK_FD");
        if (ctl_fd_env && ack_fd_env) {
            enable = true;
            ctl_fd = std::stoi(ctl_fd_env);
            ack_fd = std::stoi(ack_fd_env);
        }
    }

    // public apis

    void pause() {
        send_command(disable_cmd);
    }

    void resume() {
        send_command(enable_cmd);
    }
};

void print_matrix(int size, double *m){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << m[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void init(int size, double *A, double *B)
{
    auto L = (double *) malloc(size * size * sizeof(double));
    auto U = (double *) malloc(size * size * sizeof(double));
    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++){
            if(i >= j)
                L[i*size + j] = i-j+1;
            else
                L[i*size + j] = 0;
            if(i <= j)
                U[i*size + j] = j-i+1;
            else
                U[i*size + j] = 0;
        }
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++){
            for(int k = 0; k < size; k++) {
                A[i*size + j] += L[i*size + k] * U[k*size + j];
            }
        }
    }
    /// copy A to B for validation
    for (int i = 0; i < size * size; i++) {
        B[i] = A[i];
    }
    free(L);
    free(U);
}


void simple_lu(int n, int ldA, double *A)
{
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            A[i*ldA+j] /= A[i*ldA+i];

            for (int k = i+1; k < n; k++)
                A[k*ldA+j] -= A[i*ldA+j] * A[k*ldA+i];
        }
    }
}

    /// A: width * dsize, U: dsize * dsize, A -> A U^-1
void backward_solve(const double *U, double *A, int dsize, int width, int N)
{
    for (int j = 0; j < width; ++j) {
        for (int i = 0; i < dsize; ++i) {
            for (int k = 0; k < i; ++k) {
                A[j * N + i] -= U[i + k * N] * A[j * N + k];
            }
        }
    }
}

/// L: dsize * dsize, A: dsize * width, A -> L^-1 A
void forward_solve(const double *L, double *A, int dsize, int width, int N) {
    for (int j = 0; j < width; j++) {        // loop over columns
        for (int i = 0; i < dsize; i++) {    // loop over rows
            //double sum = 0.0;
            for (int k = 0; k < i; k++)
                A[i * N + j] -= L[i * N + k] * A[k * N + j];
            //A[i * N + j] = (A[i * N + j] - sum) / L[i * N + i];
        }
    }
}

/// A: height * dsize, B: dsize * width, C: height * width, C -> AxB
void matmul(const double *A, const double *B, double *C, int height, int dsize, int width, int N) {
    for (int i = 0; i < height; i++) {
        for (int k = 0; k < dsize; k++) {
            for (int j = 0; j < width; j++) {
                C[i * N + j] -= A[i * N + k] * B[k * N + j];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

void blocked_lu(int block_size, int N, double *A)
{
    int max_prio = omp_get_max_task_priority();
    // allocate and fill an array that stores the block pointers
    int block_count = DIVCEIL(N, block_size);
    double ***blocks = (double ***) malloc(block_count * sizeof(double**));
    for (int i = 0; i < block_count; i++) {
        blocks[i] = (double **) malloc(block_count * sizeof(double*));

        // note that blocks are accessed like block[col][row]
        for (int j = 0; j < block_count; j++)
            blocks[i][j] = A + (i * N + j) * block_size;
    }

    // iterate through the diagonal blocks
#pragma omp parallel
#pragma omp single nowait
    for (int i = 0; i < block_count; i++) {

        // calculate diagonal block size
        int dsize = MIN(block_size, N - i * block_size);

        // process the current diagonal block
#pragma omp task depend(inout:blocks[i][i]) priority(max_prio)
        simple_lu(dsize, N, blocks[i][i]);

        // process the blocks to the right of the current diagonal block
        for (int j = i+1; j < block_count; j++) {
            int width = MIN(block_size, N - j * block_size);
#pragma omp task depend(in:blocks[i][i]) depend(inout:blocks[i][j]) priority(MAX(0, max_prio-j))
            forward_solve(blocks[i][i], blocks[i][j], dsize, width, N);
        }

        // process the blocks below the current diagonal block
        for (int j = i+1; j < block_count; j++) {
            int height = MIN(block_size, N - j * block_size);
#pragma omp task depend(in:blocks[i][i]) depend(inout:blocks[j][i]) priority(MAX(0, max_prio-j))
            backward_solve(blocks[i][i], blocks[j][i], dsize, height, N);
        }

        // process the trailing matrix
        for (int ii = i+1; ii < block_count; ii++) {
            for (int jj = i+1; jj < block_count; jj++) {
                int height = MIN(block_size, N - jj * block_size);
                int width = MIN(block_size, N - ii * block_size);

                // blocks[ii][jj] <-
                //               blocks[ii][jj] -  blocks[ii][i] * blocks[i][jj]
#pragma omp task depend(in:blocks[ii][i],blocks[i][jj]) depend(inout:blocks[ii][jj]) \
            priority(MAX(0, max_prio - (ii + jj)))
                matmul(blocks[ii][i], blocks[i][jj], blocks[ii][jj], height, dsize, width, N);
            }
        }
    }

    // free allocated resources
    for (int i = 0; i < block_count; i++)
        free(blocks[i]);
    free(blocks);
}



////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////


void validate(int size, double const *A, double *B)
{
    int errors = 0;
    double temp2;
    auto L = (double *) malloc(size * size * sizeof(double));
    auto U = (double *) malloc(size * size * sizeof(double));
    std::fill(L, L + size * size, 0);
    std::fill(U, U + size * size, 0);

    for(int i = 0;i < size; i++)
        for(int j = 0;j < size; j++)
            if (i > j){
                L[i * size + j] = A[i * size + j];
            }
            else{
                U[i * size + j] = A[i * size + j];
            }
    ///set diagonals to 1
    for(int i = 0; i < size; i++){
        L[i * size + i] = 1;
    }

    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++) {
            temp2 = 0;
            for(int k = 0; k < size; k++){
                temp2 += L[i * size + k] * U[k * size + j];
            }
            if((B[i * size + j] - temp2) / B[i * size + j] > 0.1 || (B[i * size + j] - temp2) / B[i * size + j] < -0.1){
                printf("error:[%d][%d] ", i, j);
                errors++;
            }
        }

    free(L);
    free(U);
}

int main(int argc, char **argv)
{
    //
    // check arguments
    //
    //PerfManager pmon;
    //pmon.pause();

    int n = -1, N = -1;

    // Define long options
    static struct option long_options[] = {
            {"n", required_argument, 0, 'p'},
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
    if (n <= 0 || N <= 0) {
        std::cerr << "Error: Missing required parameters.\n";
        std::cerr << "Use --help for usage.\n";
        return 1;
    }


    auto *A = (double *) malloc(N * N * sizeof(double));
    auto *B = (double *) malloc(N * N * sizeof(double));

    // A
    init(N, A, B);

    //print_matrix(N, A);


    //pmon.resume();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // A <- (L,U)
    blocked_lu(n, N, A);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //pmon.pause();

    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
              << std::endl;


   // print_matrix(N, A);

    validate(N, A, B);


    free(A);
    free(B);

    return 0;
}