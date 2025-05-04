#include "matmul.cuh"
#include <vector>


__global__ void do_matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

ProfileResult perf_gemm(int N)
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate memory on host
    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];

    // Initialize matrices h_A and h_B

    // Allocate memory on device
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch&Profile kernel
    cudaEvent_t start, stop;
    std::vector<float> durations;
    for (int i = 0; i < 10; i++)
    {
        float duration = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        do_matmul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration, start, stop);
        durations.push_back(duration);
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Find min, max, and average
    double min = *std::min_element(durations.begin(), durations.end());
    double max = *std::max_element(durations.begin(), durations.end());
    double avg = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();

    // Find standard deviation
    double sum = 0.0;
    for (double duration : durations) {
        sum += (duration - avg) * (duration - avg);
    }
    double std_dev = sqrt(sum / durations.size());

    return ProfileResult(min, max, avg, std_dev);
}
