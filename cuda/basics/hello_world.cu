#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel that runs on the GPU
// Each thread will print a message with its unique ID
__global__ void helloFromGPU(int numThreads) {
    // Calculate unique thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numThreads) {
        printf("=� Hello from GPU! Thread %d in Block %d (Global ID: %d)\n",
               threadIdx.x, blockIdx.x, tid);
    }
}

// CUDA kernel that performs parallel addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    printf("\n========================================\n");
    printf("  CUDA Hello World & Vector Addition\n");
    printf("========================================\n\n");

    // Get GPU device properties
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("L No CUDA-capable GPU found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=� GPU Information:\n");
    printf("   Device: %s\n", prop.name);
    printf("   Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("   Total Global Memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("   Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("   Max Threads per Block: %d\n\n", prop.maxThreadsPerBlock);

    // Part 1: Simple Hello World from GPU
    printf("========================================\n");
    printf("Part 1: Hello from GPU Threads\n");
    printf("========================================\n\n");

    int numBlocks = 3;
    int threadsPerBlock = 8;
    int totalThreads = numBlocks * threadsPerBlock;

    printf("Launching %d blocks with %d threads each (%d total threads)\n\n",
           numBlocks, threadsPerBlock, totalThreads);

    // Launch kernel
    helloFromGPU<<<numBlocks, threadsPerBlock>>>(totalThreads);

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Part 2: Parallel Vector Addition
    printf("\n========================================\n");
    printf("Part 2: Parallel Vector Addition\n");
    printf("========================================\n\n");

    const int N = 16;
    int size = N * sizeof(int);

    // Allocate host memory
    int h_a[N], h_b[N], h_c[N];

    // Initialize input vectors
    printf("Input vectors:\n");
    printf("A: ");
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        printf("%d ", h_a[i]);
    }
    printf("\nB: ");
    for (int i = 0; i < N; i++) {
        h_b[i] = i * 2;
        printf("%d ", h_b[i]);
    }
    printf("\n\n");

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch vector addition kernel
    int addBlocks = (N + 255) / 256;
    int addThreads = 256;
    printf("Computing C = A + B on GPU...\n");
    printf("Using %d block(s) with %d threads\n\n", addBlocks, addThreads);

    vectorAdd<<<addBlocks, addThreads>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Display results
    printf("Result C = A + B:\n");
    printf("C: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_c[i]);
    }
    printf("\n\n");

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        printf(" Vector addition successful! All values correct.\n");
    } else {
        printf("L Vector addition failed! Some values incorrect.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    printf("\n========================================\n");
    printf("  Program completed successfully!\n");
    printf("========================================\n\n");

    return 0;
}
