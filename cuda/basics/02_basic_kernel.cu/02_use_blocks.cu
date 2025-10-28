/** Explanation blocks and threads
 * This CUDA program demonstrates a kernel that efficiently uses both blocks and threads to add two vectors.
 * Unlike the thread-only version, this can handle arrays of any size (not limited to 1024 elements).
 * Each thread computes its global index using: blockIdx.x * blockDim.x + threadIdx.x
 * This allows parallel processing across multiple blocks, each containing multiple threads.
 * The grid is automatically calculated based on the array size and block size.
 * We cannot have more than 1024 threads per block, so we define a constant THREADS_PER_BLOCK.
*/

#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <time.h>

#define THREADS_PER_BLOCK 256

__global__ void setupRandomStates(curandState *state, unsigned long seed, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        curand_init(seed, index, 0, &state[index]);
    }
}

__global__ void generateVector(int *vec, curandState *state, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        vec[index] = curand(&state[index]) % 100; // Random value between 0 and 99
    }
}

__global__ void sumVectors(int *a, int *b, int *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int arraySize;

    // Read vector size from user
    printf("Enter the vector size: ");
    scanf("%d", &arraySize);

    if (arraySize <= 0) {
        printf("Vector size must be greater than 0\n");
        return 1;
    }

    // Calculate grid dimensions
    int blocksPerGrid = (arraySize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("\n=== Configuration ===\n");
    printf("Vector size: %d elements\n", arraySize);
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Blocks per grid: %d\n", blocksPerGrid);
    printf("Total threads launched: %d\n", blocksPerGrid * THREADS_PER_BLOCK);

    // Allocate host memory for result
    int *c = (int*)malloc(arraySize * sizeof(int));

    int *d_a, *d_b, *d_res;
    curandState *d_state;
    size_t size = arraySize * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_res, size);
    cudaMalloc((void**)&d_state, arraySize * sizeof(curandState));

    // Setup random states
    setupRandomStates<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_state, time(NULL), arraySize);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start_gen, stop_gen, start_sum, stop_sum;
    cudaEventCreate(&start_gen);
    cudaEventCreate(&stop_gen);
    cudaEventCreate(&start_sum);
    cudaEventCreate(&stop_sum);

    // Time vector generation
    printf("\nGenerating random vectors...\n");
    cudaEventRecord(start_gen);

    generateVector<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_state, arraySize);
    generateVector<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_b, d_state, arraySize);

    cudaEventRecord(stop_gen);
    cudaEventSynchronize(stop_gen);

    // Calculate generation time
    float genTime;
    cudaEventElapsedTime(&genTime, start_gen, stop_gen);

    // Time vector summation
    printf("Computing vector sum...\n");
    cudaEventRecord(start_sum);

    sumVectors<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_res, arraySize);

    cudaEventRecord(stop_sum);
    cudaEventSynchronize(stop_sum);

    // Calculate summation time
    float sumTime;
    cudaEventElapsedTime(&sumTime, start_sum, stop_sum);

    // Copy result array from device to host
    cudaMemcpy(c, d_res, size, cudaMemcpyDeviceToHost);

    // Print timing results
    printf("\n=== Timing Results ===\n");
    printf("Vector generation time: %.6f ms\n", genTime);
    printf("Vector summation time: %.6f ms\n", sumTime);
    printf("Total computation time: %.6f ms\n", genTime + sumTime);

    // Clean up
    cudaEventDestroy(start_gen);
    cudaEventDestroy(stop_gen);
    cudaEventDestroy(start_sum);
    cudaEventDestroy(stop_sum);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    cudaFree(d_state);

    // Free host memory
    free(c);

    return 0;
}
