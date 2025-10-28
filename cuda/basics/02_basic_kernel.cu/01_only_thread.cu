/** Explanation threads 
 * This CUDA program demonstrates a basic kernel that adds two vectors using only thread indices.
 * Each thread computes the sum of corresponding elements from two input arrays and stores the result in a third array.
 * The kernel is launched with a single block containing multiple threads, where each thread processes one element of the arrays.
 * This version reads vector size from user input, generates random vectors on GPU, and measures execution times.
 * Note: This implementation is limited to a maximum of 1024 elements due to the single block configuration.
*/

#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void setupRandomStates(curandState *state, unsigned long seed, int n) {
    int index = threadIdx.x;
    if (index < n) {
        curand_init(seed, index, 0, &state[index]);
    }
}

__global__ void generateVector(int *vec, curandState *state, int n) {
    int index = threadIdx.x;
    if (index < n) {
        vec[index] = curand(&state[index]) % 100; // Random value between 0 and 99
    }
}

__global__ void sumVectors(int *a, int *b, int *c, int n) {
    int index = threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int arraySize;
    
    // Read vector size from user
    printf("Enter the vector size: ");
    scanf("%d", &arraySize);
    
    if (arraySize <= 0 || arraySize > 1024) {
        printf("Vector size must be between 1 and 1024 (CUDA block limit)\n");
        return 1;
    }
    
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
    setupRandomStates<<<1, arraySize>>>(d_state, time(NULL), arraySize);
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
    
    generateVector<<<1, arraySize>>>(d_a, d_state, arraySize);
    generateVector<<<1, arraySize>>>(d_b, d_state, arraySize);
    
    cudaEventRecord(stop_gen);
    cudaEventSynchronize(stop_gen);
    
    // Calculate generation time
    float genTime;
    cudaEventElapsedTime(&genTime, start_gen, stop_gen);
    
    // Time vector summation
    printf("Computing vector sum...\n");
    cudaEventRecord(start_sum);
    
    sumVectors<<<1, arraySize>>>(d_a, d_b, d_res, arraySize);
    
    cudaEventRecord(stop_sum);
    cudaEventSynchronize(stop_sum);
    
    // Calculate summation time
    float sumTime;
    cudaEventElapsedTime(&sumTime, start_sum, stop_sum);
    
    // Copy result array from device to host
    cudaMemcpy(c, d_res, size, cudaMemcpyDeviceToHost);
    
    // Print timing results
    printf("\n=== Timing Results ===\n");
    printf("Vector size: %d elements\n", arraySize);
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