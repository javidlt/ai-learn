/** Explanation 2D grids and blocks
 * This CUDA program demonstrates using 2D grids and 2D blocks for matrix operations.
 * Unlike 1D arrays, matrices benefit from 2D organization where:
 * - Each thread computes one matrix element using 2D coordinates
 * - blockIdx.x, blockIdx.y identify the block position in the grid
 * - threadIdx.x, threadIdx.y identify the thread position within the block
 * - Global position: row = blockIdx.y * blockDim.y + threadIdx.y
 *                   col = blockIdx.x * blockDim.x + threadIdx.x
 * This approach is more natural for 2D data and improves memory access patterns.
*/

#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16  // 16x16 = 256 threads per block

__global__ void setupRandomStates(curandState *state, unsigned long seed, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * cols + col;

    if (row < rows && col < cols) {
        curand_init(seed, index, 0, &state[index]);
    }
}

__global__ void generateMatrix(int *matrix, curandState *state, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * cols + col;

    if (row < rows && col < cols) {
        matrix[index] = curand(&state[index]) % 100; // Random value between 0 and 99
    }
}

__global__ void sumMatrices(int *a, int *b, int *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * cols + col;

    if (row < rows && col < cols) {
        c[index] = a[index] + b[index];
    }
}

void printMatrix(int *matrix, int rows, int cols, const char *name) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    int displayRows = rows < 5 ? rows : 5;
    int displayCols = cols < 10 ? cols : 10;

    for (int i = 0; i < displayRows; i++) {
        for (int j = 0; j < displayCols; j++) {
            printf("%4d ", matrix[i * cols + j]);
        }
        if (cols > displayCols) printf("...");
        printf("\n");
    }
    if (rows > displayRows) printf("...\n");
}

int main() {
    int rows, cols;

    // Read matrix dimensions from user
    printf("Enter number of rows: ");
    scanf("%d", &rows);
    printf("Enter number of columns: ");
    scanf("%d", &cols);

    if (rows <= 0 || cols <= 0) {
        printf("Rows and columns must be greater than 0\n");
        return 1;
    }

    // Calculate 2D grid dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 16x16 threads per block
    dim3 blocksPerGrid(
        (cols + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Grid width (X dimension)
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE   // Grid height (Y dimension)
    );

    int totalElements = rows * cols;

    printf("\n=== Configuration ===\n");
    printf("Matrix size: %d x %d = %d elements\n", rows, cols, totalElements);
    printf("Threads per block: %dx%d = %d threads\n",
           BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * BLOCK_SIZE);
    printf("Blocks per grid: %dx%d = %d blocks\n",
           blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.x * blocksPerGrid.y);
    printf("Total threads launched: %d\n",
           blocksPerGrid.x * blocksPerGrid.y * BLOCK_SIZE * BLOCK_SIZE);

    // Allocate host memory
    size_t size = totalElements * sizeof(int);
    int *c = (int*)malloc(size);

    // Allocate device memory
    int *d_a, *d_b, *d_res;
    curandState *d_state;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_res, size);
    cudaMalloc((void**)&d_state, totalElements * sizeof(curandState));

    // Setup random states
    setupRandomStates<<<blocksPerGrid, threadsPerBlock>>>(d_state, time(NULL), rows, cols);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start_gen, stop_gen, start_sum, stop_sum;
    cudaEventCreate(&start_gen);
    cudaEventCreate(&stop_gen);
    cudaEventCreate(&start_sum);
    cudaEventCreate(&stop_sum);

    // Time matrix generation
    printf("\nGenerating random matrices...\n");
    cudaEventRecord(start_gen);

    generateMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_state, rows, cols);
    generateMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_state, rows, cols);

    cudaEventRecord(stop_gen);
    cudaEventSynchronize(stop_gen);

    // Calculate generation time
    float genTime;
    cudaEventElapsedTime(&genTime, start_gen, stop_gen);

    // Time matrix summation
    printf("Computing matrix sum...\n");
    cudaEventRecord(start_sum);

    sumMatrices<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_res, rows, cols);

    cudaEventRecord(stop_sum);
    cudaEventSynchronize(stop_sum);

    // Calculate summation time
    float sumTime;
    cudaEventElapsedTime(&sumTime, start_sum, stop_sum);

    // Copy result back to host
    cudaMemcpy(c, d_res, size, cudaMemcpyDeviceToHost);

    // Print timing results
    printf("\n=== Timing Results ===\n");
    printf("Matrix generation time: %.6f ms\n", genTime);
    printf("Matrix summation time: %.6f ms\n", sumTime);
    printf("Total computation time: %.6f ms\n", genTime + sumTime);

    // Print sample of result matrix
    printMatrix(c, rows, cols, "Result Matrix (sample)");

    // Clean up
    cudaEventDestroy(start_gen);
    cudaEventDestroy(stop_gen);
    cudaEventDestroy(start_sum);
    cudaEventDestroy(stop_sum);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    cudaFree(d_state);

    free(c);

    return 0;
}
