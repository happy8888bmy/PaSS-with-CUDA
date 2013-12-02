/**
 * PaSS.cu
 * The main functions of PaSS
 *
 * @author emfo
 */

#include "PaSS_BLAS.cu"

#define PASS_N 10
#define PASS_P 20

void pass_init(float*, float*);
cudaError_t pass_host(const float*, const float*, float*);
__global__ void pass_kernel(const float*, const float*, float*);


// Main function
int main() {
    // Declare varibles
    float *X = (float*)malloc(PASS_N * PASS_P * sizeof(float));
    float *Y = (float*)malloc(PASS_N * sizeof(float));
    float *I = (float*)malloc(PASS_N * sizeof(float));
    uint32_t i, j;
    
    // Call fucntions
    pass_init(X, Y);
    pass_host(X, Y, I);

    // Display data
    printf("X:\n");
    for(i = 0; i < PASS_N; i++) {
        for(j = 0; j < PASS_P; j++) {
            printf("%4.2f ", X[i*PASS_N + j]);
        }
        printf("\n");
    }
    printf("\n\nY:\n");
    for(i = 0; i < PASS_N; i++) {
        printf("%4.2f ", Y[i]);
    }
    printf("\n\nI:\n");
    for(i = 0; i < PASS_N; i++) {
        printf("%4.2f ", I[i]);
    }
    printf("\n\n");
    system("pause");
    return 0;
}

// Initialize data
void pass_init(float* X, float* Y) {
    uint32_t i, j;
    for(i = 0; i < PASS_N; i++) {
        for(j = 0; j < PASS_P; j++) {
            X[i*PASS_N + j] = 2;
        }
        Y[i] = 3;
    }
}


// Host function
cudaError_t pass_host(const float* X, const float* Y, float* I) {
    // Declare varibles
    float *dev_X = 0;
    float *dev_Y = 0;
    float *dev_I = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }
    
    // Allocate GPU buffers for data (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_X, PASS_N * PASS_P * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_Y, PASS_N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_I, PASS_N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_X, X, PASS_N * PASS_P * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_Y, Y, PASS_N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch the kernel function on the GPU with one thread for each element.
    pass_kernel<<<1, 1>>>(dev_X, dev_Y, dev_I);

    // Check for any errors launching the kernel.
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching testKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(I, dev_I, PASS_N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_X);
    cudaFree(dev_Y);
    cudaFree(dev_I);
    return cudaStatus;
}


// Test kernel function
__global__ void pass_kernel(const float* array_X, const float* array_Y, float* array_I) {
    // Declare varibles
    pass_mat *X = pass_new(PASS_N, PASS_P);
    pass_vec *Y = pass_new(PASS_N);
    pass_vec *I = pass_new(PASS_N);
    uint32_t i, j;
    
    // Copy X and Y from array to matrix
    for(j = 0; j < PASS_P; j++) {
        for(i = 0; i < PASS_N; i++) {
            X->col[j]->e[i] = array_X[i*PASS_N + j];
        }
    }
    for(uint32_t i = 0; i < PASS_N; i++) {
        Y->e[i] = array_Y[i];
    }

    // Let I = X'*Y
    I = pass_mul(Y, X);
    
    // Copy I from matrix to array
    for(uint32_t i = 0; i < PASS_N; i++) {
        array_I[i] = I->e[i];
    }
}
