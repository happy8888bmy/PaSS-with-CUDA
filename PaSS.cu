#include "PaSS_BLAS.cuh"

/**
 * construct a n-by-1 vector
 */
__device__ vec* construct(const uint32_t n){
    vec *v = (vec*)malloc(sizeof(vec));
    v->n = (uint32_t*)malloc(sizeof(uint32_t));
    *v->n = n;
    v->e = (double*)malloc(n*sizeof(double));
    return v;
}

/**
 * construct a p-by-q matrix
 */
__device__ mat* construct(const uint32_t p, const uint32_t q){
    mat *m = (mat*) malloc(sizeof(mat));
    m->n_row = p;
    m->n_col = q;
    m->col = (vec**)malloc(q*sizeof(vec*));
    for(vec **v = m->col; v < m->col+q; v++){
        *v = (vec*)malloc(sizeof(vec));
        (*v)->n = &m->n_row;
        (*v)->e = (double*)malloc(p*sizeof(double));
    }
    return m;
}

/**
 *display the vector
 */
__device__ void print(const vec* v){
    for(uint32_t i = 0; i < *v->n; i++){
        printf("%8.3f\n", v->e[i]);
    }
}

/**
 *display the matrix
 */
__device__ void print(const mat* m){
    for(uint32_t j = 0; j < m->n_row; j++){
        for(int i = 0; i < m->n_col; i++){
            printf("%8.3f", m->col[i]->e[j]);
        }
        printf("\n");
    }
}

__global__ void testKernel(){
    uint32_t n = 5, p = 3, q = 4;
    vec* v = construct(n);
    for(uint32_t i = 0; i < n; i++){
        v->e[i] = (i+1)*(i+1);
    }
    print(v);
    
    mat* m = construct(p, q);
    for(uint32_t j = 0; j < m->n_row; j++){
        for(uint32_t i = 0; i < m->n_col; i++){
            m->col[i]->e[j] = (i+1)*(j+1);
        }
    }
    print(m);
}

cudaError_t test()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    testKernel<<<1, 1>>>();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "testKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching testKernel!\n", cudaStatus);
        goto Error;
    }

Error:
    return cudaStatus;
}

int main() {
    test();
    system("pause");
    return 0;
}
