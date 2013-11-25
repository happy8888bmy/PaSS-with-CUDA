/**
 * PaSS.cu
 * The main functions of PaSS
 *
 * @author emfo
 */

#include "PaSS_BLAS.cu"


// Test kernel function
__global__ void testKernel(){
	uint32_t p = 4, q = 6;
	mat* a = construct(p, q);
	for(uint32_t i = 0; i < a->n_col; i++){
		for(uint32_t j = 0; j < a->n_row; j++){
			a->col[i]->e[j] = i*j;
		}
	}

	vec* v = construct(p);
	for(uint32_t i = 0; i < v->n; i++){
		v->e[i] = i+1;
	}
	print(v);
	shed(v);
	print(v);
	insert(v, 10);
	print(v);
	
	print(a);
	insert_col(a, v);
	print(a);
	shed_col(a);
	print(a);
}


// Test host function
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

	// Check for any errors launching the kernel.
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


// Main function
int main() {
	test();
	system("pause");
	return 0;
}
