/**
 * PaSS.cu <br>
 * The main functions of PaSS
 *
 * \author emfo
 * \date 2014.01.13 03:44
 */

#include "PaSS_BLAS.cu"
using namespace pass_blas;

#include <ctime>
#include <random>
using namespace std;

#define PASS_MAX_N 128
#define PASS_MAX_P 32

/**
 * The PaSS namespace
 */
namespace pass{
	/**
	 * The status enumeration
	*/
	enum Stat {
		init = 0,  /**< initial step */
		forw = 1,  /**< forward step */
		back = -1, /**< backward step */
	};

	/**
	 * The criterion enumeration
	 */
	enum Criterion {
		AIC,   /**< Akaike information criterion */
		BIC,   /**< Bayesian information criterion */
		EBIC,  /**< EBIC */
		HDBIC, /**< HDBIC */
		HDHQ   /**< HDHQ */
	};

	/**
	 * The parameter structure
	 */
	struct Parameter {
		u32 nP;    /**< the number of particle */
		u32 nI;    /**< the number of iteration */
		float pfg; /**< the probability for forward step: global */
		float pfl; /**< the probability for forward step: local */
		float pfr; /**< the probability for forward step: random */
		float pbl; /**< the probability for backward step: local */
		float pbr; /**< the probability for backward step: random */
	};

	/**
	 * The data structure
	 */
	struct Data {
		vec Beta[PASS_MAX_P+2];                 /**< the vector beta */
		float e;                                /**< the norm of R */
		uvec Index[PASS_MAX_P+2];               /**< the index of chosen column of X */
		mat InvA[(PASS_MAX_P+2)*PASS_MAX_P+2];  /**< the inverse of A */
		float phi;                              /**< the value given by criterion */
		vec R[PASS_MAX_N+2];                    /**< the difference between Y and Beta */
		Stat stat;                              /**< the status */
		vec Theta[PASS_MAX_P+2];                /**< the vector theta */
		mat X[(PASS_MAX_N+2)*PASS_MAX_P+2];     /**< the data we chosen */
	};
}
using namespace pass;


// Global variables
__device__ u32 n, p, id_best;
__device__ mat X;
__device__ vec Y;
__device__ Criterion cri;
__device__ Parameter par;
__device__ uvec Index_best;
__device__ float* phi_all;
__device__ curandState s;
__device__ vec CC;
__device__ uvec II;

// Functions
void pass_init(mat*, vec*);
void pass_host(const mat*, const vec*, uvec*, float*, const Criterion, const Parameter);
__global__ void pass_kernel(const mat*, const vec*, uvec*, float*, const Criterion, const Parameter);
__device__ bool pass_update_fb(Data*);
__device__ bool pass_update_cri(Data*, const u32);


/**
 * PaSS main function
 */
int main() {
	// Declare variables
	u32 host_n = 8;
	u32 host_p = 256;
	Criterion host_cri = HDBIC;
	Parameter host_par = {32, 128, .8f, .1f, .1f, .9f, .1f};
	mat* host_X = new mat[(PASS_MAX_N+2)*host_p+2];
	init_mat(host_X, host_n, host_p, PASS_MAX_N, host_p);
	vec* host_Y = new vec[PASS_MAX_N+2];
	init_vec(host_Y, host_n, PASS_MAX_N);
	uvec* host_I = new uvec[PASS_MAX_P+2];
	init_uvec(host_I, 0, PASS_MAX_P);
	float host_phi;

	if(host_p < 10) {
		printf("p must greater than 10!\n");
		system("pause");
		return 1;
	}
	
	// Initialize data
	pass_init(host_X, host_Y);
		
	// Run PaSS
	//pass_host(host_X, host_Y, host_I, &host_phi, host_cri, host_par);
	
	// Display data
	print_mat(host_X);
	print_vec(host_Y);
	print_uvec(host_I);
	system("pause");
	delete[] host_X;
	delete[] host_Y;
	delete[] host_I;
	return 0;
}


/**
 * Initialize data
 *
 * @param X the matrix X
 * @param Y the vector Y
 */
__host__ void pass_init(mat* X, vec* Y) {
	u32 i, j, n = n_rows(X), p = n_cols(X), q = 10;
	vec* Beta = new vec[p+4];
	init_vec(Beta, p, p);
	vec* X_hat = new vec[n+4];
	init_vec(X_hat, n, n);
	vec* Error = new vec[n+4];
	init_vec(Error, n, n);
	vec* Temp = new vec[p+4];
	init_vec(Temp, p, p);
	float temp;

	// Generate X and Error using normal random
	default_random_engine generator((u32)time(NULL));
	normal_distribution<float> distribution;
	for(j = 0; j < n_cols(X); j++) {
		for(i = 0; i < n_rows(X); i++) {
			entry2(X, i, j) = distribution(generator);
		}
	}
	for(i = 0; i < n_cols(Error); i++) {
		entry(Error, i) = distribution(generator);
	}
	
	// Compute X and Y
	entry(Beta, 0) = 3.0f;
	entry(Beta, 1) = 3.75f;
	entry(Beta, 2) = 4.5f;
	entry(Beta, 3) = 5.25f;
	entry(Beta, 4) = 6.0f;
	entry(Beta, 5) = 6.75f;
	entry(Beta, 6) = 7.5f;
	entry(Beta, 7) = 8.25f;
	entry(Beta, 8) = 9.0f;
	entry(Beta, 9) = 9.75f;
	for(j = q; j < p; j++) {
		entry(Beta, j) = 0;
	}
	for(j = 0; j < q; j++) {
		entry(X_hat, j) = 0;
		for(i = 0; i < n; i++) {
			entry(X_hat, j) += entry2(X, i, j);
		}
	}
	mul_vec(X_hat, X_hat, 1 / sqrt(2 * (float)q));
	for(j = q; j < p; j++) {
		mul_vec(col(X, j), col(X, j), .5f);
		add_vec(col(X, j), col(X, j), X_hat);
	}
	mul_matvec(Y, X, Beta);
	add_vec(Y, Y, Error);
	
	// Normalize X
	for(j = 0; j < p; j++) {
		norm_vec(&temp, col(X, j));
		mul_vec(col(X, j), col(X, j), 1/temp);
	}

	// Normalize Y
	norm_vec(&temp, Y);
	mul_vec(Y, Y, 1/temp);

	delete[] Beta;
	delete[] X_hat;
	delete[] Error;
	delete[] Temp;

}


///**
// * PaSS host function
// *
// * @param host_X the matrix X
// * @param host_Y the vector Y
// * @param host_I the index I
// * @param host_k length of I
// * @param host_phi the value phi
// * @param host_n the number of rows in X
// * @param host_p the number of columns in X
// * @param host_cri the criterion
// * @param host_par the parameter value
// */
//__host__ void pass_host(const mat* host_X, const vec* host_Y, uvec* host_I, float* host_phi, const Criterion host_cri, const Parameter host_par) {
//	// Declare variables
//	mat* dev_X = 0;
//	vec* dev_Y = 0;
//	uvec* dev_I = 0;
//	float* dev_phi = 0;
//	cudaError_t cudaStatus;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!	Do you have a CUDA-capable GPU installed?\n");
//		goto Error;
//	}
//	
//	// Allocate GPU buffers for data (two input, one output).
//	cudaStatus = cudaMalloc((void**)&dev_X, size_mat(host_X) * sizeof(mat));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc (X) failed!\n");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_Y, size_vec(host_Y) * sizeof(vec));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc (Y) failed!\n");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_I, size_vec(host_I) * sizeof(uvec));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc (I) failed!\n");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_phi, sizeof(float));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc (phi) failed!\n");
//		goto Error;
//	}
//
//	// Copy input from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_X, host_X, size_mat(host_X) * sizeof(mat), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy (X) failed!\n");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_Y, host_Y, size_vec(host_Y) * sizeof(vec), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy (Y) failed!\n");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_I, host_I, size_vec(host_I) * sizeof(uvec), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy (I) failed!\n");
//		goto Error;
//	}
//	
//	// Launch the kernel function on the GPU with one thread for each element.
//	pass_kernel<<<1, host_par.nP>>>(dev_X, dev_Y, dev_I, dev_phi, host_n, host_p, host_cri, host_par);
//
//	// Check for any errors launching the kernel.
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//	
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(host_I, dev_I, size_vec(host_I) * sizeof(uvec), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy (I) failed!\n");
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(host_phi, dev_phi, sizeof(float), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy (phi) failed!\n");
//		goto Error;
//	}
//
//Error:
//	cudaFree(dev_X);
//	cudaFree(dev_Y);
//	cudaFree(dev_I);
//	cudaFree(dev_phi);
//}


///**
// * PaSS kernel function
// *
// * @param host_X the matrix X
// * @param host_Y the vector Y
// * @param host_I the index I
// * @param host_k length of I
// * @param host_phi the value phi
// * @param host_n the number of rows in X
// * @param host_p the number of columns in X
// * @param host_cri the criterion
// * @param host_par the parameter value
// */
//__global__ void pass_kernel(const mat* host_X, const vec* host_Y, uvec* host_I, float* const host_phi, const Criterion host_cri, const Parameter host_par) {
//	// Declare variables
//	u32 id = threadIdx.x;
//	u32 i, j;
//	Data* data = new Data(host_n, host_p);
//	float phi_old;
//
//	u32 k_max = 0;
//	
//	if(id == 0) {
//		// Initialize Random Seed
//		curand_init(clock64(), 0, 0, &s);
//
//		// Declare variables
//		n = host_n;
//		p = host_p;
//		cri = host_cri;
//		par = host_par;
//		X = new mat(n, p);
//		Y = new vec(n);
//		Index_best = new uvec(p);
//		phi_all = new float[par.nP];	
//	
//		// Copy X and Y from array to matrix
//		for(u32 i = 0; i < n; i++) {
//			for(j = 0; j < p; j++) {
//				X->col[j]->e[i] = host_X[i*p + j];
//			}
//			Y->e[i] = host_Y[i];
//		}
//	}
//	__syncthreads();
//
//	// Initialize Particles
//	if(p >= par.nP && id == 0) {
//		CC = new vec(p);
//		II = new uvec(p);
//		mul(CC, Y, X);
//		for(i = 0; i < CC->n; i++) {
//			CC->e[i] = abs(CC->e[i]);
//		}
//		sort_index_descend(II, CC);
//	}
//	__syncthreads();
//
//	data->stat = init;
//	if(p >= par.nP) {
//		pass_update_cri(data, II->e[id]);
//	} else {
//		pass_update_cri(data, curand(&s) % p);
//	}
//	if(p >= par.nP && id == 0) {
//		delete CC;
//		delete II;
//	}
//	phi_old = data->phi;
//	data->stat = forw;
//
//	// Choose Global Best
//	if(id == 0) {
//		*host_phi = data->phi;
//		put(Index_best, data->Index);
//	}
//	__syncthreads();
//	
//	
//	// Find Best Data
//	for(i = 0; i < par.nI; i++) {
//		// Update data
//		pass_update_fb(data);
//		if(data->phi - phi_old > 0) {
//			data->stat = (Stat)(-data->stat);
//		}
//		if(data->Index->n <= 1) {
//			data->stat = forw;
//		}
//		if(data->Index->n >= p - 5) {
//			data->stat = back;
//		}
//		phi_old = data->phi;
//
//		// Choose Global Best
//		phi_all[id] = data->phi;
//		__syncthreads();
//		if(id == 0) {
//			id_best = (u32)(-1);
//			for(j = 0; j < par.nP; j++) {
//				if(phi_all[j] < *host_phi) {
//					id_best = j;
//					*host_phi = data->phi;
//				}
//			}
//		}
//		__syncthreads();
//		if(id == id_best) {
//			put(Index_best, data->Index);
//		}
//		__syncthreads();
//
//		if(data->Index->n > k_max)
//			k_max = data->Index->n;
//	}
//	
//	// Delete variables
//	delete data;
//
//	if(id == 0) {
//		// Copy Index_best from index to array
//		sort_ascend(Index_best);
//		for(j = 0; j < Index_best->n; j++) {
//			host_I[j] = Index_best->e[j];
//		}
//		*host_k = Index_best->n;
//
//		//printf("phi = %f\nk = %d\nIndex = ", *host_phi, *host_k);
//		//print(Index_best);
//
//		// Delete variables
//		delete X;
//		delete Y;
//		delete Index_best;
//		delete phi_all;
//	}
//}
//
//
///**
// * Determine forward or backward
// */
//__device__ bool pass_update_fb(Data* data) {
//	u32 index = 0;
//	switch(data->stat) {
//	case forw: // Forward
//		{
//			uvec* Index_B = new uvec(Index_best->n);
//			uvec* Index_C = new uvec(data->Index->n);
//			uvec* Index_D = new uvec(Index_B->n);
//			uvec* Index_R = new uvec(p - Index_C->n);
//
//			// Sort Index_B
//			copy(Index_B, Index_best);
//			sort_ascend(Index_B);
//			
//			// Sort Index_C
//			copy(Index_C, data->Index);
//			sort_ascend(Index_C);
//			
//			// Let Index_D be Index_B exclude Index_C
//			set_difference(Index_D, Index_B, Index_C);
//			
//			// Let Index_R be the complement of Index_C
//			complement(Index_R, Index_C, p);
//
//			// Determine the index to add
//			if(curand_uniform(&s) < par.pfg && Index_D->n > 0) {
//				index = Index_D->e[curand(&s) % Index_D->n];
//			}
//			else if(curand_uniform(&s) < par.pfl/(par.pfl+par.pfr)) {
//				float phi_max = -1, phi_temp;
//				for(u32 i = 0; i < Index_R->n; i++) {
//					inner(&phi_temp, data->R, X->col[Index_R->e[i]]);
//					phi_temp = abs(phi_temp);
//					if(phi_temp > phi_max) {
//						phi_max = phi_temp;
//						index = Index_R->e[i];
//					}
//				}
//			}
//			else{
//				index = Index_R->e[curand(&s) % Index_R->n];
//			}
//			delete Index_B;
//			delete Index_C;
//			delete Index_D;
//			delete Index_R;
//			break;
//		}
//	case back: // Backward
//		{
//			// Determine the index to remove
//			if(curand_uniform(&s)< par.pbl) {
//				mat* B = new mat(data->X->n_row, data->X->n_col);
//				vec* C = new vec(data->X->n_col);
//				u32 ii;
//				for(u32 i = 0; i < B->n_col; i++) {
//					mul(B->col[i], data->X->col[i], data->Beta->e[i]);
//					add(B->col[i], B->col[i], data->R);
//				}
//				inner(C, B);
//				find_min_index(&ii, C);
//				index = data->Index->e[ii];
//				delete B;
//				delete C;
//			}
//			else{
//				index = data->Index->e[curand(&s) % data->Index->n];
//			}
//			break;
//		}
//	}
//	pass_update_cri(data, index);
//	return true;
//}
//
//
///**
// * Compute the value given by criterion
// *
// * @param data the updating data
// * @param index the index to compute
// */
//__device__ bool pass_update_cri(Data* data, const u32 index) {
//	float gamma = 1;
//	u32 k = 0;
//	vec* Xnew = X->col[index];
//	switch(data->stat) {
//	case init: // Initial
//		{
//			data->X->col[0] = Xnew;
//
//			float a;
//			inner(&a, Xnew);
//			data->InvA->col[0]->e[0] = 1 / a;
//
//			inner(&(data->Theta->e[0]), Xnew, Y);
//
//			mul(data->Beta, data->InvA, data->Theta);
//
//			data->Index->e[0] = index;
//
//			k = 1;
//		}
//		break;
//	case forw: // Forward
//		{
//			k = data->Index->n;
//			
//			vec* B = new vec(k);
//			vec* D = new vec(k);
//			mat* InvAtemp = new mat(k+1, k+1);
//			float alpha;
//			float c1;
//			float c2;
//
//			mul(B, Xnew, data->X);
//
//			mul(D, data->InvA, B);
//			
//			inner(&c1, Xnew);
//			inner(&c2, B, D);
//			alpha = 1/(c1 - c2);
//
//			insert(D, -1.0f);
//
//			data->X->n_col++;
//			//vec** temp = new vec*[(k+1)];
//			//memcpy(temp, data->X->col, k * sizeof(vec*));
//			//delete[] data->X->col;
//			//data->X->col = temp;
//			data->X->col[k] = Xnew;
//
//			mul(InvAtemp, D, D);
//			mul(InvAtemp, InvAtemp, alpha);
//			insert(data->InvA, 0.0f);
//			add(data->InvA, data->InvA, InvAtemp);
//			
//			inner(&c1, Xnew, Y);
//			insert(data->Theta, c1);
//			
//			inner(&c2, D, data->Theta);
//			mul(D, D, alpha*c2);
//			insert(data->Beta, 0.0f);
//			add(data->Beta, data->Beta, D);
//
//			insert(data->Index, index);
//			
//			delete B;
//			delete D;
//			delete InvAtemp;
//			k++;
//		}
//		break;
//	case back: // Backward
//		{
//			k = data->Index->n - 1;
//			u32 ii;
//			mat* E = new mat(k, k);
//			vec* F = new vec(k);
//			float g;
//
//			find_index(&ii, data->Index, index);
//			if(ii != k) {
//				swap_col(data->X, ii, k);
//				swap(data->Theta, ii, k);
//				swap(data->Beta, ii, k);
//				swap_row(data->InvA, ii, k);
//				swap_col(data->InvA, ii, k);
//				swap(data->Index, ii, k);
//			}
//
//			data->X->n_col--;
//			//vec** temp = new vec*[k];
//			//memcpy(temp, data->X->col, k * sizeof(vec*));
//			//delete[] data->X->col;
//			//data->X->col = temp;
//
//			shed(data->Theta);
//			shed(data->Index);
//
//			g = data->InvA->col[k]->e[k];
//
//			shed_row(data->InvA);
//			copy(F, data->InvA->col[k]);
//			shed_col(data->InvA);
//
//			mul(E, F, F);
//			mul(E, E, 1/g);
//			sub(data->InvA, data->InvA, E);
//
//			
//			mul(F, F, data->Beta->e[k] / g);
//			shed(data->Beta);
//			sub(data->Beta, data->Beta, F);
//
//			delete E;
//			delete F;
//		}
//		break;
//	}
//
//	mul(data->R, data->X, data->Beta);
//	sub(data->R, Y, data->R);
//
//	norm(&data->e, data->R);
//	
//	switch(cri) {
//	case AIC:
//		data->phi = n * log(data->e * data->e / n) + 2 * k;
//		break;
//	case BIC:
//		data->phi = n * log(data->e * data->e / n) + log((float)n) * k;
//		break;
//	case EBIC:
//		data->phi = n * log(data->e * data->e / n) + log((float)n) * k + 2 * gamma * ((p+.5) * log((float)p) - (k+.5) * log((float)k) - (p-k+.5) * log((float)(p-k)) - .5 * log(2 * CUDART_PI_F));
//		break;
//	case HDBIC:
//		data->phi = n * log(data->e * data->e / n) + log((float)n) * log((float)p) * k;
//		break;
//	case HDHQ:
//		data->phi = n * log(data->e * data->e / n) + 2.01 * log(log((float)n)) * log((float)p) * k;
//		break;
//	}
//	return true;
//}
