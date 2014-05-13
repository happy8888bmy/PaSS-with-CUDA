/*
 * PaSS with CUDA
 * PaSS.cu
 * The main functions
 */

/**@mainpage
 * @author Mu Yang
 * @author Da-Wei Chang
 * @author Chen-Yao Lin
 * @date Jan. 19, 2014
 * @version 2014Jan19a
 */

#include "PaSS_BLAS.cu"
using namespace pass_blas;

#include <iostream>
#include <ctime>
#include <thrust/sort.h>
#include <thrust/functional.h>
using namespace std;

#define PASS_MAX_N 512
#define PASS_MAX_P 4096
#define PASS_MAX_K 16
#define PASS_GAMMA 1

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
		EBIC,  /**< Extended Bayesian information criterion */
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
		vec Beta[PASS_MAX_K+2];                /**< the vector beta */
		float e;                               /**< the norm of R */
		uvec I[PASS_MAX_K+2];                  /**< the index of chosen column of X */
		mat InvA[(PASS_MAX_K+2)*PASS_MAX_K+2]; /**< the inverse of A */
		float phi;                             /**< the value given by criterion */
		vec R[PASS_MAX_N+2];                   /**< the difference between Y and Beta */
		Stat stat;                             /**< the status */
		vec Theta[PASS_MAX_K+2];               /**< the vector theta */
		mat X[(PASS_MAX_N+2)*PASS_MAX_K+2];    /**< the data we chosen */
		vec Y[PASS_MAX_N+2];                   /**< the data Y */
	};
}
using namespace pass;


// Global variables
__device__ u32 n, p;
__device__ const mat* X;
__device__ const vec* Y;
__device__ uvec* I;
__device__ u32 id_best;
__device__ float* phi_all;
__device__ Criterion cri;
__device__ Parameter par;
__device__ curandState seed;
__device__ vec* CC;
__device__ uvec* II;

// Functions
void pass_init(mat*, vec*);
void pass_host(const mat*, const vec*, uvec*, float*, const Criterion, const Parameter, float*);
__global__ void pass_kernel(const mat*, const vec*, uvec*, float*, const Criterion, const Parameter);
__device__ bool pass_update_fb(Data*);
__device__ bool pass_update_cri(Data*, const u32);


/**
 * PaSS main function
 */
int main() {
	// Declare variables
	u32 host_n = 400;
	u32 host_p = 4000;
	Criterion host_cri = EBIC;
	Parameter host_par = {32, 128, .8f, .1f, .1f, .9f, .1f};
	
	if(host_n > PASS_MAX_N) {
		printf("n must smaller than %d!\n", PASS_MAX_N);
		system("pause");
		return 1;
	}
	if(host_p > PASS_MAX_P) {
		printf("p must smaller than %d!\n", PASS_MAX_P);
		system("pause");
		return 1;
	}
	if(host_p < 10) {
		printf("p must greater than 10!\n");
		system("pause");
		return 1;
	}
	
	// Declare matrices
	mat* host_X = new mat[(PASS_MAX_N+2)*host_p+2];
	n_cols(host_X) = host_p;
	m_cols(host_X) = host_p;
	for(u32 j = 0; j < host_p; j++) {
		n_ents(col(host_X, j)) = host_n;
		m_ents(col(host_X, j)) = PASS_MAX_N;
	}
	vec* host_Y = new vec[PASS_MAX_N+2];
	m_ents(host_Y) = PASS_MAX_N;
	uvec* host_I = new uvec[PASS_MAX_K+2];
	m_ents(host_I) = PASS_MAX_K;
	float host_phi, time;
	
	const u32 nTest = 10;
	
	for(u32 t = 0; t < nTest; t++) {
		n_ents(host_I) = 0;
		
		// Initialize data
		pass_init(host_X, host_Y);
			
		// Run PaSS
		pass_host(host_X, host_Y, host_I, &host_phi, host_cri, host_par, &time);
		
		// Display data
		printf("%3d:", t);
		print_uvec(host_I);
	}
	cout << "Used " << time/nTest  << " seconds on average." << endl;
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
	n_ents(Beta) = p;
	m_ents(Beta) = p;
	vec* X_hat = new vec[n+4];
	n_ents(X_hat) = n;
	m_ents(X_hat) = n;
	vec* Error = new vec[n+4];
	n_ents(Error) = n;
	m_ents(Error) = n;
	vec* Temp = new vec[p+4];
	m_ents(Temp) = p;
	float temp;

	// Generate X and Error using normal random
	srand (time(NULL));
	for(j = 0; j < n_cols(X); j++) {
		for(i = 0; i < n_rows(X); i++) {
			entry2(X, i, j) = randn(0,1);
		}
	}
	for(i = 0; i < n_cols(Error); i++) {
		entry(Error, i) = randn(0,1);
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
	for(i = 0; i < n; i++) {
		entry(X_hat, i) = 0;
		for(j = 0; j < q; j++) {
			entry(X_hat, i) += entry2(X, i, j);
		}
	}
	mul_vec_host(X_hat, X_hat, 1 / sqrt(2 * (float)q));
	for(j = q; j < p; j++) {
		mul_vec_host(col(X, j), col(X, j), .5f);
		add_vec_host(col(X, j), col(X, j), X_hat);
	}
	mul_matvec_host(Y, X, Beta);
	add_vec_host(Y, Y, Error);
	
	// Normalize X
	for(j = 0; j < p; j++) {
		norm_vec_host(&temp, col(X, j));
		mul_vec_host(col(X, j), col(X, j), 1/temp);
	}

	// Normalize Y
	norm_vec_host(&temp, Y);
	mul_vec_host(Y, Y, 1/temp);

	delete[] Beta;
	delete[] X_hat;
	delete[] Error;
	delete[] Temp;

}


/**
 * PaSS host function
 *
 * @param host_X the matrix X
 * @param host_Y the vector Y
 * @param host_I the index I
 * @param host_phi the value phi
 * @param host_cri the criterion
 * @param host_par the parameter value
 */
__host__ void pass_host(const mat* host_X, const vec* host_Y, uvec* host_I, float* host_phi, const Criterion host_cri, const Parameter host_par, float* time) {
	// Declare variables
	mat* dev_X = 0;
	vec* dev_Y = 0;
	uvec* dev_I = 0;
	float* dev_phi = 0;
	cudaError_t cudaStatus;
	clock_t start_clock;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!	Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	
	// Allocate GPU buffers for data->
	cudaStatus = cudaMalloc((void**)&dev_X, size_mat(host_X) * sizeof(mat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (X) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Y, size_vec(host_Y) * sizeof(vec));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (Y) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_I, size_vec(host_I) * sizeof(uvec));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (I) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_phi, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (phi) failed!\n");
		goto Error;
	}

	// Copy input from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_X, host_X, size_mat(host_X) * sizeof(mat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (X) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Y, host_Y, size_vec(host_Y) * sizeof(vec), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (Y) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_I, host_I, size_vec(host_I) * sizeof(uvec), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (I) failed!\n");
		goto Error;
	}
	
	// Launch the kernel function on the GPU with one thread for each element.
	start_clock = clock();
	pass_kernel<<<1, host_par.nP>>>(dev_X, dev_Y, dev_I, dev_phi, host_cri, host_par);
	cudaThreadSynchronize();
	*time += (float)(clock() - start_clock) / CLOCKS_PER_SEC;

	// Check for any errors launching the kernel.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_I, dev_I, size_vec(host_I) * sizeof(uvec), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (I) failed!\n");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_phi, dev_phi, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (phi) failed!\n");
		goto Error;
	}

Error:
	cudaFree(dev_X);
	cudaFree(dev_Y);
	cudaFree(dev_I);
	cudaFree(dev_phi);
}


/**
 * PaSS kernel function
 *
 * @param dev_X the matrix X
 * @param dev_Y the vector Y
 * @param dev_I the index I
 * @param dev_phi the value phi
 * @param dev_cri the criterion
 * @param dev_par the parameter value
 */
__global__ void pass_kernel(const mat* dev_X, const vec* dev_Y, uvec* dev_I, float* dev_phi, const Criterion dev_cri, const Parameter dev_par) {
	// Declare variables
	u32 tid = threadIdx.x;
	u32 i, j;
	Data* data = new Data();
	float phi_old;
	
	if(tid == 0) {
		// Initialize Random Seed
		curand_init(clock64(), 0, 0, &seed);

		// Declare variables
		n = n_rows(dev_X);
		p = n_cols(dev_X);
		cri = dev_cri;
		par = dev_par;
		X = dev_X;
		Y = dev_Y;
		I = dev_I;
		phi_all = new float[par.nP];
	}
	__syncthreads();

	// Initialize Particles
	if(p >= par.nP && tid == 0) {
		CC = new vec[p];
		II = new uvec[p];
		m_ents(CC) = p;
		m_ents(II) = p;
		mul_vecmat(CC, Y, X);
		for(i = 0; i < n_ents(CC); i++) {
			entry(CC, i) = abs(entry(CC, i));
		}
		sort_index_descend(II, CC);
	}
	__syncthreads();

	data->stat = init;
	if(p >= par.nP) {
		pass_update_cri(data, entry(II, tid));
	} else {
		pass_update_cri(data, curand(&seed) % p);
	}
	if(p >= par.nP && tid == 0) {
		delete[] CC;
		delete[] II;
	}
	phi_old = data->phi;
	data->stat = forw;

	// Choose Global Best
	if(tid == 0) {
		*dev_phi = data->phi;
		copy_uvec(I, data->I);
	}
	__syncthreads();
	
	
	// Find Best Data
	for(i = 0; i < par.nI; i++) {
		// Update data
		pass_update_fb(data);
		if(data->phi - phi_old > 0) {
			data->stat = (Stat)(-data->stat);
		}
		if(n_ents(data->I) <= 1) {
			data->stat = forw;
		}
		if(n_ents(data->I) >= PASS_MAX_K-1) {
			data->stat = back;
		}
		phi_old = data->phi;

		// Choose Global Best
		phi_all[tid] = data->phi;
		__syncthreads();
		if(tid == 0) {
			id_best = (u32)(-1);
			for(j = 0; j < par.nP; j++) {
				if(phi_all[j] < *dev_phi) {
					id_best = j;
					*dev_phi = data->phi;
				}
			}
		}
		__syncthreads();
		if(tid == id_best) {
			copy_uvec(I, data->I);
		}
		__syncthreads();
	}
	
	if(tid == 0) {
		// Sort I
		sort_ascend(I);
		
		// Delete variables
		delete phi_all;
	}
	
	delete data;
}


/**
 * Determine forward or backward
 * 
 * @param data the updating data
 */
__device__ bool pass_update_fb(Data* data) {
	u32 i, index = 0;
	switch(data->stat) {
	case forw: // Forward
		{
			uvec* I_B = new uvec[PASS_MAX_K+2];
			uvec* I_C = new uvec[PASS_MAX_K+2];
			uvec* I_D = new uvec[PASS_MAX_K+2];
			uvec* I_R = new uvec[PASS_MAX_P+2];
			m_ents(I_D) = PASS_MAX_K;
			m_ents(I_R) = PASS_MAX_P;

			// Sort I_B
			copy_uvec(I_B, I);
			sort_ascend(I_B);
			
			// Sort I_C
			copy_uvec(I_C, data->I);
			sort_ascend(I_C);
			
			// Let I_D be I_B exclude I_C
			set_diff(I_D, I_B, I_C);
			
			// Let I_R be the complement of I_C
			complement(I_R, I_C, n_cols(X));

			// Determine the index to add
			if(curand_uniform(&seed) < par.pfg && n_ents(I_D) > 0) {
				index = entry(I_D, curand(&seed) % n_ents(I_D));
			}
			else if(curand_uniform(&seed) < par.pfl/(par.pfl+par.pfr)) {
				float phi_max = -1, phi_temp;
				for(i = 0; i < n_ents(I_R); i++) {
					inner_vec(&phi_temp, data->R, col(X, entry(I_R, i)));
					phi_temp = abs(phi_temp);
					if(phi_temp > phi_max) {
						phi_max = phi_temp;
						index = entry(I_R, i);
					}
				}
			}
			else{
				index = entry(I_R, curand(&seed) % n_ents(I_R));
			}
			
			delete[] I_B;
			delete[] I_C;
			delete[] I_D;
			delete[] I_R;
			
			break;
		}
	case back: // Backward
		{
			// Determine the index to remove
			if(curand_uniform(&seed)< par.pbl) {
				mat* B = new mat[(PASS_MAX_N+2)*PASS_MAX_K+2];
				vec* C = new vec[PASS_MAX_K+2];
				n_cols(B) = n_cols(data->X);
				m_cols(B) = PASS_MAX_K;
				for(i = 0; i < n_cols(B); i++) {
					m_ents(col(B, i)) = PASS_MAX_N;
				}
				m_ents(C) = PASS_MAX_K;
				for(i = 0; i < n_cols(B); i++) {
					mul_vec(col(B, i), col(data->X, i), entry(data->Beta, i));
					add_vec(col(B, i), col(B, i), data->R);
				}
				inner_mat(C, B);
				find_min_index(&i, C);
				index = entry(data->I, i);
				delete[] B;
				delete[] C;
			}
			else{
				index = entry(data->I, curand(&seed) % n_ents(data->I));
			}
			break;
		}
	}
	pass_update_cri(data, index);
	return true;
}


/**
 * Compute the value given by criterion
 *
 * @param data the updating data
 * @param index the index to compute
 */
__device__ bool pass_update_cri(Data* data, const u32 index) {
	u32 k = 0;
	vec* Xnew;
	switch(data->stat) {
	case init: // Initial
		{
			n_cols(data->X) = 1;
			m_cols(data->X) = PASS_MAX_K;
			copy_vec(col(data->X, 0), col(X, index));
			Xnew = (vec*)col(X, index);
			
			n_rows(data->InvA) = 1;
			n_cols(data->InvA) = 1;
			m_rows(data->InvA) = PASS_MAX_K;
			m_cols(data->InvA) = PASS_MAX_K;
			float a;
			inner_vec(&a, Xnew);
			entry2(data->InvA, 0, 0) = 1 / a;
			
			n_ents(data->Theta) = 1;
			m_ents(data->Theta) = PASS_MAX_K;
			inner_vec(&entry(data->Theta, 0), Xnew, Y);
			
			m_ents(data->Beta) = PASS_MAX_K;
			mul_matvec(data->Beta, data->InvA, data->Theta);
			
			n_ents(data->I) = 1;
			m_ents(data->I) = PASS_MAX_K;
			entry(data->I, 0) = index;
			
			m_ents(data->R) = PASS_MAX_N;
			
			copy_vec(data->Y, Y);
			
			k = 1;
		}
		break;
	case forw: // Forward
		{
			k = n_ents(data->I);
			
			vec* B = new vec[PASS_MAX_K+2];
			vec* D = new vec[PASS_MAX_K+2];
			mat* M = new mat[(PASS_MAX_K+2)*PASS_MAX_K+2];
			float alpha;
			float c1;
			float c2;
			m_ents(B) = PASS_MAX_K;
			m_ents(D) = PASS_MAX_K;
			m_rows(M) = PASS_MAX_K;
			m_cols(M) = PASS_MAX_K;

			copy_vec(col(data->X, k), col(X, index));
			Xnew = col(data->X, k);
			mul_vecmat(B, Xnew, data->X);

			mul_matvec(D, data->InvA, B);
			
			inner_vec(&c1, Xnew);
			inner_vec(&c2, B, D);
			alpha = 1/(c1 - c2);
			
			insert_vec(D, -1.0f);

			n_cols(data->X)++;

			mul_vecvec(M, D, D);
			mul_mat(M, M, alpha);
			insert_mat(data->InvA, 0.0f);
			add_mat(data->InvA, data->InvA, M);
			
			inner_vec(&c1, Xnew, Y);
			insert_vec(data->Theta, c1);
			
			inner_vec(&c2, D, data->Theta);
			mul_vec(D, D, alpha*c2);
			insert_vec(data->Beta, 0.0f);
			add_vec(data->Beta, data->Beta, D);

			insert_uvec(data->I, index);
			
			delete[] B;
			delete[] D;
			delete[] M;
			
			k++;
		}
		break;
	case back: // Backward
		{
			k = n_ents(data->I) - 1;
			u32* i = new u32[1];
			mat* E = new mat[(PASS_MAX_K+2)*PASS_MAX_K+2];
			vec* F = new vec[PASS_MAX_K+2];
			float g;
			m_rows(E) = PASS_MAX_K;
			m_cols(E) = PASS_MAX_K;
			m_ents(F) = PASS_MAX_K;

			find_index(i, data->I, index);
			if(*i != k) {
				swap_col(data->X, *i, k);
				swap_vec(data->Theta, *i, k);
				swap_vec(data->Beta, *i, k);
				swap_row(data->InvA, *i, k);
				swap_col(data->InvA, *i, k);
				swap_uvec(data->I, *i, k);
			}

			shed_col(data->X);
			shed_vec(data->Theta);
			shed_uvec(data->I);

			g = entry2(data->InvA, k, k);
			
			shed_row(data->InvA);
			copy_vec(F, col(data->InvA, k));
			shed_col(data->InvA);
			
			mul_vecvec(E, F, F);
			mul_mat(E, E, 1/g);
			sub_mat(data->InvA, data->InvA, E);
			
			mul_vec(F, F, entry(data->Beta, k) / g);
			shed_vec(data->Beta);
			sub_vec(data->Beta, data->Beta, F);
			
			delete[] i;
			delete[] E;
			delete[] F;
		}
		break;
	}

	mul_matvec(data->R, data->X, data->Beta);
	sub_vec(data->R, data->Y, data->R);

	norm_vec(&data->e, data->R);
	
	switch(cri) {
	case AIC:
		data->phi = n * log(data->e * data->e / n) + 2 * k;
		break;
	case BIC:
		data->phi = n * log(data->e * data->e / n) + log((float)n) * k;
		break;
	case EBIC:
		data->phi = n * log(data->e * data->e / n) + log((float)n) * k + 2 * PASS_GAMMA * ((p+.5) * log((float)p) - (k+.5) * log((float)k) - (p-k+.5) * log((float)(p-k)) - .5 * log(2 * CUDART_PI_F));
		break;
	case HDBIC:
		data->phi = n * log(data->e * data->e / n) + log((float)n) * log((float)p) * k;
		break;
	case HDHQ:
		data->phi = n * log(data->e * data->e / n) + 2.01 * log(log((float)n)) * log((float)p) * k;
		break;
	}
	return true;
}
