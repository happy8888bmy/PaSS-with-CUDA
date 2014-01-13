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
		vec Beta[PASS_MAX_P+2];                /**< the vector beta */
		float e;                               /**< the norm of R */
		uvec I[PASS_MAX_P+2];                  /**< the index of chosen column of X */
		mat InvA[(PASS_MAX_P+2)*PASS_MAX_P+2]; /**< the inverse of A */
		float phi;                             /**< the value given by criterion */
		vec R[PASS_MAX_N+2];                   /**< the difference between Y and Beta */
		Stat stat;                             /**< the status */
		vec Theta[PASS_MAX_P+2];               /**< the vector theta */
		mat X[(PASS_MAX_N+2)*PASS_MAX_P+2];    /**< the data we chosen */
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
	u32 host_p = 16;
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
	pass_host(host_X, host_Y, host_I, &host_phi, host_cri, host_par);
	
	// Display data
	printf("X:\n");
	print_mat(host_X);
	printf("Y:\n");
	print_vec(host_Y);
	printf("I:\n");
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


/**
 * PaSS host function
 *
 * @param host_X the matrix X
 * @param host_Y the vector Y
 * @param host_I the index IIII
 * @param host_phi the value phi
 * @param host_cri the criterion
 * @param host_par the parameter value
 */
__host__ void pass_host(const mat* host_X, const vec* host_Y, uvec* host_I, float* host_phi, const Criterion host_cri, const Parameter host_par) {
	// Declare variables
	mat* dev_X = 0;
	vec* dev_Y = 0;
	uvec* dev_I = 0;
	float* dev_phi = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!	Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	
	// Allocate GPU buffers for data.
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
	// pass_kernel<<<1, host_par.nP>>>(dev_X, dev_Y, dev_I, dev_phi, host_cri, host_par);
	pass_kernel<<<1, 1>>>(dev_X, dev_Y, dev_I, dev_phi, host_cri, host_par);

	// Check for any errors launching the kernel.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %seed\n", cudaGetErrorString(cudaStatus));
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
		fprintf(stderr, "cudaMemcpy (IIII) failed!\n");
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
__global__ void pass_kernel(const mat* dev_X, const vec* dev_Y, uvec* dev_I, float* const dev_phi, Criterion dev_cri, Parameter dev_par) {
	// Declare variables
	u32 id = threadIdx.x;
	u32 i, j;
	Data data;
	float phi_old;
	
	if(id == 0) {
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
	if(p >= par.nP && id == 0) {
		CC = new vec[p];
		II = new uvec[p];
		mul_vecmat(CC, Y, X);
		for(i = 0; i < n_ents(CC); i++) {
			entry(CC, i) = abs(entry(CC, i));
		}
		sort_index_descend(II, CC);
	}
	__syncthreads();

	data.stat = init;
	if(p >= par.nP) {
		pass_update_cri(&data, entry(II, id));
	} else {
		pass_update_cri(&data, curand(&seed) % p);
	}
	if(p >= par.nP && id == 0) {
		delete CC;
		delete II;
	}
	phi_old = data.phi;
	data.stat = forw;

	// Choose Global Best
	if(id == 0) {
		*dev_phi = data.phi;
		copy_uvec(I, data.I);
	}
	__syncthreads();
	
	
	// Find Best Data
	for(i = 0; i < par.nI; i++) {
		// Update data
		pass_update_fb(&data);
		if(data.phi - phi_old > 0) {
			data.stat = (Stat)(-data.stat);
		}
		if(n_ents(data.I) <= 1) {
			data.stat = forw;
		}
		if(n_ents(data.I) >= PASS_MAX_P-1) {
			data.stat = back;
		}
		phi_old = data.phi;

		// Choose Global Best
		phi_all[id] = data.phi;
		__syncthreads();
		if(id == 0) {
			id_best = (u32)(-1);
			for(j = 0; j < par.nP; j++) {
				if(phi_all[j] < *dev_phi) {
					id_best = j;
					*dev_phi = data.phi;
				}
			}
		}
		__syncthreads();
		if(id == id_best) {
			copy_uvec(I, data.I);
		}
		__syncthreads();
	}
	
	if(id == 0) {
		// Sort I
		sort_ascend(I);
		
		// Delete variables
		delete phi_all;
	}
}


/**
 * Determine forward or backward
 */
__device__ bool pass_update_fb(Data* data) {
	u32 i, index = 0;
	switch(data->stat) {
	case forw: // Forward
		{
			uvec I_B[PASS_MAX_P+2];
			uvec I_C[PASS_MAX_P+2];
			uvec I_D[PASS_MAX_P+2];
			uvec I_R[PASS_MAX_P+2];

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
			break;
		}
	case back: // Backward
		{
			// Determine the index to remove
			if(curand_uniform(&seed)< par.pbl) {
				mat B[(PASS_MAX_N+2)*PASS_MAX_P+2];
				init_mat(B, n_rows(data->X), n_cols(data->X), PASS_MAX_N, PASS_MAX_P);
				vec C[PASS_MAX_P+2];
				init_vec(C, n_cols(data->X), PASS_MAX_P);
				u32 ii;
				for(i = 0; i < n_cols(B); i++) {
					mul_vec(col(B, i), col(data->X, i), entry(data->Beta, i));
					add_vec(col(B, i), col(B, i), data->R);
				}
				inner_mat(C, B);
				find_min_index(&ii, C);
				index = entry(data->I, ii);
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
			m_cols(data->X) = PASS_MAX_P;
			copy_vec(col(data->X, 0), col(X, index));
			Xnew = (vec*)col(X, index);

			init_mat(data->InvA, 1, 1, PASS_MAX_P, PASS_MAX_P);
			float a;
			inner_vec(&a, Xnew);
			entry2(data->InvA, 0, 0) = 1 / a;

			init_vec(data->Theta, 1, PASS_MAX_P);
			inner_vec(&entry(data->Theta, 0), Xnew, Y);
			
			init_vec(data->Beta, 1, PASS_MAX_P);
			mul_matvec(data->Beta, data->InvA, data->Theta);
			
			init_uvec(data->I, 1, PASS_MAX_P);
			entry(data->I, 0) = index;

			init_vec(data->R, 1, PASS_MAX_N);

			k = 1;
		}
		break;
	case forw: // Forward
		{
			k = n_ents(data->I);
			
			vec B[PASS_MAX_P+2];
			vec D[PASS_MAX_P+2];
			mat InvAtemp[(PASS_MAX_P+2)*PASS_MAX_P+2];
			float alpha;
			float c1;
			float c2;
			init_vec(B, k, PASS_MAX_P);
			init_vec(D, k, PASS_MAX_P);
			init_mat(InvAtemp, k+1, k+1, PASS_MAX_P, PASS_MAX_P);

			copy_vec(col(data->X, k), col(X, index));
			Xnew = col(data->X, k);
			mul_vecmat(B, Xnew, data->X);

			mul_matvec(D, data->InvA, B);
			
			inner_vec(&c1, Xnew);
			inner_vec(&c2, B, D);
			alpha = 1/(c1 - c2);
			
			insert_vec(D, -1.0f);

			n_cols(data->X)++;

			mul_vecvec(InvAtemp, D, D);
			mul_mat(InvAtemp, InvAtemp, alpha);
			insert_mat(data->InvA, 0.0f);
			add_mat(data->InvA, data->InvA, InvAtemp);
			
			inner_vec(&c1, Xnew, Y);
			insert_vec(data->Theta, c1);
			
			inner_vec(&c2, D, data->Theta);
			mul_vec(D, D, alpha*c2);
			insert_vec(data->Beta, 0.0f);
			add_vec(data->Beta, data->Beta, D);

			insert_uvec(data->I, index);
			
			k++;
		}
		break;
	case back: // Backward
		{
			k = n_ents(data->I) - 1;
			u32 ii;
			mat E[(PASS_MAX_P+2)*PASS_MAX_P+2];
			vec F[PASS_MAX_P+2];
			float g;

			find_index(&ii, data->I, index);
			if(ii != k) {
				swap_col(data->X, ii, k);
				swap_vec(data->Theta, ii, k);
				swap_vec(data->Beta, ii, k);
				swap_row(data->InvA, ii, k);
				swap_col(data->InvA, ii, k);
				swap_uvec(data->I, ii, k);
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
		}
		break;
	}

	mul_matvec(data->R, data->X, data->Beta);
	sub_vec(data->R, Y, data->R);

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
