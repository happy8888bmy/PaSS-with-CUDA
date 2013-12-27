/**
 * PaSS.cu
 * The main functions of PaSS
 *
 * @author emfo
 */

#include "PaSS_BLAS.cu"
using namespace pass_blas;

#include <ctime>
#include <random>
using namespace std;

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
		vec* Beta;  /**< the vector beta */
		float e;    /**< the norm of R */
		idx* Index; /**< the index of chosen column of X */
		mat* InvA;  /**< the inverse of A */
		float phi;  /**< the value given by criterion */
		vec* R;     /**< the difference between Y and Beta */
		Stat stat;  /**< the status */
		vec* Theta; /**< the vector theta */
		mat* X;     /**< the data we chosen */

		__host__ __device__ Data(u32 n) {
			this->Beta = new vec(1);
			this->Index = new idx(1);
			this->InvA = new mat(1, 1);
			this->R = new vec(n);
			this->Theta = new vec(1);
			this->X = new mat(n, 1);
		}

		__host__ __device__ ~Data() {
			delete this->Beta;
			delete this->Index;
			delete this->InvA;
			delete this->R;
			delete this->Theta;
			delete this->X;
		}
	};
}
using namespace pass;


// Global variables
__device__ u32 n, p, id_best;
__device__ mat* X;
__device__ vec* Y;
__device__ Criterion cri;
__device__ Parameter par;
__device__ idx* Index_best;
__device__ float* phi_all;
__device__ curandState s;

// Functions
void pass_init(float*, float*, const u32, const u32);
void pass_host(const float*, const float*, u32*, u32*, float*, const u32, const u32, const Criterion, const Parameter);
__global__ void pass_kernel(const float*, const float*, u32*, u32*, float*, const u32, const u32, const Criterion, const Parameter);
__device__ bool pass_update_fb(Data*);
__device__ bool pass_update_cri(Data*, const u32);


/**
 * PaSS main function
 */
int main() {
	// Declare variables
	u32 host_n = 8;
	u32 host_p = 256;
	Criterion host_cri = EBIC;
	Parameter host_par = {32, 128, .8f, .1f, .1f, .9f, .1f};
	float* host_X = (float*)malloc(host_n * host_p * sizeof(float));
	float* host_Y = (float*)malloc(host_n * sizeof(float));
	u32* host_I = (u32*)malloc(host_p * sizeof(u32));
	u32 host_k = 0;
	float host_phi;
	
	// Initialize data
	pass_init(host_X, host_Y, host_n, host_p);
		
	// Run PaSS
	pass_host(host_X, host_Y, host_I, &host_k, &host_phi, host_n, host_p, host_cri, host_par);
	
	// Display data
	printf("phi = %f\nIndex = ", host_phi);
	for(u32 i = 0; i < host_k; i++) {
		printf("%d ", host_I[i]);
	}
	printf("\n");
	system("pause");
	return 0;
}


/**
 * Initialize data
 *
 * @param host_X the matrix X
 * @param host_Y the vector Y
 * @param host_n the number of rows in X
 * @param host_p the number of columns in X
 */
__host__ void pass_init(float* host_X, float* host_Y, const u32 host_n, const u32 host_p) {
	u32 i, j;
	u32 q = 10;
	mat* XX = new mat(host_n, host_p);
	vec* YY = new vec(host_n);
	vec* Beta = new vec(host_p, 0.0f);
	vec* X_hat = new vec(host_n, 0.0f);
	vec* Error = new vec(host_n, 0.0f);
	vec* Temp = new vec(host_p);
	float temp;

	// Generate XX and Error using normal random
	default_random_engine generator((u32)time(NULL));
	normal_distribution<float> distribution;
	for(j = 0; j < host_p; j++) {
		for(i = 0; i < host_n; i++) {
			XX->col[j]->e[i] = distribution(generator);
		}
	}
	for(i = 0; i < host_n; i++) {
		Error->e[i] = distribution(generator);
	}
	
	// Compute XX and YY
	Beta->e[0] = 3.0f;
	Beta->e[1] = 3.75f;
	Beta->e[2] = 4.5f;
	Beta->e[3] = 5.25f;
	Beta->e[4] = 6.0f;
	Beta->e[5] = 6.75f;
	Beta->e[6] = 7.5f;
	Beta->e[7] = 8.25f;
	Beta->e[8] = 9.0f;
	Beta->e[9] = 9.75f;
	for(j = 0; j < q; j++) {
		add(X_hat, X_hat, XX->col[j]);
	}
	mul(X_hat, X_hat, 1 / sqrt(2 * (float)q));
	for(j = q; j < host_p; j++) {
		mul(XX->col[j], XX->col[j], .5f);
		add(XX->col[j], XX->col[j], X_hat);
	}
	mul(YY, XX, Beta);
	add(YY, YY, Error);
	
	// Normalize XX
	for(j = 0; j < host_p; j++) {
		norm(&temp, XX->col[j]);
		mul(XX->col[j], XX->col[j], 1/temp);
	}

	// Normalize YY
	norm(&temp, YY);
	mul(YY, YY, 1/temp);

	// Put XX and YY into host_X and host_Y
	for(j = 0; j < host_p; j++) {
		for(i = 0; i < host_n; i++) {
			host_X[i*host_p + j] = XX->col[j]->e[i];
		}
	}
	for(i = 0; i < host_n; i++) {
		host_Y[i] = YY->e[i];
	}
	
	delete XX;
	delete YY;
	delete Beta;
	delete X_hat;
	delete Error;
	delete Temp;
}


/**
 * PaSS host function
 *
 * @param host_X the matrix X
 * @param host_Y the vector Y
 * @param host_I the index I
 * @param host_k length of I
 * @param host_phi the value phi
 * @param host_n the number of rows in X
 * @param host_p the number of columns in X
 * @param host_cri the criterion
 * @param host_par the parameter value
 */
__host__ void pass_host(const float* host_X, const float* host_Y, u32* host_I, u32* host_k, float* host_phi, const u32 host_n, const u32 host_p, const Criterion host_cri, const Parameter host_par) {
	// Declare variables
	float* dev_X = 0;
	float* dev_Y = 0;
	u32* dev_I = 0;
	u32* dev_k = 0;
	float* dev_phi = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!	Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	
	// Allocate GPU buffers for data (two input, one output).
	cudaStatus = cudaMalloc((void**)&dev_X, host_n * host_p * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (X) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Y, host_n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (Y) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_I, host_p * sizeof(u32));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (I) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_k, sizeof(u32));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (k) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_phi, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (phi) failed!\n");
		goto Error;
	}

	// Copy input from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_X, host_X, host_n * host_p * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (X) failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Y, host_Y, host_n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (Y) failed!\n");
		goto Error;
	}
	
	// Launch the kernel function on the GPU with one thread for each element.
	pass_kernel<<<1, host_par.nP>>>(dev_X, dev_Y, dev_I, dev_k, dev_phi, host_n, host_p, host_cri, host_par);

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
	cudaStatus = cudaMemcpy(host_phi, dev_phi, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (phi) failed!\n");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_k, dev_k, sizeof(u32), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (k) failed!\n");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_I, dev_I, *host_k * sizeof(u32), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (I) failed!\n");
		goto Error;
	}

Error:
	cudaFree(dev_X);
	cudaFree(dev_Y);
	cudaFree(dev_I);
	cudaFree(dev_k);
}


/**
 * PaSS kernel function
 *
 * @param host_X the matrix X
 * @param host_Y the vector Y
 * @param host_I the index I
 * @param host_k length of I
 * @param host_phi the value phi
 * @param host_n the number of rows in X
 * @param host_p the number of columns in X
 * @param host_cri the criterion
 * @param host_par the parameter value
 */
__global__ void pass_kernel(const float* host_X, const float* host_Y, u32* host_I, u32* host_k, float* host_phi, const u32 host_n, const u32 host_p, const Criterion host_cri, const Parameter host_par) {
	// Declare variables
	u32 id = threadIdx.x;
	u32 i, j;
	Data* data = new Data(host_n);
	float phi_old;
	
	if(id == 0) {
		// Initialize Random Seed
		curand_init(clock64(), 0, 0, &s);

		// Declare variables
		n = host_n;
		p = host_p;
		cri = host_cri;
		par = host_par;
		X = new mat(n, p);
		Y = new vec(n);
		Index_best = new idx(p);
		phi_all = new float[par.nP];	
	
		// Copy X and Y from array to matrix
		for(u32 i = 0; i < n; i++) {
			for(j = 0; j < p; j++) {
				X->col[j]->e[i] = host_X[i*p + j];
			}
			Y->e[i] = host_Y[i];
		}
	}
	__syncthreads();

	// Initialize Particles
	__shared__ vec* C;
	__shared__ idx* I;
	if(p >= par.nP && id == 0) {
		C = new vec(p);
		I = new idx(p);
		mul(C, Y, X);
		for(i = 0; i < C->n; i++) {
			C->e[i] = abs(C->e[i]);
		}
		sort_index_descend(I, C);
	}
	__syncthreads();

	data->stat = init;
	if(p >= par.nP) {
		pass_update_cri(data, I->e[id]);
	} else {
		pass_update_cri(data, curand(&s) % p);
	}
	if(p >= par.nP && id == 0) {
		delete C;
		delete I;
	}
	phi_old = data->phi;
	data->stat = forw;

	// Choose Global Best
	if(id == 0) {
		*host_phi = data->phi;
		put(Index_best, data->Index);
	}
	__syncthreads();
	
	
	// Find Best Data
	for(i = 0; i < par.nI; i++) {
		// Update data
		pass_update_fb(data);
		if(data->phi - phi_old > 0) {
			data->stat = (Stat)(-data->stat);
		}
		if(data->Index->n <= 1) {
			data->stat = forw;
		}
		if(data->Index->n >= p - 5) {
			data->stat = back;
		}
		phi_old = data->phi;

		// Choose Global Best
		phi_all[id] = data->phi;
		__syncthreads();
		if(id == 0) {
			id_best = (u32)(-1);
			for(j = 0; j < par.nP; j++) {
				if(phi_all[j] < *host_phi) {
					id_best = j;
					*host_phi = data->phi;
				}
			}
		}
		__syncthreads();
		if(id == id_best) {
			put(Index_best, data->Index);
		}
		__syncthreads();
	}
	
	// Delete variables
	delete data;

	if(id == 0) {
		// Copy Index_best from index to array
		sort_ascend(Index_best);
		for(j = 0; j < Index_best->n; j++) {
			host_I[j] = Index_best->e[j];
		}
		*host_k = Index_best->n;
		printf("phi = %f\nk = %d\nIndex = ", *host_phi, *host_k);
		print(Index_best);

		// Delete variables
		delete X;
		delete Y;
		delete Index_best;
		delete phi_all;
	}
}


/**
 * Determine forward or backward
 */
__device__ bool pass_update_fb(Data* data) {
	u32 index = 0;
	switch(data->stat) {
	case forw: // Forward
		{
			idx* Index_B = new idx(Index_best->n);
			idx* Index_C = new idx(data->Index->n);
			idx* Index_D = new idx(Index_B->n);
			idx* Index_R = new idx(p - Index_C->n);

			// Sort Index_B
			copy(Index_B, Index_best);
			sort_ascend(Index_B);
			
			// Sort Index_C
			copy(Index_C, data->Index);
			sort_ascend(Index_C);
			
			// Let Index_D be Index_B exclude Index_C
			set_difference(Index_D, Index_B, Index_C);
			
			// Let Index_R be the complement of Index_C
			complement(Index_R, Index_C, p);

			// Determine the index to add
			if(curand_uniform(&s) < par.pfg && Index_D->n > 0) {
				index = Index_D->e[curand(&s) % Index_D->n];
			}
			else if(curand_uniform(&s) < par.pfl/(par.pfl+par.pfr)) {
				float phi_max = -1, phi_temp;
				for(u32 i = 0; i < Index_R->n; i++) {
					inner(&phi_temp, data->R, X->col[Index_R->e[i]]);
					phi_temp = abs(phi_temp);
					if(phi_temp > phi_max) {
						phi_max = phi_temp;
						index = Index_R->e[i];
					}
				}
			}
			else{
				index = Index_R->e[curand(&s) % Index_R->n];
			}
			delete Index_B;
			delete Index_C;
			delete Index_D;
			delete Index_R;
			break;
		}
	case back: // Backward
		{
			// Determine the index to remove
			if(curand_uniform(&s)< par.pbl) {
				mat* B = new mat(data->X->n_row, data->X->n_col);
				vec* C = new vec(data->X->n_col);
				u32 ii;
				for(u32 i = 0; i < B->n_col; i++) {
					mul(B->col[i], data->X->col[i], data->Beta->e[i]);
					add(B->col[i], B->col[i], data->R);
				}
				inner(C, B);
				find_min_index(&ii, C);
				index = data->Index->e[ii];
				delete B;
				delete C;
			}
			else{
				index = data->Index->e[curand(&s) % data->Index->n];
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
	float gamma = 1;
	u32 k = 0;
	vec* Xnew = X->col[index];
	switch(data->stat) {
	case init: // Initial
		{
			copy(data->X->col[0], Xnew);

			float a;
			inner(&a, Xnew);
			data->InvA->col[0]->e[0] = 1 / a;

			inner(&(data->Theta->e[0]), Xnew, Y);

			mul(data->Beta, data->InvA, data->Theta);

			data->Index->e[0] = index;

			k = 1;
		}
		break;
	case forw: // Forward
		{
			k = data->Index->n;
			
			vec* B = new vec(k);
			vec* D = new vec(k);
			mat* InvAtemp = new mat(k+1, k+1);
			float alpha;
			float c1;
			float c2;

			mul(B, Xnew, data->X);

			mul(D, data->InvA, B);
			
			inner(&c1, Xnew);
			inner(&c2, B, D);
			alpha = 1/(c1 - c2);

			insert(D, -1.0f);

			insert_col(data->X, Xnew);

			mul(InvAtemp, D, D);
			mul(InvAtemp, InvAtemp, alpha);
			insert(data->InvA, 0.0f);
			add(data->InvA, data->InvA, InvAtemp);
			
			inner(&c1, Xnew, Y);
			insert(data->Theta, c1);
			
			inner(&c2, D, data->Theta);
			mul(D, D, alpha*c2);
			insert(data->Beta, 0.0f);
			add(data->Beta, data->Beta, D);

			insert(data->Index, index);
			
			delete B;
			delete D;
			delete InvAtemp;
			k++;
		}
		break;
	case back: // Backward
		{
			k = data->Index->n - 1;
			u32 ii;
			mat* E = new mat(k, k);
			vec* F = new vec(k);
			float g;

			find_index(&ii, data->Index, index);
			if(ii != k) {
				swap_col(data->X, ii, k);
				swap(data->Theta, ii, k);
				swap(data->Beta, ii, k);
				swap_row(data->InvA, ii, k);
				swap_col(data->InvA, ii, k);
				swap(data->Index, ii, k);
			}

			shed_col(data->X);
			shed(data->Theta);
			shed(data->Index);

			g = data->InvA->col[k]->e[k];

			shed_row(data->InvA);
			copy(F, data->InvA->col[k]);
			shed_col(data->InvA);

			mul(E, F, F);
			mul(E, E, 1/g);
			sub(data->InvA, data->InvA, E);

			
			mul(F, F, data->Beta->e[k] / g);
			shed(data->Beta);
			sub(data->Beta, data->Beta, F);

			delete E;
			delete F;
		}
		break;
	}

	mul(data->R, data->X, data->Beta);
	sub(data->R, Y, data->R);

	norm(&data->e, data->R);
	
	switch(cri) {
	case AIC:
		data->phi = n * log(data->e * data->e / n) + 2 * k;
		break;
	case BIC:
		data->phi = n * log(data->e * data->e / n) + log((float)n) * k;
		break;
	case EBIC:
		data->phi = n * log(data->e * data->e / n) + log((float)n) * k + 2 * gamma * ((p+.5) * log((float)p) - (k+.5) * log((float)k) - (p-k+.5) * log((float)(p-k)) - .5 * log(2 * CUDART_PI_F));
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
