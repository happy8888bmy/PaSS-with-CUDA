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
	};
}
using namespace pass;


// Global variables
__device__ u32 n, p;
__device__ mat* X;
__device__ vec* Y;
__device__ Criterion cri = EBIC;
__device__ Parameter par;
__device__ float data_best_phi;
__device__ idx* data_best_Index;
__device__ curandState s;

__device__ vec* Phi;
__device__ Data* data;

// Functions
void pass_init(float*, float*, const u32, const u32);
u32 pass_host(const float*, const float*, u32*, const u32, const u32, const Parameter);
__global__ void pass_kernel(const float*, const float*, u32*, const u32, const u32, const Parameter, u32*);
__device__ bool pass_update_fb(Data*);
__device__ bool pass_update_cri(Data*, const u32);

__global__ void pass_find_best(const u32);


/**
 * PaSS main function
 */
int main() {
	// Declare variables
	u32 host_n = 20;
	u32 host_p = 16;
	Parameter host_par = {16, 128, .8f, .1f, .1f, .9f, .1f};
	u32 k;
	float* host_X = (float*)malloc(host_n * host_p * sizeof(float));
	float* host_Y = (float*)malloc(host_n * sizeof(float));
	u32* host_I = (u32*)malloc(host_p * sizeof(u32));
	
	// Initialize data
	pass_init(host_X, host_Y, host_n, host_p);
		
	// Run PaSS
	k = pass_host(host_X, host_Y, host_I, host_n, host_p, host_par);
	
	// Display data
	for(u32 i = 0; i < k; i++) {
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
 * @param host_n the number of rows in X
 * @param host_p the number of columns in X
 * @param host_par the parameter data
 */
__host__ u32 pass_host(const float* host_X, const float* host_Y, u32* host_I, const u32 host_n, const u32 host_p, const Parameter host_par) {
	// Declare variables
	u32 host_k = 0;
	float* dev_X = 0;
	float* dev_Y = 0;
	u32* dev_I = 0;
	u32* dev_k = 0;
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
	pass_kernel<<<1, 1>>>(dev_X, dev_Y, dev_I, host_n, host_p, host_par, dev_k);

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

	//system("pause");
	for(u32 i = 0; i < host_par.nI; i++) {
		//printf("iteration = %d\n", i);
		pass_find_best<<<1, 1>>>(i);
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
		//system("pause");
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_I, dev_I, host_p * sizeof(u32), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (I) failed!\n");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(&host_k, dev_k, sizeof(u32), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (k) failed!\n");
		goto Error;
	}

Error:
	cudaFree(dev_X);
	cudaFree(dev_Y);
	cudaFree(dev_I);
	cudaFree(dev_k);
	return host_k;
}


/**
 * PaSS kernel function
 *
 * @param array_X the matrix X
 * @param array_Y the vector Y
 * @param host_n the number of rows in X
 * @param host_p the number of columns in X
 * @param host_par the parameter data
 */
__global__ void pass_kernel(const float* array_X, const float* array_Y, u32* array_I, const u32 host_n, const u32 host_p, const Parameter host_par, u32* k) {
	// Initialize Random Seed
	curand_init(clock64(), 0, 0, &s);

	// Declare variables
	n = host_n;
	p = host_p;
	par = host_par;
	X = new mat(n, p);
	Y = new vec(n);
	Phi = new vec(par.nP);
	data = new Data[par.nP];
	u32 i, j;
	
	// Copy X and Y from array to matrix
	for(j = 0; j < p; j++) {
		for(i = 0; i < n; i++) {
			X->col[j]->e[i] = array_X[i*p + j];
		}
	}
	for(u32 i = 0; i < n; i++) {
		Y->e[i] = array_Y[i];
	}

	// Set Random Initial
	bool isRandInitial = false;
	
	// Compute Correct Criterion
	Data data_T;
	data_T.stat = init;
	pass_update_cri(&data_T, 0);
	printf("Phi = %8.3f, Index = ", data_T.phi); print(data_T.Index);
	data_T.stat = forw;
	for(u32 ii = 1; ii < 10; ii++) {
		pass_update_cri(&data_T, ii);
		printf("Phi = %8.3f, Index = ", data_T.phi); print(data_T.Index);
		
	}
	printf("Beta = "); print(data_T.Beta);
	// Initialize Particles
	if(isRandInitial || p < par.nP) {
		for(j = 0; j < par.nP; j++) {
			//printf("particle = %d\n", j);
			data[j].stat = init;
			pass_update_cri(&data[j], curand(&s) % p);
			Phi->e[j] = data[j].phi;
			data[j].stat = forw;
		}
	}
	else {
		vec* C = new vec(p);
		idx* I = new idx(p);
		mul(C, Y, X);
		sort_index(I, C);
		for(j = 0; j < par.nP; j++) {
			//printf("particle = %d\n", j);
			data[j].stat = init;
			pass_update_cri(&data[j], I->e[j]);
			Phi->e[j] = data[j].phi;
			data[j].stat = forw;
		}
		delete C;
		delete I;
	}
	
	// Choose Global Best
	data_best_phi = data[0].phi;
	data_best_Index = data[0].Index;

	// Find Best Data
	//for(u32 i = 0; i < par.nI; i++) {
	//	for(u32 j = 0; j < par.nP; j++) {
	//		pass_update_fb(&data[j]);
	//		if(data_best_phi > data[j].phi) {
	//			data_best_phi = data[j].phi;
	//			data_best_Index = data[j].Index;
	//		}
	//		if(data[j].phi - Phi->e[j] > 0) {
	//			data[j].stat = (Stat)(-data[j].stat);
	//		}
	//		if(data[j].Index->n <= 1) {
	//			data[j].stat = forw;
	//		}
	//		if(data[j].Index->n >= p - 5) {
	//			data[j].stat = back;
	//		}
	//		Phi->e[j] = data[j].phi;
	//	}
	//	printf("phi = %f  Index = ", data_best_phi);
	//	print(data_best_Index);
	//}
	
	//*k = data_best_Index->n;
	//for(u32 i = 0; i < *k; i++) {
	//	array_I[i] = data_best_Index->e[i];
	//}

	//delete X;
	//delete Y;
	//delete Phi;
	//delete data;
	
	*k = 0;
}

// PaSS Find Best
__global__ void pass_find_best(u32 const i) {
	for(u32 j = 0; j < par.nP; j++) {
		//printf("particle = %d\n", j);
		pass_update_fb(&data[j]);
		if(data_best_phi > data[j].phi) {
			data_best_phi = data[j].phi;
			data_best_Index = data[j].Index;
		}
		if(data[j].phi - Phi->e[j] > 0) {
			data[j].stat = (Stat)(-data[j].stat);
		}
		if(data[j].Index->n <= 1) {
			data[j].stat = forw;
		}
		if(data[j].Index->n >= p - 5) {
			data[j].stat = back;
		}
		Phi->e[j] = data[j].phi;
	}
	printf("phi = %f  Index = ", data_best_phi);
	print(data_best_Index);
}


/**
 * Determine forward or backward
 */
__device__ bool pass_update_fb(Data* data) {
	u32 index = 0;
	switch(data->stat) {
	case forw: // Forward
		{
			idx* Index_B = new idx(data_best_Index->n);
			idx* Index_C = new idx(data->Index->n);
			idx* Index_D = new idx(Index_B->n);
			idx* Index_R = new idx(p - Index_C->n);

			// Sort Index_B
			copy(Index_B, data_best_Index);
			sort_ascend(Index_B);
			
			// Sort Index_C
			copy(Index_C, data->Index);
			sort_ascend(Index_C);
			
			// Let Index_D be Index_B exclude Index_C
			set_difference(Index_D, Index_B, Index_C);
			
			// Let Index_R be the complement of Index_C
			complement(Index_R, Index_C, p);

			// Determine the index to add
			if(curand_uniform(&s) < par.pfg && Index_D->n > 0){
				index = Index_D->e[curand(&s) % Index_D->n];
			}
			else if(curand_uniform(&s) < par.pfl/(par.pfl+par.pfr)){
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
			if(curand_uniform(&s)< par.pbl){
				mat* B = new mat(data->X->n_row, data->X->n_col);
				vec* C = new vec(data->X->n_col);
				u32 ii;
				for(u32 i = 0; i < B->n_col; i++){
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
			data->X = new mat(n, 1);
			copy(data->X->col[0], Xnew);

			data->InvA = new mat(1, 1);
			float a;
			inner(&a, Xnew);
			data->InvA->col[0]->e[0] = 1 / a;

			data->Theta = new vec(1);
			inner(&(data->Theta->e[0]), Xnew, Y);

			data->Beta = new vec(1);
			mul(data->Beta, data->InvA, data->Theta);

			data->Index = new idx(1);
			data->Index->e[0] = index;

			data->R = new vec(n);

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
