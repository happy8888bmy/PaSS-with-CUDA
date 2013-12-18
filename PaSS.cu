/**
 * PaSS.cu
 * The main functions of PaSS
 *
 * @author emfo
 */

#include "PaSS_BLAS.cu"
using namespace pass;

#include <algorithm>
using namespace std;

/**
 * The status type
 *  0: Initial
 *  1: Forward
 * -1: Backward
*/
typedef int Stat;


/**
 * The criterion enumeration
 */
enum Criterion {
	AIC,
	BIC,
	EBIC,
	HDBIC,
	HDHQ
};


/**
 * The parameter structure
 */
struct Parameter {
	u32 nP;    // the number of particle
	u32 nI;    // the number of iteration
	float pfg; // the probability for forward step: global
	float pfl; // the probability for forward step: local
	float pfr; // the probability for forward step: random
	float pbl; // the probability for backward step: local
	float pbr; // the probability for backward step: random
};


/**
 * The data structure
 */
struct Data {
	vec* Beta;
	float e;
	idx* Index; // the index of chosen column of X
	mat* InvA;
	float phi;  // the value given by criterion
	vec* R;
	Stat stat;  // the status
	vec* Theta;
	mat* X;     // the data we chosen
};


/**
 * Global variables
 */
__device__ u32 n, p;
__device__ mat* X;
__device__ vec* Y;
__device__ Criterion cri = HDBIC;
__device__ Parameter par = {32, 128, .8, .1, .1, .9, .1};
__device__ Data data_best;
__device__ curandState s;



void pass_init(float*, float*, const u32, const u32);
cudaError_t pass_host(const float*, const float*, u32*, const u32, const u32);
__global__ void pass_kernel(const float*, const float*, u32*, const u32, const u32);
__device__ bool pass_update_fb(Data*);
__device__ bool pass_update_cri(Data*, const u32);


/**
 * PaSS main function
 */
int main() {
	// Declare variables
	u32 host_n = 100;
	u32 host_p = 20;
	float* host_X = (float*)malloc(host_n * host_p * sizeof(float));
	float* host_Y = (float*)malloc(host_n * sizeof(float));
	u32 *host_I = (u32*)malloc(host_p * sizeof(u32));
	
	// Initialize data
	pass_init(host_X, host_Y, host_n, host_p);

	//// Display data
	//u32 i, j;
	//printf("X:\n");
	//for(i = 0; i < host_n; i++) {
	//	for(j = 0; j < host_p; j++) {
	//		printf("%8.3f  ", host_X[i*host_p + j]);
	//	}
	//	printf("\n");
	//}
	//printf("\n\nY:\n");
	//for(i = 0; i < host_n; i++) {
	//	printf("%8.3f  ", host_Y[i]);
	//}
	//printf("\n\n");
	
	// Run PaSS
	pass_host(host_X, host_Y, host_I, host_n, host_p);
	
	system("pause");
	return 0;
}


/**
 * Initialize data
 */
void pass_init(float* host_X, float* host_Y, const u32 host_n, const u32 host_p) {
	u32 i, j;
	for(i = 0; i < host_n; i++) {
		for(j = 0; j < host_p; j++) {
			host_X[i*host_p + j] = (i+1)*(2*j+1);
		}
		host_Y[i] = 10*i;
	}
}


/**
 * PaSS host function
 */
cudaError_t pass_host(const float* host_X, const float* host_Y, u32* host_I, const u32 host_n, const u32 host_p) {
	// Declare variables
	float* dev_X = 0;
	float* dev_Y = 0;
	u32 *dev_I = 0;
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
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Y, host_n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_I, host_n * sizeof(u32));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_X, host_X, host_n * host_p * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Y, host_Y, host_n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch the kernel function on the GPU with one thread for each element.
	pass_kernel<<<1, 1>>>(dev_X, dev_Y, dev_I, host_n, host_p);

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
	cudaStatus = cudaMemcpy(host_I, dev_I, host_p * sizeof(u32), cudaMemcpyDeviceToHost);
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


/**
 * PaSS kernel function
 */
__global__ void pass_kernel(const float* array_X, const float* array_Y, u32* array_I, const u32 host_n, const u32 host_p) {
	// Initialize Random Seed
	curand_init(clock64(), 0, 0, &s);

	// Declare variables
	n = host_n;
	p = host_p;
	X = new mat(n, p);
	Y = new vec(n);
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

	u32 index;

	// Initialize Particles
	Data* data = new Data[par.nP];
	mat* Phi = new mat(par.nI+1, par.nP);
	if(isRandInitial || p < par.nP) {
		for(j = 0; j < par.nP; j++) {
			data[j].stat = 0;
			pass_update_cri(&data[j], curand(&s) % p);
			Phi->col[j]->e[0] = data[j].phi;
			data[j].stat = 1;
		}
	}
	else {
		vec* C = new vec(p);
		idx* I = new idx(p);
		for(j = 0; j < par.nP; j++) {
			mul(C, Y, X);
			sort_index_descend(I, C);
			data[j].stat = 0;
			pass_update_cri(&(data[j]), I->e[j]);
			Phi->col[j]->e[0] = data[j].phi;
			data[j].stat = 1;
		}
		delete C;
		delete I;
	}
	
	// Choose Global Best
	data_best = data[0];

	// Find Best Data
	for(i = 0; i < par.nI; i++) {
		for(j = 0; j < par.nP; j++) {
			pass_update_fb(&(data[j]));
			Phi->col[j]->e[i+1] = data[j].phi;
			if(data_best.phi > data[j].phi) {
				data_best = data[j];
			}
			if(data[j].phi - Phi->col[j]->e[i] > 0) {
				data[j].stat = -data[j].stat;
			}
			if(data[j].Index->n <= 1) {
				data[j].stat = 1;
			}
			if(data[j].Index->n >= p - 5) {
				data[j].stat = -1;
			}
		}
	}
	
	delete X;
	delete Y;
}


/**
 * Determine forward or backward
 *
 * @param data the updating data
 */
__device__ bool pass_update_fb(Data* data) {
	u32 index = 0;

	switch(data->stat) {
	case 1: // Forward
		{
			idx* Index_B = new idx(data_best.Index->n);
			idx* Index_C = new idx(data->Index->n);
			
			// Determined the length of Index_B, C and initialize the length of Index_D
			u32 length_D_max = Index_B->n + Index_C->n;
			u32 length_D = 0;
			
			// Let Index_D be Index_B Exclude Index_C
			memcpy(Index_B->e, data_best.Index, (sizeof(u32)) * Index_B->n);
			if(~sort_ascend(Index_B)) return false;
			
			memcpy(Index_C->e, data->Index, (sizeof(u32)) * Index_C->n);
			if(~sort_ascend(Index_C)) return false;
			
			u32* Index_D_temp = (u32*)malloc(sizeof(u32) * length_D_max);

			u32 i = 0, j = 0;
			while(i < Index_B->n){
				if(Index_B->e[i] < Index_C->e[j] || j == Index_C->n - 1){
					Index_D_temp[length_D] = Index_B->e[i];
					length_D = length_D + 1;
					i++;
				}
				else if(Index_B->e[i] == Index_C->e[j]){
					i++;
					j++;
				}
				else{
					j++;
				}
			}

			idx* Index_D = new idx(length_D);
			memcpy(Index_D->e, Index_D_temp, (sizeof(u32)) * Index_D->n);

			// Let Index_R be the Complement of Index_C
			idx* Index_R = new idx(p - Index_C->n);
			u32 num = 0;

			for(i = 0, j = 0; i < Index_R->n; num++){
				if(j < Index_C->n && num == Index_C->e[j]){
					j++;
				}
				else{
					Index_R->e[i] = num;
					i++;
				}
			}

			// Determine the index to add
			float randnum = curand_uniform(&s);

			if((float)randnum < par.pfg && length_D > 0){
				index = Index_D->e[curand(&s) % Index_D->n];
			}
			else if(randnum < par.pfl + par.pfg){
				float phi_max  = 0, phi_temp = 0;
				if(~inner(&phi_max, data->R, X->col[Index_R->e[0]])) return false;
				for(i = 0; i < Index_R->n; i++){
					if(~inner(&phi_temp, data->R, X->col[Index_R->e[i]])) return false;
					if(phi_temp > phi_max){
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
	case -1: // Backward
		{
			float randnum = curand_uniform(&s);

			if(randnum < par.pbl){
				mat* temp1 = new mat(n, p);
				// temp1 = (*data).X * diagmat((*data).Beta)
				for(int i = 0; i < temp1->n_col; i++){
						if(~mul(temp1->col[i], data->X->col[i], data->Beta->e[i]))return false;
				}

				mat* temp2 = new mat(n, p);
				// temp2 = (*data).R * ones<rowvec>((*data).Index.n_rows)
				for(int i = 0; i < temp2->n_col; i++){
					if(~mul(temp2->col[i], data->R, 1))return false;
				}

				mat* bb = new mat(n, p);
				if(~add(bb, temp1, temp2))return false;
				
				delete temp1;
				delete temp2;

				// Find arcmin of rowvec bbbb = sum(bb%bb)
				float min_temp, temp;
				if(~norm(&min_temp, bb->col[0]))return false;
				u32 cc = 0;

				for(int i = 0; i < bb->n_col; i++){
					if(~norm(&temp, bb->col[i]))return false;
					if(temp > min_temp){
						min_temp = temp;
						cc = i;
					}
				}
				
				delete bb;

				index = data->Index->e[cc];
				
				
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
	switch((*data).stat) {
	case 0: // Initial
		{
			(*data).X = new mat(n, 1);
			copy((*data).X->col[0], Xnew);

			(*data).InvA = new mat(1, 1);
			float a;
			inner(&a, Xnew);
			(*data).InvA->col[0]->e[0] = 1 / a;

			(*data).Theta = new vec(1);
			inner((*data).Theta->e, Xnew, Y);

			(*data).Beta = new vec(1);
			mul((*data).Beta, (*data).InvA, (*data).Theta);

			(*data).Index = new idx(1);
			(*data).Index ->e[0] = index;

			(*data).R = new vec(n);

			k = 1;
		}
		break;
	case 1: // Forward
		{

			k = (*data).Index->n;
			vec* B = new vec(k);
			mul(B, Xnew, (*data).X);

			vec* D = new vec(k);
			mul(D, (*data).InvA, B);
			
			float c1;
			float c2;
			inner(&c1, Xnew);
			inner(&c2, B, D);
			float alpha = 1/(c1 - c2);

			insert(D, -1);

			insert_col((*data).X, Xnew);

			mat* InvAtemp = new mat(k+1, k+1);
			mul(InvAtemp, D, D);
			mul(InvAtemp, InvAtemp, alpha);
			insert((*data).InvA, 0);
			add((*data).InvA, (*data).InvA, InvAtemp);
			
			inner(&c1, Xnew, Y);
			insert((*data).Theta, c1);
			
			inner(&c2, D, (*data).Theta);
			mul(D, D, alpha*c2);
			insert((*data).Beta, 1);
			add((*data).Beta, (*data).Beta, D);

			insert((*data).Index, index);
			
			delete B;
			delete D;
			delete InvAtemp;
			k++;
		}
		break;
	case -1: // Backward
		{
			k = (*data).Index->n - 1;
			u32 ii;
			find_index(&ii, (*data).Index, index);
			if(ii != k) {
				swap_col((*data).X, ii, k);
				swap((*data).Theta, ii, k);
				swap((*data).Beta, ii, k);
				swap_row((*data).InvA, ii, k);
				swap_col((*data).InvA, ii, k);
				swap((*data).Index, ii, k);
			}

			shed_col((*data).X);
			shed((*data).Theta);
			shed((*data).Index);

			float g = (*data).InvA->col[k]->e[k];

			vec* F = new vec(k);
			shed_row((*data).InvA);
			copy(F, (*data).InvA->col[k]);
			shed_col((*data).InvA);

			mat* E = new mat(k, k);
			mul(E, F, F);
			mul(E, E, 1/g);
			sub((*data).InvA, (*data).InvA, E);

			
			mul(F, F, (*data).Beta->e[k] / g);
			shed((*data).Beta);
			add((*data).Beta, (*data).Beta, F);

			delete E;
			delete F;
		}
		break;
	}

	mul((*data).R, (*data).X,  (*data).Beta);
	sub((*data).R, Y, (*data).R);

	norm(&(*data).e, (*data).R);

	switch(cri) {
	case AIC:
		(*data).phi = n * log((*data).e * (*data).e / n) + 2 * k;
		break;
	case BIC:
		(*data).phi = n * log((*data).e * (*data).e / n) + log((float)n) * k;
		break;
	case EBIC:
		(*data).phi = n * log((*data).e * (*data).e / n) + log((float)n) * k + 2 * gamma * ((p+.5) * log((float)p) - (k+.5) * log((float)k) - (p-k+.5) * log((float)(p-k)) - .5 * log(2 * CUDART_PI_F));
		break;
	case HDBIC:
		(*data).phi = n * log((*data).e * (*data).e / n) + log((float)n) * log((float)p) * k;
		break;
	case HDHQ:
		(*data).phi = n * log((*data).e * (*data).e / n) + 2.01 * log(log((float)n)) * log((float)p) * k;
		break;
	}
	return true;
}
