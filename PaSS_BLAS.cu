/**
 * PaSS_BLAS.cu <br>
 * The basic linear algebra sub-programs for PaSS
 *
 * @author Mu Yang
 * @author Da-Wei Chang
 * @author Chen-Yao Lin
 * @date 2014.01.18 12:39
 * @version 1.0
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "curand_kernel.h"
#include "device_functions.h"
#include "float.h"
#include "math_constants.h"
#include "math_functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/**
 * The rule of defines:<br>
 *	n_    : the current length of a vector or matrix <br>
 *	m_    : the maximal length of a vector or matrix <br>
 *	size_ : the number of total elements (which contains the memory that record the information of vector or matrix) <br>
 *
 */

#define n_ents(a) \
	(a)[0]
#define m_ents(a) \
	(a)[1]
#define size_vec(a) \
	(m_ents(a)+2)

#define entry(a, i) \
	(a)[(i)+2]

#define n_cols(a) \
	(a)[0]
#define m_cols(a) \
	(a)[1]
#define n_rows(a) \
	(a)[2]
#define m_rows(a) \
	(a)[3]
#define size_mat(a) \
	((m_rows(a)+2)*m_cols(a)+2)

#define col(a, j) \
	((a)+(j)*((u32)m_rows(a)+2)+2)
#define entry2(a, i, j) \
	(a)[(j)*((u32)m_rows(a)+2)+i+4]


/**
 * The PaSS_BLAS namespace
 */
namespace pass_blas {
	/**
	 * 32-bit unsigned integer type.
	 */
	typedef uint32_t u32;
	/**
	 * vector type. <br>
	 * | number of entries | max number of entries  | entries |
	 */
	typedef float vec;
	/**
	 * u32 vector type. <br>
	 * | number of entries | max number of entries  | entries |
	 */
	typedef u32 uvec;
	/**
	 * matrix type. <br>
	 * | number of columns | max number of columns | column 0 (vec) | column 1 (vec) | ... |
	 */
	typedef float mat;

	/**
	 * Copy a vector.
	 *
	 * @param u the new vector.
	 * @param v the original vector.
	 */
	__host__ __device__ void copy_vec(vec* u, const vec* v) {
		memcpy(u, v, size_vec(v) * sizeof(vec));
	}
	

	/**
	 * Copy a u32 vector.
	 *
	 * @param z the new index.
	 * @param x the original index.
	 */
	__host__ __device__ void copy_uvec(uvec* z, const uvec* x) {
		memcpy(z, x, size_vec(x) * sizeof(uvec));
	}


	/**
	 * Copy a matrix.
	 *
	 * @param c the new matrix.
	 * @param a the original matrix.
	 */
	__host__ __device__ void copy_mat(mat* c, const mat* a) {
		memcpy(c, a, size_mat(a) * sizeof(mat));
	}


	/**
	 * Display the vector.
	 *
	 * @param v the vector.
	 */
	__host__ __device__ void print_vec(const vec* v) {
		u32 i;
		for(i = 0; i < n_ents(v); i++) {
			printf("%8.3f  ", entry(v, i));
		}
		printf("\n");
	}


	/**
	 * Display the u32 vector.
	 *
	 * @param x the u32 vector.
	 */
	__host__ __device__ void print_uvec(const uvec* x) {
		u32 i;
		for(i = 0; i < n_ents(x); i++) {
			printf("%4u ", entry(x, i));
		}
		printf("\n");
	}


	/**
	 * Display the matrix.
	 *
	 * @param a the vector.
	 */
	__host__ __device__ void print_mat(const mat* a) {
		u32 i, j;
		for(i = 0; i < n_rows(a); i++) {
			for(j = 0; j < n_cols(a); j++) {
				printf("%8.3f  ", entry2(a, i, j));
			}
			printf("\n");
		}
		printf("\n");
	}


	/**
	 * u = v+w.
	 *
	 * @param u the sum vector.
	 * @param v the augend vector.
	 * @param w the addend vector.
	 */
	__host__ void add_vec_host(vec* u, const vec* v, const vec* w) {
		//if(m_ents(u) != m_ents(v) || m_ents(u) != m_ents(w)) {
		//	printf("Warning: add_vec overflowed!\n");
		//}
		//if(n_ents(v) != n_ents(w)) {
		//	printf("(add_vec) not aligned!\n");
		//}
		u32 i;
		n_ents(u) = n_ents(v);
		for(i = 0; i < n_ents(v); i++) {
			entry(u, i) = entry(v, i) + entry(w, i);
		}
	}
	
	__global__ void add_vec_child(vec* u, const vec* v, const vec* w) {
		entry(u, threadIdx.x) = entry(v, threadIdx.x) + entry(w, threadIdx.x);
	}
	
	__device__ inline void add_vec(vec* u, const vec* v, const vec* w) {
		n_ents(u) = n_ents(v);
		add_vec_child<<<1, n_ents(u)>>>(u, v, w);
		cudaDeviceSynchronize();
	}


	/**
	 * c = a+b.
	 *
	 * @param c the sum matrix.
	 * @param a the augend matrix.
	 * @param b the addend matrix.
	 */
	__host__ void add_mat_host(mat* c, const mat* a, const mat* b) {
		//if(m_rows(c) != m_rows(a) || m_rows(c) != m_rows(b) || m_cols(c) != m_cols(a) || m_cols(c) != m_cols(b)) {
		//	printf("Warning: add_mat overflowed!\n");
		//}
		//if(n_rows(a) != n_rows(b) || n_cols(a) != n_cols(b)) {
		//	printf("(add_mat) not aligned!\n");
		//}
		u32 i, j;
		n_cols(c) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			n_ents(col(c, j)) = n_rows(a);
			m_ents(col(c, j)) = m_rows(c);
			for(i = 0; i < n_rows(a); i++) {
				entry2(c, i, j) = entry2(a, i, j) + entry2(b, i, j);
			}
		}
	}
	
 	__global__ void add_mat_child1(mat* c, const mat* a) {
		n_ents(col(c, threadIdx.x)) = n_rows(a);
		m_ents(col(c, threadIdx.x)) = m_rows(c);
	}
	
	__global__ void add_mat_child2(mat* c, const mat* a, const mat* b) {
		entry2(c, blockIdx.x, threadIdx.x) = entry2(a, blockIdx.x, threadIdx.x) + entry2(b, blockIdx.x, threadIdx.x);
	}
	
	__device__ inline void add_mat(mat* c, const mat* a, const mat* b) {
		n_cols(c) = n_cols(a);
		add_mat_child1<<<1, n_cols(a)>>>(c, a);
		add_mat_child2<<<n_rows(a), n_cols(a)>>>(c, a, b);
		cudaDeviceSynchronize();
	}


	/**
	 * u = v-w.
	 *
	 * @param u the difference vector.
	 * @param v the minuend vector.
	 * @param w the subtrahend vector.
	 */
	__host__ void sub_vec_host(vec* u, const vec* v, const vec* w) {
		//if(m_ents(u) != m_ents(v) || m_ents(u) != m_ents(w)) {
		//	printf("Warning: sub_vec overflowed!\n");
		//};
		//if(n_ents(v) != n_ents(w)) {
		//	printf("(sub_vec) not aligned!\n");
		//}
		u32 i;
		n_ents(u) = n_ents(v);
		for(i = 0; i < n_ents(v); i++) {
			entry(u, i) = entry(v, i) - entry(w, i);
		}
	}
	
	__global__ void sub_vec_child(vec* u, const vec* v, const vec* w) {
		entry(u, threadIdx.x) = entry(v, threadIdx.x) - entry(w, threadIdx.x);
	}
	
	__device__ inline void sub_vec(vec* u, const vec* v, const vec* w) {
		n_ents(u) = n_ents(v);
		sub_vec_child<<<1, n_ents(u)>>>(u, v, w);
		cudaDeviceSynchronize();
	}


	/**
	 * c = a-b.
	 *
	 * @param c the difference matrix.
	 * @param a the minuend matrix.
	 * @param b the subtrahend matrix.
	 */
	__host__ void sub_mat_host(mat* c, const mat* a, const mat* b) {
		//if(m_rows(c) != m_rows(a) || m_rows(c) != m_rows(b) || m_cols(c) != m_cols(a) || m_cols(c) != m_cols(b)) {
		//	printf("Warning: sub_mat overflowed!\n");
		//}
		//if(n_cols(a) != n_cols(b) || n_rows(a) != n_rows(b)) {
		//	printf("(sub_mat) not aligned!\n");
		//}
		u32 i, j;
		n_cols(c) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			n_ents(col(c, j)) = n_rows(a);
			m_ents(col(c, j)) = m_rows(c);
			for(i = 0; i < n_rows(a); i++) {
				entry2(c, i, j) = entry2(a, i, j) - entry2(b, i, j);
			}
		}
	}
	
 	__global__ void sub_mat_child1(mat* c, const mat* a) {
		n_ents(col(c, threadIdx.x)) = n_rows(a);
		m_ents(col(c, threadIdx.x)) = m_rows(c);
	}
	
	__global__ void sub_mat_child2(mat* c, const mat* a, const mat* b) {
		entry2(c, blockIdx.x, threadIdx.x) = entry2(a, blockIdx.x, threadIdx.x) - entry2(b, blockIdx.x, threadIdx.x);
	}
	
	__device__ inline void sub_mat(mat* c, const mat* a, const mat* b) {
		n_cols(c) = n_cols(a);
		sub_mat_child1<<<1, n_cols(a)>>>(c, a);
		sub_mat_child2<<<n_rows(a), n_cols(a)>>>(c, a, b);
		cudaDeviceSynchronize();
	}


	/**
	 * u = f*v.
	 *
	 * @param u the product vector.
	 * @param v the multiplier vector.
	 * @param f the multiplicand number.
	 */
	__host__ void mul_vec_host(vec* u, const vec* v, float f) {
		//if(m_ents(u) != m_ents(v)) {
		//	printf("Warning: mul_vec overflowed!\n");
		//}
		u32 i;
		n_ents(u) = n_ents(v);
		for(i = 0; i < n_ents(v); i++) {
			entry(u, i) = entry(v, i) * f;
		}
	}
	
	__global__ void mul_vec_child(vec* u, const vec* v, float f) {
		entry(u, threadIdx.x) = entry(v, threadIdx.x) * f;
	}
	
	__device__ inline void mul_vec(vec* u, const vec* v, float f) {
		n_ents(u) = n_ents(v);
		mul_vec_child<<<1, n_ents(u)>>>(u, v, f);
		cudaDeviceSynchronize();
	}


	/**
	 * c = f*a.
	 *
	 * @param c the product matrix.
	 * @param a the multiplier vector.
	 * @param f the multiplicand number.
	 */
	__host__ void mul_mat_host(mat* c, const mat* a, float f) {
		//if(m_rows(c) != m_rows(a) || m_cols(c) != m_cols(a)) {
		//	printf("Warning: mul_mat overflowed!\n");
		//}
		u32 i, j;
		n_cols(c) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			n_ents(col(c, j)) = n_rows(a);
			m_ents(col(c, j)) = m_rows(c);
			for(i = 0; i < n_rows(a); i++) {
				entry2(c, i, j) = entry2(a, i, j) * f;
			}
		}
	}
	
 	__global__ void mul_mat_child1(mat* c, const mat* a) {
		n_ents(col(c, threadIdx.x)) = n_rows(a);
		m_ents(col(c, threadIdx.x)) = m_rows(c);
	}
	
	__global__ void mul_mat_child2(mat* c, const mat* a, float f) {
		entry2(c, blockIdx.x, threadIdx.x) = entry2(a, blockIdx.x, threadIdx.x) * f;
	}
	
	__device__ inline void mul_mat(mat* c, const mat* a, float f) {
		n_cols(c) = n_cols(a);
		mul_mat_child1<<<1, n_cols(a)>>>(c, a);
		mul_mat_child2<<<n_rows(a), n_cols(a)>>>(c, a, f);
		cudaDeviceSynchronize();
	}


	/**
	 * u = a*v <br>
	 *
	 * @note This function does not allow that u and v are the same pointer.
	 * @param u the product vector.
	 * @param a the multiplicand matrix.
	 * @param v the multiplier vector.
	 */
	__host__ __device__ void mul_matvec(vec* u, const mat* a, const vec* v) {
		//if(m_ents(u) != m_rows(a)) {
		//	printf("Warning: mul_matvec overflowed!");
		//}
		//if(n_ents(v) != n_cols(a)) {
		//	printf("(mul_matvec) not aligned!\n");
		//}
		u32 i, j;
		n_ents(u) = n_rows(a);
		for(i = 0; i < n_rows(a); i++) {
			entry(u, i) = 0;
			for(j = 0; j < n_cols(a); j++) {
				entry(u, i) += entry2(a, i, j) * entry(v, j);
			}
		}
	}


	/**
	 * u' = v'*a (u = a'*v)
	 *
	 * @note This function does not allow that u and v are the same pointer.
	 * @param u the product vector.
	 * @param v the multiplicand vector.
	 * @param a the multiplier matrix.
	 */
	__host__ __device__ void mul_vecmat(vec* u, const vec* v, const mat* a) {
		//if(m_ents(u) != m_cols(a)) {
		//	printf("Warning: mul_vecmat overflowed!\n");
		//}
		//if(n_ents(v) != n_rows(a)) {
		//	printf("(mul_vecmat) not aligned!\n");
		//}
		u32 i, j;
		n_ents(u) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			entry(u, j) = 0;
			for(i = 0; i < n_rows(a); i++) {
				entry(u, j) += entry2(a, i, j) * entry(v, i);
			}
		}
	}


	/**
	 * c = v*w'
	 *
	 * @param c the product matrix.
	 * @param v the multiplicand vector.
	 * @param w the multiplier vector.
	 */
	__host__ __device__ void mul_vecvec(mat* c, const vec* v, const vec* w) {
		//if(m_rows(c) != m_ents(v) || m_cols(c) != m_ents(w)) {
		//	printf("Warning: mul_vecvec overflowed!\n");
		//}
		u32 i, j;
		n_rows(c) = n_ents(v);
		n_cols(c) = n_ents(w);
		for(j = 0; j < n_cols(c); j++) {
			n_ents(col(c, j)) = n_ents(v);
			m_ents(col(c, j)) = m_rows(c);
			for(i = 0; i < n_rows(c); i++) {
				entry2(c, i, j) = entry(v, i) * entry(w, j);
			}
		}
	}


	/**
	 * f = sum(v.*v).
	 *
	 * @param f the product number.
	 * @param v the vector.
	 */
	__host__ __device__ void inner_vec(float* f, const vec* v) {
		u32 i;
		*f = 0;
		for(i = 0; i < n_ents(v); i++) {
			*f += entry(v, i) * entry(v, i);
		}
	}


	/**
	 * f = sum(v.*w).
	 *
	 * @param v the multiplicand vector.
	 * @param w the multiplier vector.
	 * @param f the product number.
	 */
	__host__ __device__ void inner_vec(float* f, const vec* v, const vec* w) {
		//if(n_ents(v) != n_ents(w)) {
		//	printf("(inner_vec) not aligned!\n");
		//}
		u32 i;
		*f = 0;
		for(i = 0; i < n_ents(v); i++) {
			*f += entry(v, i) * entry(w, i);
		}
	}


	/**
	 * u[i] = sum(a_ith_col.*a_ith_col).
	 *
	 * @param u the product vector.
	 * @param a the matrix.
	 */
	__host__ void inner_mat_host(vec* u, const mat* a) {
		//if(m_ents(u) != m_cols(a)) {
		//	printf("Warning: inner_mat (ver1) overflowed!\n");
		//}
		u32 i, j;
		n_ents(u) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			entry(u, j) = 0;
			for(i = 0; i < n_rows(a); i++) {
				entry(u, j) += entry2(a, i, j) * entry2(a, i, j);
			}
		}
	}
	
	__global__ void inner_mat_child(vec* u, const mat* a) {
		u32 i;
		entry(u, threadIdx.x) = 0;
		for(i = 0; i < n_rows(a); i++) {
			entry(u, threadIdx.x) += entry2(a, i, threadIdx.x) * entry2(a, i, threadIdx.x);
		}
	}
	
	__device__ inline void inner_mat(vec* u, const mat* a) {
		n_ents(u) = n_cols(a);
		inner_mat_child<<<1, n_cols(a)>>>(u, a);
		cudaDeviceSynchronize();
	}


	/**
	 * u[i] = sum(a_ith_col.*b_ith_col).
	 *
	 * @param u the product vector.
	 * @param a the multiplicand matrix.
	 * @param b the multiplier matrix.
	 */
	__host__ void inner_mat_host(vec* u, const mat* a, const mat* b) {
		//if(m_ents(u) != m_cols(a) || m_ents(u) != m_cols(b)) {
		//	printf("Warning: inner_mat (ver2) overflowed!\n");
		//}
		//if(n_rows(a) != n_rows(b) || n_cols(a) != n_cols(b)) {
		//	printf("(inner_mat) not aligned!\n");
		//}
		u32 i, j;
		n_ents(u) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			entry(u, j) = 0;
			for(i = 0; i < n_rows(a); i++) {
				entry(u, j) += entry2(a, i, j) * entry2(b, i, j);
			}
		}
	}
	
	__global__ void inner_mat_child(vec* u, const mat* a, const mat* b) {
		u32 i;
		entry(u, threadIdx.x) = 0;
		for(i = 0; i < n_rows(a); i++) {
			entry(u, threadIdx.x) += entry2(a, i, threadIdx.x) * entry2(b, i, threadIdx.x);
		}
	}
	
	__device__ inline void inner_mat(vec* u, const mat* a, const mat* b) {
		n_ents(u) = n_cols(a);
		inner_mat_child<<<1, n_cols(a)>>>(u, a, b);
		cudaDeviceSynchronize();
	}


	/**
	 * f = norm(v, 2).
	 *
	 * @param f the Euclidean norm.
	 * @param v the vector.
	 */
	__host__ __device__ void norm_vec(float* f, vec* v) {
		inner_vec(f, v);
		*f = sqrt(*f);
	}


	/**
	 * Add a new entry f at the end of a vector.
	 *
	 * @param v the vector.
	 * @param f the new entry.
	 */
	__host__ __device__ void insert_vec(vec* v, float f) {
		entry(v, (u32)n_ents(v)) = f;
		n_ents(v)++;
	}


	/**
	 * Add a new entry f at the end of a uvec.
	 *
	 * @param x the index.
	 * @param i the new entry.
	 */
	__host__ __device__ void insert_uvec(uvec* x, u32 i) {
		entry(x, n_ents(x)) = i;
		n_ents(x)++;
	}


	/**
	 * Add a new row and a new column filled with f as the last row and column of a matrix.
	 *
	 * @param a the matrix.
	 * @param f the number.
	 */
	__host__ void insert_mat_host(mat* a, float f) {
		u32 i, nr = n_rows(a)+1;
		n_cols(a)++;
		for(i = 0; i < n_rows(a); i++) {
			entry2(a, i, (u32)n_cols(a)) = f;
		}
		for(i = 0; i < n_cols(a); i++) {
			entry2(a, (u32)n_rows(a), i) = f;
			n_ents(col(a, i)) = nr;
		}
		m_ents(col(a, (u32)m_rows(a))) = m_rows(a);
	}
	
	__global__ void insert_mat_child1(mat* a, float f) {
		entry2(a, threadIdx.x, (u32)n_cols(a)) = f;
	}
	
	__global__ void insert_mat_child2(mat* a, float f, u32 nr) {
		entry2(a, (u32)n_rows(a), threadIdx.x) = f;
		n_ents(col(a, threadIdx.x)) = nr;
	}
	
	__device__ inline void insert_mat(mat* a, float f) {
		u32 nr = n_rows(a)+1;
		n_cols(a)++;
		insert_mat_child1<<<1, n_rows(a)>>>(a, f);
		insert_mat_child2<<<1, n_cols(a)>>>(a, f, nr);
		m_ents(col(a, (u32)m_rows(a))) = m_rows(a);
		cudaDeviceSynchronize();
	}


	/**
	 * Let a vector be the new last row of a matrix.
	 *
	 * @param a the matrix.
	 * @param v the new vector.
	 */
	__host__ void insert_row_host(mat* a, vec* v) {
		//if(n_cols(a) != n_ents(v)) {
		//	printf("(insert_row) not aligned!\n");
		//}
		u32 j;
		for(j = 0; j < n_cols(a); j++) {
			n_ents(col(a, j))++;
			entry2(a, (u32)n_rows(a), j) = entry(v, j);
		}
	}
	
	__global__ void insert_row_child(mat* a, vec* v) {
		n_ents(col(a, threadIdx.x))++;
		entry2(a, (u32)n_rows(a), threadIdx.x) = entry(v, threadIdx.x);
	}
	
	__device__ inline void insert_row(mat* a, vec* v) {
		inner_mat_child<<<1, n_cols(a)>>>(a, v);
		cudaDeviceSynchronize();
	}


	/**
	 * Let a vector be the new last column of a matrix.
	 *
	 * @param a the matrix.
	 * @param v the new vector.
	 */
	__host__ __device__ void insert_col(mat* a, vec* v) {
		//if(n_rows(a) != n_ents(v)) {
		//	printf("(insert_col) not aligned!\n");
		//}
		copy_vec(col(a, (u32)n_cols(a)), v);
		n_cols(a)++;
	}


	/**
	 * Remove the last entry.
	 *
	 * @param v the vector.
	 */
	__host__ __device__ void shed_vec(vec* v) {
		//if(n_ents(v) == 0) {
		//	printf("(shed: vector) empty!\n");
		//}
		n_ents(v)--;
	}


	/**
	 * Remove the last entry.
	 *
	 * @param x the index.
	 */
	__host__ __device__ void shed_uvec(uvec* x) {
		//if(n_ents(x) == 0) {
		//	printf("(shed: index) empty!\n");
		//}
		n_ents(x)--;
	}


	/**
	 * Remove the last n entries.
	 *
	 * @param x the index.
	 * @param n number of entries.
	 */
	__host__ __device__ void shed_uvec(uvec* x, u32 n) {
		//if(n_ents(x) < n) {
		//	printf("(shed: index) n too large!\n");
		//}
		n_ents(x) -= n;
	}


	/**
	 * Remove the last row.
	 *
	 * @param a the matrix.
	 */
	__host__ void shed_row_host(mat* a) {
		//if(n_rows(a) == 0) {
		//	printf("(shed_row) empty!\n");
		//}
		u32 j;
		for(j = 0; j < n_cols(a); j++) {
			n_ents(col(a, j))--;
		}
	}
	
	__global__ void shed_row_child(mat* a) {
		n_ents(col(a, threadIdx.x))--;
	}
	
	__device__ inline void shed_row(mat* a) {
		shed_row_child<<<1, n_cols(a)>>>(a);
		cudaDeviceSynchronize();
	}


	/**
	 * Remove the last column.
	 *
	 * @param a the matrix.
	 */
	__host__ __device__ void shed_col(mat* a) {
		//if(n_cols(a) == 0) {
		//	printf("(shed_col) empty!\n");
		//}
		n_cols(a)--;
	}


	/**
	 * Swap two entries.
	 *
	 * @param v the vector.
	 * @param i the index of first entry.
	 * @param j the index of second entry.
	 */
	__host__ __device__ void swap_vec(vec* v, const u32 i, const u32 j) {
		float temp = entry(v, i);
		entry(v, i) = entry(v, j);
		entry(v, j) = temp;
	}


	/**
	 * Swap two entries.
	 *
	 * @param x the index.
	 * @param i the index of first entry.
	 * @param j the index of second entry.
	 */
	__host__ __device__ void swap_uvec(uvec* x, const u32 i, const u32 j) {
		u32 temp = entry(x, i);
		entry(x, i) = entry(x, j);
		entry(x, j) = temp;
	}


	/**
	 * Swap two rows.
	 *
	 * @param a the matrix.
	 * @param i the index of first row.
	 * @param j the index of second row.
	 */
	__host__ void swap_row_host(mat* a, const u32 i, const u32 j) {
		u32 k;
		float temp;
		for(k = 0; k < n_cols(a); k++) {
			temp = entry2(a, i, k);
			entry2(a, i, k) = entry2(a, j, k);
			entry2(a, j, k) = temp;
		}
	}
	
	__global__ void swap_row_child(mat* a, const u32 i, const u32 j) {
		float temp;
		temp = entry2(a, i, threadIdx.x);
		entry2(a, i, threadIdx.x) = entry2(a, j, threadIdx.x);
		entry2(a, j, threadIdx.x) = temp;
	}
	
	__device__ inline void swap_row(mat* a, const u32 i, const u32 j) {
		swap_row_child<<<1, n_cols(a)>>>(a, i, j);
		cudaDeviceSynchronize();
	}


	/**
	 * Swap two columns.
	 *
	 * @param a the matrix.
	 * @param i the index of first column.
	 * @param j the index of second column.
	 */
	__host__ void swap_col_host(mat* a, const u32 i, const u32 j) {
		u32 k;
		float temp;
		for(k = 0; k < n_rows(a); k++) {
			temp = entry2(a, k, i);
			entry2(a, k, i) = entry2(a, k, j);
			entry2(a, k, j) = temp;
		}
	}
	
	__global__ void swap_col_child(mat* a, const u32 i, const u32 j) {
		float temp;
		temp = entry2(a, threadIdx.x, i);
		entry2(a, threadIdx.x, i) = entry2(a, threadIdx.x, j);
		entry2(a, threadIdx.x, j) = temp;
	}
	
	__device__ inline void swap_col(mat* a, const u32 i, const u32 j) {
		swap_col_child<<<1, n_rows(a)>>>(a, i, j);
		cudaDeviceSynchronize();
	}


	/**
	 * Find the index of target element (avaliable only for uvec)
	 *
	 * @param k the index.
	 * @param x the vector.
	 * @param i the element.
	 */
	__host__ __device__ void find_index(u32* k, const uvec* x, const u32 i) {
		u32 j;
		for(j = 0; j < n_ents(x); j++) {
			if(entry(x, j) == i) {
				*k = j;
				return;
			}
		}
		*k = (u32)(-1);
		//printf("(find_index) index not found!\n");
	}


	/**
	 * Find the index of minimal element
	 *
	 * @param k the index.
	 * @param v the vector.
	 */
	__host__ __device__ void find_min_index(u32* k, const vec* v) {
		u32 j;
		float f = FLT_MAX;
		for(j = 0; j < n_ents(v); j++) {
			if(entry(v, j) < f) {
				f = entry(v, j);
				*k = j;
			}
		}
	}


	/**
	 * Sort a uvec in ascending order
	 *
	 * @param x the index.
	 */
	__host__ __device__ void sort_ascend(uvec* x) {
		u32 i, j, temp;
		for(i = n_ents(x)-1; i > 0; i--) {
			for(j = 0; j < i; j++) {
				if(entry(x, j) > entry(x, j+1)) {
					temp = entry(x, j);
					entry(x, j) = entry(x, j+1);
					entry(x, j+1) = temp;
				}
			}
		}
	}


	/**
	 * Sort a uvec in descending order
	 *
	 * @param x the index.
	 */
	__host__ __device__ void sort_descend(uvec* x) {
		u32 i, j, temp;
		for(i = n_ents(x)-1; i > 0; i--) {
			for(j = 0; j < i; j++) {
				if(entry(x, j) < entry(x, j+1)) {
					temp = entry(x, j);
					entry(x, j) = entry(x, j+1);
					entry(x, j+1) = temp;
				}
			}
		}
	}


	/**
	 * Sort a vector in ascending order, and record the indices of each elements.
	 *
	 * @note The vector will also be sorted.
	 * @param z the recorded index, should have the same length as v.
	 * @param v the vector
	 */
	__host__ __device__ void sort_index_ascend(uvec* z, vec* v) {
		//if(m_ents(z) != m_ents(v)) {
		//	printf("Warning: sort_index_ascend overflowed!\n");
		//}
		u32 i, j, ztemp;
		float vtemp;
		n_ents(z) = n_ents(v);
		for(u32 i = 0; i < n_ents(v); i++) {
			entry(z, i) = i;
		}
		if(n_ents(v) >= 2) {
			for(i = n_ents(v)-1; i > 0; i--) {
				for(j = 0; j < i; j++) {
					if(entry(v, j) > entry(v, j+1)) {
						vtemp = entry(v, j);
						entry(v, j) = entry(v, j+1);
						entry(v, j+1) = vtemp;
						ztemp = entry(z, j);
						entry(z, j) = entry(z, j+1);
						entry(z, j+1) = ztemp;
					}
				}
			}
		}
	}


	/**
	 * Sort a vector in descending order, and record the indices of each elements.
	 *
	 * @param z the sorted index, should have the same length as v.
	 * @param v the vector (will be sorted).
	 */
	__host__ __device__ void sort_index_descend(uvec* z, vec* v) {
		//if(m_ents(z) != m_ents(v)) {
		//	printf("Warning: sort_index_descend overflowed!\n");
		//}
		u32 i, j, ztemp;
		float vtemp;
		n_ents(z) = n_ents(v);
		for(u32 i = 0; i < n_ents(v); i++) {
			entry(z, i) = i;
		}
		if(n_ents(v) >= 2) {
			for(i = n_ents(v)-1; i > 0; i--) {
				for(j = 0; j < i; j++) {
					if(entry(v, j) < entry(v, j+1)) {
						vtemp = entry(v, j);
						entry(v, j) = entry(v, j+1);
						entry(v, j+1) = vtemp;
						ztemp = entry(z, j);
						entry(z, j) = entry(z, j+1);
						entry(z, j+1) = ztemp;
					}
				}
			}
		}
	}


	/**
	 * Set complement of a ascending sorted index
	 *
	 * @param z the complement set index.
	 * @param x the origin set index.
	 * @param n the length of universe.
	 */
	__host__ __device__ void complement(uvec* z, uvec* x, u32 n) {
		//if(m_ents(z) < n) {
		//	printf("Warning: complement overflowed!\n");
		//}
		u32 i, j, f;
		n_ents(z) = n - n_ents(x);
		for(i = 0, j = 0, f = 0; j < n_ents(z); f++) {
			if(i < n_ents(x) && f == entry(x, i)) {
				i++;
			}
			else {
				entry(z, j) = f;
				j++;
			}
		}
	}


	/**
	 * Set difference of two ascending sorted index (z = x \ y)
	 *
	 * @param z the difference set index.
	 * @param x the minuend set index.
	 * @param y the subtrahend set index.
	 */
	__host__ __device__ void set_diff(uvec* z, uvec* x, uvec* y) {
		//if(m_ents(z) != m_ents(x)) {
		//	printf("Warning: set_diff overflowed!\n");
		//}
		u32 i, j, k;
		for(i = 0, j = 0, k = 0; i < n_ents(x) && j < y[0];) {
			if(entry(x, i) < entry(y, j)) {
				entry(z, k) = entry(x, i);
				i++;
				k++;
			} else if(entry(x, i) == entry(y, j)) {
				i++;
				j++;
			} else {
				j++;
			}
		}
		for(; i < n_ents(x); i++, k++) {
			entry(z, k) = entry(x, i);
		}
		n_ents(z) = k;
	}

	/**
	 * Generate random value in normal distribution
	 *
	 * @param mu the difference set index.
	 * @param sigma the minuend set index.
	 * @param y the subtrahend set index.
	 */
	__host__ float randn(float mu, float sigma)
	{
		float u1, u2, w, m;
		static float x1, x2;
		static int called = 0;
		if(called == 1)
		{
			called = !called;
			return (mu + sigma * x2);
		}
		do
		{
			u1 = -1 + ((float)rand() / RAND_MAX) * 2;
			u2 = -1 + ((float)rand() / RAND_MAX) * 2;
			w = u1 * u1 + u2 * u2;
		}
		while(w >= 1 || w == 0);
		
		m = sqrt ((-2 * log(w)) / w);
		x1 = u1 * m;
		x2 = u2 * m;
		
		called = !called;
		
		return (mu + sigma * (float)x1);
	}
}
