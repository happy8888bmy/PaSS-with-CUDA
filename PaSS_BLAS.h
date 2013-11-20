/**
 * PaSS_BLAS.h <p>
 * The basic linear algebra subprograms for PaSS
 *
 * @author emfo
 */

#include <cstdint>

/**
 * The vector struct
 */
struct vec {
    uint32_t* n;    // the length
    double* e;      // the array of entries
};

/**
 * The matrix struct
 * <remarks>
 * Note that *(m->col[i]->n) = m->n_row for all i, where m is a mat*
 * </remarks>
 */
struct mat {
    uint32_t n_row; // the number of rows
    uint32_t n_col; // the number of cols
    vec** col;      // the array of columns
};

vec* construct(const uint32_t* n);                          // construct a n-by-1 vector
mat* construct(const uint32_t* p, const uint32_t* q);       // construct a p-by-q matrix

void destruct(const vec* v);                                // destruct the vector
void destruct(const mat* m);                                // destruct the matrix

void print(const vec* v);                                   // display the vector
void print(const mat* m);                                   // display the matrix

vec* copy(const vec* m);                                    // copy the vector
mat* copy(const mat* v);                                    // copy the matrix

vec* add(const vec* a, const vec* b);                       // calculate a+b
mat* add(const mat* a, const mat* b);                       // calculate a+b
void add(vec* a, const vec* b);                             // a += b
void add(mat* a, const mat* b);                             // a += b

vec* mul(const vec* v, const double d);                     // calculate d*v
mat* mul(const mat* m, const double d);                     // calculate d*m
vec* mul(const mat* m, const vec* v);                       // calculate m*v
vec* mul(const vec* v, const mat* m);                       // calculate m'*v (same as (v'*m)')
mat* mul(const vec* v, const vec* w);                       // calculate v*w'
void mul(vec* v, const double d);                           // v *= d
void mul(mat* m, const double d);                           // m *= d
void mul(const mat* m, vec* v);                             // v = m*v
void mul(vec* v, const mat* m);                             // v = m'*v

double inner(const vec* v);                                 // calculate sum(v.*v)
double inner(const vec* a, const vec* b);                   // calculate sum(a.*b)
double inner(const mat* m);                                 // calculate sum(sum(m.*m))
double inner(const mat* a, const mat* b);                   // calculate sum(sum(a.*b))

void insert(vec* v, const double e);                        // add a new entry at the end
void insert_col(mat* m, const vec* v);                      // add a new col at the end
void insert_row(mat* m, const vec* v);                      // add a new row at the end

void shed(vec* v);                                          // remove the last entry
void shed_col(mat* m);                                      // remove the last col
void shed_row(mat* m);                                      // remove the last row

void swap(vec* v, const uint32_t i, const uint32_t j);      // swap two entries
void swap_col(mat* m, const uint32_t i, const uint32_t j);  // swap two cols
void swap_row(mat* m, const uint32_t i, const uint32_t j);  // swap two rows
