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

void print(const vec* v);   // print v
void print(const mat* m);   // print m

vec* copy(const vec* m);    // copy v
mat* copy(const mat* v);    // copy m

vec* add(const vec* a, const vec* b);   // calculate a+b
mat* add(const mat* a, const mat* b);   // calculate a+b

vec* sub(const vec* a, const vec* b);   // calculate a-b
mat* sub(const mat* a, const mat* b);   // calculate a-b

vec* mul(const vec* v, const double e); // calculate e*v
mat* mul(const mat* m, const double e); // calculate e*m
vec* mul(const mat* m, const vec* v);   // calculate m*v
vec* mul(const vec* v, const mat* m);   // calculate (v'*m)' = m'v
mat* mul(const vec* v, const vec* w);   // calculate v*w'

double inner(const vec* v);                 // calculate sum(v.*v)
double inner(const vec* a, const vec* b);   // calculate sum(a.*b)
double inner(const mat* m);                 // calculate sum(sum(m.*m))
double inner(const mat* a, const mat* b);   // calculate sum(sum(a.*b))

void insert(vec* v, const double e);    // add a new entry at the end
void insert_col(mat* m, const vec* v);  // add a new col at the end
void insert_row(mat* m, const vec* v);  // add a new row at the end

void shed(vec* v);      // remove the final entry
void shed_col(mat* m);  // remove the final col
void shed_row(mat* m);  // remove the final row

void swap(vec* v, const uint32_t i, const uint32_t j);      // swap two entries
void swap_col(mat* m, const uint32_t i, const uint32_t j);  // swap two cols
void swap_row(mat* m, const uint32_t i, const uint32_t j);  // swap two rows
