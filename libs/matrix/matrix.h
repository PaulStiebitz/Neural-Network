#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

/* 1-D array of floats with an explicit length. */
typedef struct {
    uint32_t rows; /* Number of elements. */
    float  *data; /* Heap-allocated element array. */
} Vector;

/* 2-D matrix stored in row-major order as a flat float array. */
typedef struct {
    uint32_t rows;    /* Number of rows. */
    uint32_t columns; /* Number of columns. */
    float *data;    /* Heap-allocated data buffer (size = rows * columns). */
} Matrix;

/* A matrix paired with a class label (e.g. a digit image and its digit 0-9). */
typedef struct {
    uint8_t label;  /* Class label. */
    Matrix *matrix; /* Pixel data stored as floats. */
} LabeledMatrix;

/* An ordered collection of LabeledMatrix pointers, all sharing the same
   row and column dimensions. */
typedef struct {
    uint32_t list_length;    /* Number of entries in the list. */
    uint32_t matrix_rows;    /* Rows of every matrix in the list. */
    uint32_t matrix_columns; /* Columns of every matrix in the list. */
    LabeledMatrix **list;    /* Heap-allocated array of pointers. */
} LabeledMatrixList;

/* Vector functions */

/* Allocates a zero-initialised Vector of pRows elements. Returns NULL on failure. */
Vector *createVector(uint32_t pRows);

/* Flattens a Matrix into a new Vector (row-major). Returns NULL on failure. */
Vector *matrixToVector(Matrix *pMatrix);

/* Computes W * v and stores the result in pProduct_Destination.
   Dimensions must satisfy: W.columns == v.rows and dest.rows == W.rows. */
void matrixTimesVector(Matrix *pMatrix, Vector *pVector, Vector *pProduct_Destination);

/* Computes W^T * v and stores the result in pProduct_Destination.
   Dimensions must satisfy: W.rows == v.rows and dest.rows == W.columns. */
void matrixTransposeTimesVector(Matrix *pMatrix, Vector *pVector, Vector *pProduct_Destination);

/* Adds pVector_a to pVector_b_destination in-place (b = b + a).
   Both vectors must have the same length. */
void vectorPlusVector(Vector *pVector_a, Vector *pVector_b_destination);

/* Divides every element of pVector by pDivisor in-place. No-op if divisor is 0. */
void normalizeVector(Vector *pVector, float pDivisor);

/* Prints all elements of pVector separated by commas. */
void printVector(Vector *pVector);

/* Frees a Vector and its data buffer. No-op if pVector is NULL. */
void freeVector(Vector *pVector);

/* Matrix functions */

/* Allocates an uninitialised Matrix of the given dimensions. Returns NULL on failure. */
Matrix *createMatrix(uint32_t pRows, uint32_t pColumns);

/* Prints every element of pMatrix row by row.
   pData_type: "int" prints pixels as 0/1, "float" prints two decimal places. */
void printMatrix(const Matrix *pMatrix, char pData_type[]);

/* Fills every element of pMatrix with a small random value using uniform
   distribution in [-0.05, 0.05]. */
void fillMatrixRandom(Matrix *pMatrix);

/* Frees a Matrix and its data buffer. No-op if pMatrix is NULL. */
void freeMatrix(Matrix *pMatrix);

/* LabeledMatrix functions */

/* Allocates a LabeledMatrix with an uninitialised pixel buffer of rows x columns. */
LabeledMatrix *createLabeledMatrix(uint32_t rows, uint32_t columns);

/* Prints the label and then the pixel data of pLabeled_Matrix.
   No-op if pLabeled_Matrix is NULL. */
void printLabeledMatrix(const LabeledMatrix *pLabeled_Matrix, char type[]);

/* Frees a LabeledMatrix and its inner Matrix. No-op if pLabeled_Matrix is NULL. */
void freeLabeledMatrix(LabeledMatrix *pLabeled_Matrix);

/* LabeledMatrixList functions */

/* Allocates a LabeledMatrixList of pMatrix_list_length entries, each a
   rows x columns LabeledMatrix. Returns NULL on any allocation failure. */
LabeledMatrixList *createLabeledMatrixList(uint32_t pMatrix_list_length, uint32_t rows, uint32_t columns);

/* Prints up to pPrint_limit entries from the list.
   If pPrint_limit is 0, all entries are printed. */
void printLabeledMatrixList(const LabeledMatrixList *pLabeled_matrix_list, uint32_t pPrint_limit, char *pData_type);

/* Frees all LabeledMatrix entries, the pointer array, and the list struct itself. */
void freeLabeledMatrixList(LabeledMatrixList *pMatrix_list);

#endif
