#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

typedef struct {
    uint32_t rows;
    float *data;
} Vector;

typedef struct {
    uint32_t rows;
    uint32_t columns;
    float *data;
} Matrix;

typedef struct {
    uint8_t label;
    Matrix *matrix;
} LabeledMatrix;

typedef struct {
    uint32_t list_length;
    uint32_t matrix_rows;
    uint32_t matrix_columns;
    LabeledMatrix ** list;
} LabeledMatrixList;

Vector *createVector(uint32_t pRows);
Vector *matrixToVector(Matrix *pMatrix);
void matrixTimesVector(Matrix *pMatrix, Vector *pVector, Vector *pProduct_Destination);
void vectorPlusVector(Vector *pVector_a, Vector *pVector_b_destination);
void normalizeVector(Vector *pVector, float pDivisor);
void printVector(Vector *pVector);
void freeVector(Vector *pVector);

Matrix *createMatrix(uint32_t pRows, uint32_t pColumns);
void printMatrix(const Matrix *pMatrix, char pData_type[]);
void fillMatrixRandom(Matrix *pMatrix);
void freeMatrix(Matrix *pMatrix);

LabeledMatrix * createLabeledMatrix(uint32_t rows, uint32_t columns);
void printLabeledMatrix(const LabeledMatrix *pLabeled_Matrix, char type[]);
void freeLabeledMatrix(LabeledMatrix *pLabeled_Matrix);

LabeledMatrixList *createLabeledMatrixList(uint32_t pMatrix_list_length, uint32_t rows, uint32_t columns);
void printLabeledMatrixList(const LabeledMatrixList * pLabeled_matrix_list, uint32_t pPrint_limit, char *pData_type);
void freeLabeledMatrixList(LabeledMatrixList *pMatrix_list);

#endif