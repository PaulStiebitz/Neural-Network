#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

typedef struct {
    uint32_t rows;
    uint32_t columns;
    float *data;
} Matrix;

typedef struct {
    uint32_t list_length;
    uint32_t matrix_rows;
    uint32_t matrix_columns;
    Matrix ** list;
} MatrixList;

Matrix *createMatrix(uint32_t rows, uint32_t columns);
void removeMatrix(Matrix *pMatrix);
void printMatrix(const Matrix *pMatrix, char pDataType[]);

MatrixList *createMatrixList(uint32_t pMatrixListLength, uint32_t rows, uint32_t columns);
void freeMatrixList(MatrixList *pMatrixList);
void printMatrixList(const MatrixList *pMatrix_listm, uint32_t pPrintLimit, char pDataType[]);

Matrix *MatrixListToVector(const MatrixList *pMatrixList);

#endif