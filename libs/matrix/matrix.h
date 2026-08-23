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

Matrix *createMatrix(uint32_t rows, uint32_t columns);
void freeMatrix(Matrix *pMatrix);
void printMatrix(const Matrix *pMatrix, char pData_type[]);

LabeledMatrix * createLabeledMatrix(uint32_t rows, uint32_t columns);
void freeLabeledMatrix(LabeledMatrix *pLabeled_Matrix);
void printLabeledMatrix(const LabeledMatrix *pLabeled_Matrix, char type[]);

LabeledMatrixList *createLabeledMatrixList(uint32_t pMatrix_list_length, uint32_t rows, uint32_t columns);
void freeLabeledMatrixList(LabeledMatrixList *pMatrix_list);
void printLabeledMatrixList(const LabeledMatrixList * pLabeled_matrix_list, uint32_t pPrint_limit, char *pData_type);

void freeMatrixList(LabeledMatrixList *pMatrix_list);
void printMatrixList(const LabeledMatrixList *pMatrix_list, uint32_t pPrint_limit, char pData_type[]);

Matrix *MatrixListToVector(const LabeledMatrixList *pMatrix_list);

#endif