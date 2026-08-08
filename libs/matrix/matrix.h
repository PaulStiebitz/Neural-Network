#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

typedef struct
{
    uint32_t rows;
    uint32_t columns;
    uint32_t **data;
} Matrix;

Matrix * createMatrix(uint32_t rows, uint32_t columns);

void removeMatrix(Matrix *pMatrix);

void printMatrix(Matrix *pMatrix, uint32_t rows, uint32_t columns);

#endif