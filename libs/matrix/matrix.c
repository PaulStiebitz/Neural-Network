#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

Matrix * createMatrix(uint32_t rows, uint32_t columns) {
    Matrix * matrix  = malloc(sizeof(Matrix));
    if(matrix == NULL) {
        return NULL;
    }
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->data = malloc(sizeof(uint32_t) * rows);

    if(matrix->data == NULL) {
        free(matrix);
        return NULL;
    }

    for(uint32_t i = 0; i < rows; i++) {
        matrix->data[i] = malloc(sizeof(uint32_t) * columns);
        if(matrix->data[i] == NULL) {
            for(uint32_t j = 0; j < i; j++) {
                free(matrix->data[j]);
            }
            free(matrix->data);
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void removeMatrix(Matrix *pMatrix) {
    if(pMatrix == NULL) {
        return;
    }
    for(uint32_t i = 0; i < pMatrix->rows; i++) {
        free(pMatrix->data[i]);
    }
    free(pMatrix->data);
    free(pMatrix);
}

void printMatrix(Matrix *pMatrix, uint32_t rows, uint32_t columns) {
    for(uint32_t i = 0; i < rows; i++) {
        for(uint32_t j = 0; j < columns; j++) {
            printf("%d", pMatrix->data[i][j]);
        }
        printf("\n");
    }
}