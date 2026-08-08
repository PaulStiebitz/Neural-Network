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

void freeMatrix(Matrix *pMatrix) {
    if(pMatrix == NULL) {
        return;
    }
    for(uint32_t i = 0; i < pMatrix->rows; i++) {
        free(pMatrix->data[i]);
    }
    free(pMatrix->data);
    free(pMatrix);
}

MatrixList * createMatrixList(uint32_t length, uint32_t rows, uint32_t columns) {
    size_t MatrixList_mem_req = sizeof(MatrixList);
    MatrixList * matrix_list = malloc(MatrixList_mem_req);
    if(matrix_list == NULL) {
        return NULL;
    }

    matrix_list->list_length = length;
    matrix_list->matrix_rows = rows;
    matrix_list->matrix_columns = columns;
    size_t Matrix_innerList_mem_req = sizeof(Matrix) * length;

    matrix_list->list = malloc(Matrix_innerList_mem_req);
    if(matrix_list->list == NULL) {
        return NULL;
    }

    size_t Matrix_mem_req = sizeof(Matrix);
    for(uint32_t i = 0; i < length; i++) {
        matrix_list->list[i] = malloc(Matrix_mem_req);
        if(matrix_list->list[i] == NULL) {
            for(uint32_t j = 0; j < i; j++) {
                free(matrix_list->list[i]);
            }
            free(matrix_list);
            return NULL;
        }
        matrix_list->list[i] = createMatrix(rows, columns);
    }
    return matrix_list;
}

void freeMatrixList(MatrixList *pMatrixList) {
    for(uint32_t i = 0; i < pMatrixList->list_length; i++) {
        freeMatrix(pMatrixList->list[i]);
    }
    free(pMatrixList);
}

void printMatrix(Matrix *pMatrix, uint32_t rows, uint32_t columns) {
    for(uint32_t i = 0; i < rows; i++) {
        for(uint32_t j = 0; j < columns; j++) {
            printf("%d", pMatrix->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixList(MatrixList * pMatrix_list, uint32_t pPrintLimit) {
    uint32_t printLimit = 0;
    if(pPrintLimit > 0) {
        printLimit = pPrintLimit;
    } else {
        printLimit = pPrintLimit;
    }

    for(uint32_t i = 0; i < printLimit; i++) {
        printf("%d.th Matrix \n", i + 1);
        printMatrix(pMatrix_list->list[i], pMatrix_list->matrix_rows, pMatrix_list->matrix_columns);
    }
}