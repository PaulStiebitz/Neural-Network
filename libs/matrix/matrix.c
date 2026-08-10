#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"

/* Allocates a Matrix with the given dimensions. Returns NULL on failure. */
Matrix * createMatrix(uint32_t rows, uint32_t columns) {
    Matrix * matrix  = malloc(sizeof(Matrix));
    if(matrix == NULL) {
        return NULL;
    }
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->data = malloc(sizeof(float) * rows * columns);

    if(matrix->data == NULL) {
        free(matrix);
        return NULL;
    }
    return matrix;
}

/* Frees a Matrix and its data buffer. No-op if pMatrix is NULL. */
void freeMatrix(Matrix *pMatrix) {
    if(pMatrix == NULL) {
        return;
    }

    free(pMatrix->data);
    free(pMatrix);
}

/* Allocates a MatrixList of 'length' matrices, each of size rows x columns. Returns NULL on failure. */
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
        free(matrix_list);
        return NULL;
    }

    for(uint32_t i = 0; i < length; i++) {
        matrix_list->list[i] = createMatrix(rows, columns);
        if(matrix_list->list[i] == NULL) {
            for(uint32_t j = 0; j < i; j++) {
                freeMatrix(matrix_list->list[j]);
            }
            free(matrix_list->list);
            free(matrix_list);
            return NULL;
        }
    }
    return matrix_list;
}

/* Frees all matrices in the list, the list array, and the MatrixList itself. */
void freeMatrixList(MatrixList *pMatrixList) {
    for(uint32_t i = 0; i < pMatrixList->list_length; i++) {
        freeMatrix(pMatrixList->list[i]);
    }
    free(pMatrixList->list);
    free(pMatrixList);
}

/* Prints all elements of a matrix row by row. pDataType: "int" or "float". */
void printMatrix(const Matrix *pMatrix, char *pDataType) {
    uint32_t matrixIndex = 0;
    for(uint32_t i = 0; i < pMatrix->rows; i++) {
        for(uint32_t j = 0; j < pMatrix->columns; j++) {
            if(strcmp(pDataType, "int") == 0) {
                printf("%d", (uint8_t)pMatrix->data[matrixIndex]);
            } else if(strcmp(pDataType, "float") == 0) {
                printf("%.3f", pMatrix->data[matrixIndex]);
            }
            matrixIndex++;
        }
        printf("\n");
    }
    printf("\n");
}

/* Prints up to pPrintLimit matrices from the list. If pPrintLimit is 0, prints all. */
void printMatrixList(const MatrixList * pMatrix_list, uint32_t pPrintLimit, char *pDataType) {
    uint32_t printLimit = 0;
    if(pPrintLimit > 0) {
        printLimit = pPrintLimit;
    } else {
        printLimit = pMatrix_list->list_length;
    }

    for(uint32_t i = 0; i < printLimit; i++) {
        printf("%d.th Matrix \n", i + 1);
        if(strcmp(pDataType, "int") == 0) {
                printMatrix(pMatrix_list->list[i], "int");
            } else if(strcmp(pDataType, "float") == 0) {
                printMatrix(pMatrix_list->list[i], "float");
        }
    }
}

/* Flattens all matrices in the list into a single column vector. Returns NULL on failure. */
Matrix *MatrixListToVector(const MatrixList *pMatrixList) {
    if(pMatrixList == NULL) {
        return NULL;
    }
    size_t matrix_mem_req = sizeof(Matrix);
    Matrix * vector = malloc(matrix_mem_req);
    if(vector == NULL) {
        return NULL;
    }
    uint32_t matrix_elements = pMatrixList->matrix_rows * pMatrixList->matrix_columns;
    uint32_t vector_elements = pMatrixList->list_length * matrix_elements;
    size_t vector_elements_mem_req = sizeof(float) * vector_elements;
    vector->data = malloc(vector_elements_mem_req);
    if(vector->data == NULL) {
        free(vector);
        return NULL;
    }

    vector->rows = vector_elements;
    vector->columns = 1;

    uint32_t vector_index = 0;
    for(uint32_t i = 0; i < pMatrixList->list_length; i++) {
        Matrix * current_matrix = pMatrixList->list[i];
        for(uint32_t j = 0; j < matrix_elements; j++) {
            float current_data = current_matrix->data[j];
            vector->data[vector_index] = current_data;
            vector_index++;
        }
    }
    return vector;
}