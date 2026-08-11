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

LabeledMatrix * createLabeledMatrix(uint32_t pRows, uint32_t pColumns) {
    size_t labeled_matrix_mem_req = sizeof(LabeledMatrix);
    LabeledMatrix *labeled_matrix = malloc(labeled_matrix_mem_req);
    if(labeled_matrix == NULL) {
        return NULL;
    }
    labeled_matrix->label = 0;
    size_t labeled_matrix_inner_matrix_mem_req = sizeof(Matrix);
    labeled_matrix->matrix = malloc(labeled_matrix_inner_matrix_mem_req);

    if(labeled_matrix->matrix == NULL) {
        free(labeled_matrix);
        return NULL;
    }

    size_t labeled_matrix_inner_data_mem_req = sizeof(float) * pRows * pColumns;
    labeled_matrix->matrix->data = malloc(labeled_matrix_inner_data_mem_req);
    if(labeled_matrix->matrix->data == NULL) {
        freeMatrix(labeled_matrix->matrix);
        free(labeled_matrix);
        return NULL;
    }
    labeled_matrix->matrix->rows = pRows;
    labeled_matrix->matrix->columns = pColumns;

    return labeled_matrix;
}

void freeLabeledMatrix(LabeledMatrix *pLabeled_Matrix) {
    if(pLabeled_Matrix == NULL) {
        return;
    }
    freeMatrix(pLabeled_Matrix->matrix);
    free(pLabeled_Matrix);
}

void printLabeledMatrix(const LabeledMatrix *pLabeled_Matrix, char type[]) {
    printf("Label: %d\n", pLabeled_Matrix->label);
    printMatrix(pLabeled_Matrix->matrix, type);
}

/* Allocates a MatrixList of 'length' matrices, each of size rows x columns. Returns NULL on failure. */
LabeledMatrixList *createLabeledMatrixList(uint32_t pLabeled_matrix_list_length, uint32_t pRows, uint32_t pColumns) {
    size_t labeled_matrix_list_mem_req = sizeof(LabeledMatrixList);
    LabeledMatrixList * labeled_matrix_list = malloc(labeled_matrix_list_mem_req);
    if(labeled_matrix_list == NULL) {
        return NULL;
    }

    labeled_matrix_list->list_length = pLabeled_matrix_list_length;
    labeled_matrix_list->matrix_rows = pRows;
    labeled_matrix_list->matrix_columns = pColumns;

    size_t Matrix_innerList_mem_req = sizeof(LabeledMatrix) * pLabeled_matrix_list_length;
    labeled_matrix_list->list = malloc(Matrix_innerList_mem_req);
    if(labeled_matrix_list->list == NULL) {
        free(labeled_matrix_list);
        return NULL;
    }

    for(uint32_t i = 0; i < pLabeled_matrix_list_length; i++) {
        labeled_matrix_list->list[i] = createLabeledMatrix(pRows, pColumns);
        if(labeled_matrix_list->list[i] == NULL) {
            freeLabeledMatrixList(labeled_matrix_list);
            return NULL;
        }
    }
    return labeled_matrix_list;
}

/* Frees all matrices in the list, the list array, and the MatrixList itself. */
void freeLabeledMatrixList(LabeledMatrixList *pLabeled_matrix_list) {
    for(uint32_t i = 0; i < pLabeled_matrix_list->list_length; i++) {
        freeLabeledMatrix(pLabeled_matrix_list->list[i]);
    }
    free(pLabeled_matrix_list->list);
    free(pLabeled_matrix_list);
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
void printLabeledMatrixList(const LabeledMatrixList * pLabeled_matrix_list, uint32_t pPrint_limit, char *pData_type) {
    uint32_t printLimit = 0;
    if(pPrint_limit > 0) {
        printLimit = pPrint_limit;
    } else {
        printLimit = pLabeled_matrix_list->list_length;
    }

    for(uint32_t i = 0; i < printLimit; i++) {
        uint8_t label = pLabeled_matrix_list->list[i]->label;
        printf("%d.th Matrix \nLabel: %d\n", i + 1, label);
        if(strcmp(pData_type, "int") == 0) {
                printMatrix(pLabeled_matrix_list->list[i]->matrix, "int");
            } else if(strcmp(pData_type, "float") == 0) {
                printMatrix(pLabeled_matrix_list->list[i]->matrix, "float");
        }
    }
}

/* Flattens all matrices in the list into a single column vector. Returns NULL on failure. */
Matrix *MatrixListToVector(const LabeledMatrixList *pLabeled_matrix_list) {
    if(pLabeled_matrix_list == NULL) {
        return NULL;
    }
    size_t matrix_mem_req = sizeof(Matrix);
    Matrix * vector = malloc(matrix_mem_req);
    if(vector == NULL) {
        return NULL;
    }
    uint32_t matrix_elements = pLabeled_matrix_list->matrix_rows * pLabeled_matrix_list->matrix_columns;
    uint32_t vector_elements = pLabeled_matrix_list->list_length * matrix_elements;
    size_t vector_elements_mem_req = sizeof(float) * vector_elements;
    vector->data = malloc(vector_elements_mem_req);
    if(vector->data == NULL) {
        free(vector);
        return NULL;
    }

    vector->rows = vector_elements;
    vector->columns = 1;

    uint32_t vector_index = 0;
    for(uint32_t i = 0; i < pLabeled_matrix_list->list_length; i++) {
        Matrix * current_matrix = pLabeled_matrix_list->list[i]->matrix;
        for(uint32_t j = 0; j < matrix_elements; j++) {
            float current_data = current_matrix->data[j];
            vector->data[vector_index] = current_data;
            vector_index++;
        }
    }
    return vector;
}