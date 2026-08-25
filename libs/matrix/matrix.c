#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"


Vector *createVector(uint32_t pRows) {
    size_t vector_mem_req = sizeof(Vector);
    Vector * vector = malloc(vector_mem_req);
    if(vector == NULL) {
        return NULL;
    }
    vector->rows = pRows;
    size_t float_mem_req = sizeof(float);
    vector->data = calloc(pRows, float_mem_req);
    if(vector->data == NULL) {
        free(vector);
        return NULL;
    }
    return vector;
}

Vector *matrixToVector(Matrix *pMatrix) {
    if(pMatrix == NULL) {
        return NULL;
    }
    size_t vector_mem_req = sizeof(Vector);
    Vector *temp_vector = malloc(vector_mem_req);
    if(temp_vector == NULL) {
        return NULL;
    }
    size_t float_mem_req = sizeof(float);
    uint32_t num_data_fields = pMatrix->rows * pMatrix->columns;
    temp_vector->data = calloc(num_data_fields, float_mem_req);
    if(temp_vector->data == NULL) {
        free(temp_vector);
        return NULL;
    }
    temp_vector->rows = num_data_fields;

    for(uint32_t i = 0; i < num_data_fields; i++) {
        temp_vector->data[i] = pMatrix->data[i];
    }
    return temp_vector;
}

void normalizeVector(Vector *pVector, float pDivisor) {
    if(pVector == NULL || pDivisor == 0.0f) {
        return;
    }
    for(uint32_t i = 0; i < pVector->rows; i++) {
        pVector->data[i] = pVector->data[i] / pDivisor;
    }
}

void freeVector(Vector *pVector) {
    if(pVector == NULL) {
        return;    }
    free(pVector->data);
    free(pVector);
}

void printVector(Vector *pVector) {
    if(pVector == NULL) {
        return;
    }

    for(uint32_t i = 0; i < pVector->rows; i++) {
        if(i+1 != pVector->rows) {
            printf("%.2f, ", pVector->data[i]);
        } else {
            printf("%.2f", pVector->data[i]);
        }
    }
    printf("\n");
}

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

void fillMatrixRandom(Matrix *pMatrix) {
    if(pMatrix == NULL) {
        return;
    }
    uint32_t num_values = pMatrix->rows * pMatrix->columns;
    for(uint32_t i = 0; i < num_values; i++) {
        pMatrix->data[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
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

    labeled_matrix_list->list = calloc(pLabeled_matrix_list_length, sizeof(LabeledMatrix *));
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
    if(pLabeled_matrix_list == NULL) {
        return;
    }
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
                printf("%.2f ", pMatrix->data[matrixIndex]);
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

void matrixTimesVector(Matrix *pMatrix, Vector *pVector, Vector *pProduct_Destination) {
    if(pMatrix == NULL || pProduct_Destination == NULL || pVector == NULL) {
        return;
    }

    if(pMatrix->columns != pVector->rows || pProduct_Destination->rows != pMatrix->rows) {
        return;
    }

    for(uint32_t i = 0; i < pProduct_Destination->rows; i++) {
        pProduct_Destination->data[i] = 0.0f;
    }

    for(uint32_t i = 0; i < pMatrix->rows; i++) {
        for(uint32_t j = 0; j < pMatrix->columns; j++) {
            uint32_t current_index = i * pMatrix->columns + j;
            pProduct_Destination->data[i] += pMatrix->data[current_index] * pVector->data[j];
        }
    }
}

void vectorPlusVector(Vector *pVector_a, Vector *pVector_b_destination) {
    if(pVector_a == NULL || pVector_b_destination == NULL || pVector_a->rows != pVector_b_destination->rows) {
        return;
    }
    for(uint32_t i = 0; i < pVector_b_destination->rows; i++) {
        pVector_b_destination->data[i] =  pVector_b_destination->data[i] + pVector_a->data[i];
    }
}