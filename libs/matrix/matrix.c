#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

/* Allocates a zero-initialised Vector of pRows elements.
   Returns NULL if allocation fails. */
Vector *createVector(uint32_t pRows) {
    Vector *vector = malloc(sizeof(Vector));
    if(vector == NULL) {
        return NULL;
    }
    vector->rows = pRows;
    vector->data = calloc(pRows, sizeof(float));
    if(vector->data == NULL) {
        free(vector);
        return NULL;
    }
    return vector;
}

/* Flattens pMatrix into a newly allocated Vector stored in row-major order.
   Returns NULL if pMatrix is NULL or allocation fails. */
Vector *matrixToVector(Matrix *pMatrix) {
    if(pMatrix == NULL) {
        return NULL;
    }
    Vector *temp_vector = malloc(sizeof(Vector));
    if(temp_vector == NULL) {
        return NULL;
    }
    uint32_t num_data_fields = pMatrix->rows * pMatrix->columns;
    temp_vector->data = calloc(num_data_fields, sizeof(float));
    if(temp_vector->data == NULL) {
        free(temp_vector);
        return NULL;
    }
    temp_vector->rows = num_data_fields;

    /* Copy every element from the matrix into the flat vector. */
    for(uint32_t i = 0; i < num_data_fields; i++) {
        temp_vector->data[i] = pMatrix->data[i];
    }
    return temp_vector;
}

/* Divides every element of pVector by pDivisor in-place.
   No-op if pVector is NULL or pDivisor is zero. */
void normalizeVector(Vector *pVector, float pDivisor) {
    if(pVector == NULL || pDivisor == 0.0f) {
        return;
    }
    for(uint32_t i = 0; i < pVector->rows; i++) {
        pVector->data[i] = pVector->data[i] / pDivisor;
    }
}

/* Frees the data buffer and the Vector struct itself.
   No-op if pVector is NULL. */
void freeVector(Vector *pVector) {
    if(pVector == NULL) {
        return;
    }
    free(pVector->data);
    free(pVector);
}

/* Prints all elements of pVector on one line, separated by commas.
   No-op if pVector is NULL. */
void printVector(Vector *pVector) {
    if(pVector == NULL) {
        return;
    }
    for(uint32_t i = 0; i < pVector->rows; i++) {
        if(i + 1 != pVector->rows) {
            printf("%.2f, ", pVector->data[i]);
        } else {
            printf("%.2f", pVector->data[i]);
        }
    }
    printf("\n");
}

/* Allocates an uninitialised Matrix of rows x columns floats.
   Returns NULL if allocation fails. */
Matrix *createMatrix(uint32_t rows, uint32_t columns) {
    Matrix *matrix = malloc(sizeof(Matrix));
    if(matrix == NULL) {
        return NULL;
    }
    matrix->rows    = rows;
    matrix->columns = columns;
    matrix->data    = malloc(sizeof(float) * rows * columns);
    if(matrix->data == NULL) {
        free(matrix);
        return NULL;
    }
    return matrix;
}

/* Frees the data buffer and the Matrix struct itself.
   No-op if pMatrix is NULL. */
void freeMatrix(Matrix *pMatrix) {
    if(pMatrix == NULL) {
        return;
    }
    free(pMatrix->data);
    free(pMatrix);
}

/* Fills every element of pMatrix with a uniform random value in [-0.05, 0.05].
   No-op if pMatrix is NULL. */
void fillMatrixRandom(Matrix *pMatrix) {
    if(pMatrix == NULL) {
        return;
    }
    uint32_t num_values = pMatrix->rows * pMatrix->columns;
    for(uint32_t i = 0; i < num_values; i++) {
        pMatrix->data[i] = (((float)rand() / RAND_MAX) - 0.5f) * 0.1f;
    }
}

/* Allocates a LabeledMatrix with a zero label and an uninitialised pixel
   buffer of pRows x pColumns floats. Returns NULL if allocation fails. */
LabeledMatrix *createLabeledMatrix(uint32_t pRows, uint32_t pColumns) {
    LabeledMatrix *labeled_matrix = malloc(sizeof(LabeledMatrix));
    if(labeled_matrix == NULL) {
        return NULL;
    }
    labeled_matrix->label  = 0;
    labeled_matrix->matrix = malloc(sizeof(Matrix));
    if(labeled_matrix->matrix == NULL) {
        free(labeled_matrix);
        return NULL;
    }
    labeled_matrix->matrix->data = malloc(sizeof(float) * pRows * pColumns);
    if(labeled_matrix->matrix->data == NULL) {
        freeMatrix(labeled_matrix->matrix);
        free(labeled_matrix);
        return NULL;
    }
    labeled_matrix->matrix->rows    = pRows;
    labeled_matrix->matrix->columns = pColumns;
    return labeled_matrix;
}

/* Frees the inner Matrix and the LabeledMatrix struct itself.
   No-op if pLabeled_Matrix is NULL. */
void freeLabeledMatrix(LabeledMatrix *pLabeled_Matrix) {
    if(pLabeled_Matrix == NULL) {
        return;
    }
    freeMatrix(pLabeled_Matrix->matrix);
    free(pLabeled_Matrix);
}

/* Prints the label followed by the pixel data of pLabeled_Matrix.
   No-op if pLabeled_Matrix is NULL. */
void printLabeledMatrix(const LabeledMatrix *pLabeled_Matrix, char type[]) {
    if(pLabeled_Matrix == NULL) {
        return;
    }
    printf("Label: %d\n", pLabeled_Matrix->label);
    printMatrix(pLabeled_Matrix->matrix, type);
}

/* Allocates a LabeledMatrixList of pLabeled_matrix_list_length entries.
   Each entry is an individually allocated pRows x pColumns LabeledMatrix.
   On any allocation failure the partially built list is freed and NULL is returned. */
LabeledMatrixList *createLabeledMatrixList(uint32_t pLabeled_matrix_list_length, uint32_t pRows, uint32_t pColumns) {
    LabeledMatrixList *labeled_matrix_list = malloc(sizeof(LabeledMatrixList));
    if(labeled_matrix_list == NULL) {
        return NULL;
    }
    labeled_matrix_list->list_length    = pLabeled_matrix_list_length;
    labeled_matrix_list->matrix_rows    = pRows;
    labeled_matrix_list->matrix_columns = pColumns;

    /* Allocate the pointer array, zero-initialised so partial frees are safe. */
    labeled_matrix_list->list = calloc(pLabeled_matrix_list_length, sizeof(LabeledMatrix *));
    if(labeled_matrix_list->list == NULL) {
        free(labeled_matrix_list);
        return NULL;
    }

    for(uint32_t i = 0; i < pLabeled_matrix_list_length; i++) {
        labeled_matrix_list->list[i] = createLabeledMatrix(pRows, pColumns);
        if(labeled_matrix_list->list[i] == NULL) {
            /* Free everything allocated so far and bail out. */
            freeLabeledMatrixList(labeled_matrix_list);
            return NULL;
        }
    }
    return labeled_matrix_list;
}

/* Frees every LabeledMatrix in the list, then the pointer array, then the
   list struct itself. No-op if pLabeled_matrix_list is NULL. */
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

/* Prints pMatrix row by row.
   pDataType "int"   -> each element printed as 0 (zero pixel) or 1 (non-zero pixel).
   pDataType "float" -> each element printed with two decimal places. */
void printMatrix(const Matrix *pMatrix, char *pDataType) {
    uint32_t matrixIndex = 0;
    for(uint32_t i = 0; i < pMatrix->rows; i++) {
        for(uint32_t j = 0; j < pMatrix->columns; j++) {
            if(strcmp(pDataType, "int") == 0) {
                /* Threshold: any non-zero pixel is shown as 1. */
                printf("%d", pMatrix->data[matrixIndex] > 0.0f ? 1 : 0);
            } else if(strcmp(pDataType, "float") == 0) {
                printf("%.2f ", pMatrix->data[matrixIndex]);
            }
            matrixIndex++;
        }
        printf("\n");
    }
    printf("\n");
}

/* Prints up to pPrint_limit entries from pLabeled_matrix_list, each with its
   label header and pixel data. If pPrint_limit is 0, all entries are printed. */
void printLabeledMatrixList(const LabeledMatrixList *pLabeled_matrix_list, uint32_t pPrint_limit, char *pData_type) {
    uint32_t printLimit;
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

/* Computes the matrix-vector product W * v (pMatrix x pVector) and stores
   the result in pProduct_Destination. All three pointers must be non-NULL and
   dimensions must satisfy: W.columns == v.rows, dest.rows == W.rows. */
void matrixTimesVector(Matrix *pMatrix, Vector *pVector, Vector *pProduct_Destination) {
    if(pMatrix == NULL || pVector == NULL || pProduct_Destination == NULL) {
        return;
    }
    if(pMatrix->columns != pVector->rows || pProduct_Destination->rows != pMatrix->rows) {
        return;
    }

    /* Zero the destination before accumulation. */
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

/* Computes the transposed matrix-vector product W^T * v and stores the result
   in pProduct_Destination. Dimensions must satisfy: W.rows == v.rows,
   dest.rows == W.columns. */
void matrixTransposeTimesVector(Matrix *pMatrix, Vector *pVector, Vector *pProduct_Destination) {
    if(pMatrix == NULL || pVector == NULL || pProduct_Destination == NULL) {
        return;
    }
    if(pMatrix->rows != pVector->rows || pProduct_Destination->rows != pMatrix->columns) {
        return;
    }

    /* Zero the destination before accumulation. */
    for(uint32_t i = 0; i < pProduct_Destination->rows; i++) {
        pProduct_Destination->data[i] = 0.0f;
    }

    /* Iterate over columns of W^T (= rows of W). */
    for(uint32_t i = 0; i < pMatrix->columns; i++) {
        for(uint32_t j = 0; j < pMatrix->rows; j++) {
            /* Access element A[j][i] in row-major storage. */
            uint32_t current_index = j * pMatrix->columns + i;
            pProduct_Destination->data[i] += pMatrix->data[current_index] * pVector->data[j];
        }
    }
}

/* Adds pVector_a into pVector_b_destination element-wise in-place (b = b + a).
   No-op if either pointer is NULL or the lengths differ. */
void vectorPlusVector(Vector *pVector_a, Vector *pVector_b_destination) {
    if(pVector_a == NULL || pVector_b_destination == NULL
       || pVector_a->rows != pVector_b_destination->rows) {
        return;
    }
    for(uint32_t i = 0; i < pVector_b_destination->rows; i++) {
        pVector_b_destination->data[i] = pVector_b_destination->data[i] + pVector_a->data[i];
    }
}
