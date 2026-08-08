#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "idx3_io.h"
#include "../matrix/matrix.h"
/*

Task
Initial: A B C D
Goal: D C B A

1.) num >> 24
A B C D
0 0 0 A

2.) num << 8
A B C D
B C D 0

3.) num >> 8
A B C D
0 0 B C

4.) num << 24
A B C D

*/

uint32_t flip_endian(uint32_t num) {
    return ((num >> 24) & 0xff) |
           ((num << 8) & 0xff0000) |
           ((num >> 8) & 0xff00) |
           ((num << 24) & 0xff000000);
}
/*
Bytes 0–1: Always zero 0x00 0x00
Byte 2: Data type code (0x08 for unsigned byte)
Byte 3: Number of dimensions (0x03 for 3D matrix)
Bytes 4–7: Number of images/matrices (32-bit big-endian integer)
Bytes 8–11: Number of rows per image (32-bit big-endian integer)
Bytes 12–15: Number of columns per image (32-bit big-endian integer)
Bytes 16+: Raw pixel data as a continuous stream of unsigned bytes

*/
MatrixList * getIDXdata(const char * path) {
    if(path == NULL) {
        return NULL;
    }

    FILE *file = fopen(path, "rb");
    if(file == NULL) {
        return NULL;
    }

    IDXMetadata idxMetadata = readIDXMetadata(file);
    uint32_t matrix_list_length = idxMetadata.matrix_count;
    uint32_t matrix_rows = idxMetadata.matrix_rows;
    uint32_t matrix_columns = idxMetadata.matrix_columns;
    MatrixList * matrix_list = createMatrixList(matrix_list_length, matrix_rows, matrix_columns);
    if(matrix_list == NULL) {
        return NULL;
    }

    buildIDXMatrixList(file, matrix_list);

    fclose(file);
    free(file);
    return matrix_list;
}

IDXMetadata readIDXMetadata(FILE *pFile) {
    if(pFile == NULL) {
        IDXMetadata emptyMeta = {0};
        return emptyMeta;
    }

    IDXMetadata idxMetadata = {0};

    size_t size_uint32_t = sizeof(uint32_t);
    size_t size_uint8_t = sizeof(uint8_t);

    fread(&idxMetadata.zero_bytes, size_uint8_t, 2, pFile);
    fread(&idxMetadata.bytes, size_uint8_t, 1, pFile);
    fread(&idxMetadata.dimension, size_uint8_t, 1, pFile);
    fread(&idxMetadata.matrix_count, size_uint32_t, 1, pFile);
    fread(&idxMetadata.matrix_rows, size_uint32_t, 1, pFile);
    fread(&idxMetadata.matrix_columns, size_uint32_t, 1, pFile);

    // 1 Byte fields are already correct!
    idxMetadata.matrix_count = flip_endian(idxMetadata.matrix_count);
    idxMetadata.matrix_rows = flip_endian(idxMetadata.matrix_rows);
    idxMetadata.matrix_columns = flip_endian(idxMetadata.matrix_columns);
    return idxMetadata;
}

void buildIDXMatrixList(FILE * pFile, MatrixList * pMatrixList) {
    printf("Reading [IDXData]:\n"
           "Matrix count: %d\n"
           "Matrix rows: %d\n"
           "Matrix columns: %d\n",
           pMatrixList->list_length, pMatrixList->matrix_rows, pMatrixList->matrix_columns
          );

   for(uint32_t i = 0; i < pMatrixList->list_length; i++) {
        pMatrixList->list[i] = createIDXDataMatrix(pMatrixList->matrix_rows, pMatrixList->matrix_columns, pFile);
   }
}

Matrix * createIDXDataMatrix(uint32_t rows, uint32_t columns, FILE * pFile) {
    Matrix * matrix = createMatrix(rows, columns);
    if(matrix == NULL) {
        return NULL;
    }

    uint32_t pixel_threshold = 127;
    for(uint32_t i = 0; i < rows; i++) {
        for(uint32_t j = 0; j < columns; j++) {
            uint8_t tmp_pixel = 0;
            fread(&tmp_pixel, sizeof(uint8_t), 1, pFile);

            if(tmp_pixel >= pixel_threshold) {
                matrix->data[i][j] = 1;
            } else {
                matrix->data[i][j] = 0;
            }
        }
    }
    return matrix;
}

