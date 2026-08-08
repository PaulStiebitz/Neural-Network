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

Matrix ** getIDXdata(const char * path) {
    if(path == NULL) {
        return NULL;
    }
    FILE *file = fopen(path, "rb");

    if(file == NULL) {
        return NULL;
    }

    IDXMeta idxmeta = readIDXMeta(file);
    size_t matrix_list_size =  sizeof(Matrix) * idxmeta.matrix_count;
    Matrix ** matrix_list = malloc(matrix_list_size);

    if(matrix_list == NULL) {
        return NULL;
    }

    readIDXData(idxmeta, file, matrix_list);

    fclose(file);
    free(file);

    return matrix_list;
}

IDXMeta readIDXMeta(FILE *pFile) {
    if(pFile == NULL) {
        IDXMeta emptyMeta = {0};
        return emptyMeta;
    }

    IDXMeta idxmeta = {0};

    size_t size_uint32_t = sizeof(uint32_t);
    size_t size_uint8_t = sizeof(uint8_t);

    fread(&idxmeta.zero_bytes, size_uint8_t, 2, pFile);
    fread(&idxmeta.bytes, size_uint8_t, 1, pFile);
    fread(&idxmeta.dimension, size_uint8_t, 1, pFile);
    fread(&idxmeta.matrix_count, size_uint32_t, 1, pFile);
    fread(&idxmeta.matrix_rows, size_uint32_t, 1, pFile);
    fread(&idxmeta.matrix_columns, size_uint32_t, 1, pFile);

    // 1 Byte fields are already correct!
    idxmeta.matrix_count = flip_endian(idxmeta.matrix_count);
    idxmeta.matrix_rows = flip_endian(idxmeta.matrix_rows);
    idxmeta.matrix_columns = flip_endian(idxmeta.matrix_columns);
    return idxmeta;
}

void readIDXData(IDXMeta idxmeta, FILE * pFile, Matrix ** pList) {
    printf("Reading [IDXData]:\n"
           "Matrix Dimension: %d\n"
           "Matrix count: %d\n"
           "Matrix rows: %d\n"
           "Matrix columns: %d\n",
           idxmeta.dimension, idxmeta.matrix_count, idxmeta.matrix_rows, idxmeta.matrix_columns
          );


   for(uint32_t i = 0; i < idxmeta.matrix_count; i++) {
        pList[i] = fillIDXData(idxmeta.matrix_rows, idxmeta.matrix_columns, pFile);
   }
}

Matrix * fillIDXData(uint32_t rows, uint32_t columns, FILE * pFile) {
    Matrix * matrix = createMatrix(rows, columns);
    if(matrix == NULL) {
        return NULL;
    }
    uint32_t threshold = 127;

    for(uint32_t i = 0; i < rows; i++) {
        for(uint32_t j = 0; j < columns; j++) {
            uint8_t tmp_pixel = 0;
            fread(&tmp_pixel, sizeof(uint8_t), 1, pFile);

            if(tmp_pixel >= threshold) {
                matrix->data[i][j] = 1;
            } else {
                matrix->data[i][j] = 0;
            }
           //matrix->data[i][j] = tmp_pixel;
        }
    }
    return matrix;
}

