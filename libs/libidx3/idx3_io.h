#ifndef IDX3_IO_H
#define IDX3_IO_H

#include <stdio.h>
#include <stdint.h>
#include "../matrix/matrix.h"

typedef struct {
    uint16_t zero_bytes;
    uint32_t bytes;
    uint32_t dimension;
    uint32_t matrix_count;
    uint32_t matrix_rows;
    uint32_t matrix_columns;
} IDXMetadata;

uint32_t flip_endian(uint32_t num);

MatrixList * getIDXDataMatrixList(const char * path);

IDXMetadata readIDXMetadata(FILE *pFile);

void createIDXMatrixList(FILE * pFile, MatrixList * pMatrixList);

Matrix * createIDXDataMatrix(uint32_t rows, uint32_t columns, FILE * pFile);

#endif