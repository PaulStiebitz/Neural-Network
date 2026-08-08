#ifndef IDX3_IO_H
#define IDX3_IO_H

#include <stdint.h>
#include "../matrix/matrix.h"

typedef struct {
    uint16_t zero_bytes;
    uint32_t bytes;
    uint32_t dimension;
    uint32_t matrix_count;
    uint32_t matrix_rows;
    uint32_t matrix_columns;
} IDXMeta;

uint32_t flip_endian(uint32_t num);

Matrix ** getIDXdata(const char *path);

IDXMeta readIDXMeta(FILE *pFile);

void readIDXData(IDXMeta idxmeta, FILE * pFile, Matrix **pList);

Matrix * fillIDXData(uint32_t rows, uint32_t cols, FILE * pFile);

#endif