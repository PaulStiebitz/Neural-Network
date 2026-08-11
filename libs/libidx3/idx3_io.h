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

LabeledMatrixList *getIDXLabeledMatrixList(const char *pData_path, const char *pLabel_path);

IDXMetadata readIDXMetadata(FILE *pIDX_file, char idx_format[]);

void fillIDXLabeledMatrixList(FILE * pData_File, FILE * pLabel_file, LabeledMatrixList * pLabeled_matrix_list);

void fillLabeledMatrix(FILE *pData_file, FILE * pLabel_file, LabeledMatrix *pLabeled_matrix);

#endif