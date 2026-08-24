#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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

/* Reverses the byte order of a 32-bit integer (big-endian to little-endian). */
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
/* Opens an IDX3 file at path, reads its metadata and pixel data, returns a MatrixList. Returns NULL on failure. */
LabeledMatrixList * getIDXLabeledMatrixList(const char * pData_Path, const char * pLabel_Path) {
    if(pData_Path == NULL || pLabel_Path == NULL) {
        return NULL;
    }

    FILE *data_file = fopen(pData_Path, "rb");
    FILE *label_file = fopen(pLabel_Path, "rb");
    if(data_file == NULL || label_file == NULL) {
        if(data_file != NULL) fclose(data_file);
        if(label_file != NULL) fclose(label_file);
        return NULL;
    }

    IDXMetadata idx_Data_Metadata = readIDXMetadata(data_file, "IDX3");
    readIDXMetadata(label_file, "IDX1");

    uint32_t data_matrix_list_length = idx_Data_Metadata.matrix_count;
    uint32_t data_matrix_rows = idx_Data_Metadata.matrix_rows;
    uint32_t data_matrix_columns = idx_Data_Metadata.matrix_columns;

    LabeledMatrixList * labeled_matrix_list = createLabeledMatrixList(data_matrix_list_length, data_matrix_rows, data_matrix_columns);
    fillIDXLabeledMatrixList(data_file, label_file, labeled_matrix_list);

    fclose(data_file);
    fclose(label_file);
    return labeled_matrix_list;
}

/* Reads the 16-byte IDX3 file header and returns its fields as IDXMetadata. */

/*
IDX3
Byte 0-1 : 00 00
2
Byte 2 : 08
3
Byte 3 : 03
4
Byte 4-7 : Anzahl Bilder
5
Byte 8-11 : Zeilen
6
Byte 12-15 : Spalten

IDX1
Byte 0-1 : 00 00
2
Byte 2 : 08
3
Byte 3 : 01
4
Byte 4-7 : Anzahl Labels
*/
IDXMetadata readIDXMetadata(FILE *pIDX_file, char idx_format[]) {
    if(pIDX_file == NULL) {
        IDXMetadata emptyMeta = {0};
        return emptyMeta;
    }

    IDXMetadata idxMetadata = {0};

    size_t size_uint32_t = sizeof(uint32_t);
    size_t size_uint8_t = sizeof(uint8_t);

    fread(&idxMetadata.zero_bytes, size_uint8_t, 2, pIDX_file);
    fread(&idxMetadata.bytes, size_uint8_t, 1, pIDX_file);
    fread(&idxMetadata.dimension, size_uint8_t, 1, pIDX_file);
    fread(&idxMetadata.matrix_count, size_uint32_t, 1, pIDX_file);
    if(strcmp(idx_format, "IDX3") == 0) {
        fread(&idxMetadata.matrix_rows, size_uint32_t, 1, pIDX_file);
        fread(&idxMetadata.matrix_columns, size_uint32_t, 1, pIDX_file);
        idxMetadata.matrix_rows = flip_endian(idxMetadata.matrix_rows);
        idxMetadata.matrix_columns = flip_endian(idxMetadata.matrix_columns);
    } else if(strcmp(idx_format, "IDX1") == 0) {
        idxMetadata.matrix_rows = 1;
        idxMetadata.matrix_columns = 1;
    }
    // 1 Byte fields are already correct!
    idxMetadata.matrix_count = flip_endian(idxMetadata.matrix_count);
    printf("[IDXData] Reading:\n"
           "Dimension: %d\n"
           "Matrix count: %d\n"
           "Matrix rows: %d\n"
           "Matrix columns: %d\n\n",
           idxMetadata.dimension, idxMetadata.matrix_count, idxMetadata.matrix_rows, idxMetadata.matrix_columns
          );
    return idxMetadata;
}

/* Reads pixel data from pFile and fills each matrix in pMatrixList. */
void fillIDXLabeledMatrixList(FILE * pData_File, FILE * pLabel_file, LabeledMatrixList * pLabeled_matrix_list) {
    if(pData_File == NULL || pLabel_file == NULL || pLabeled_matrix_list == NULL) {
        return;
    }

    for(uint32_t i = 0; i < pLabeled_matrix_list->list_length; i++) {
        LabeledMatrix * current_labeled_matrix = pLabeled_matrix_list->list[i];
        fillLabeledMatrix(pData_File, pLabel_file, current_labeled_matrix);
        if(pLabeled_matrix_list->list[i] == NULL) {
            for(uint32_t j = 0; j < i; j++) {
                freeLabeledMatrix(pLabeled_matrix_list->list[j]);
                free(pLabeled_matrix_list);
                return;
            }
        }
    }
}

/* Reads one image from pFile and returns it as a binary Matrix (0 or 1 per pixel). */
void fillLabeledMatrix(FILE *pData_file, FILE * pLabel_file, LabeledMatrix *pLabeled_matrix) {
    if(pData_file == NULL || pLabel_file == NULL || pLabeled_matrix == NULL) {
        return;
    }
    uint32_t rows = pLabeled_matrix->matrix->rows;
    uint32_t columns = pLabeled_matrix->matrix->columns;

    uint8_t tmp_label = 0;
    fread(&tmp_label, sizeof(uint8_t), 1, pLabel_file);
    pLabeled_matrix->label = tmp_label;

    uint8_t threshold = 127;
    uint32_t matrixIndex = 0;
    for(uint32_t i = 0; i < rows; i++) {
        for(uint32_t j = 0; j < columns; j++) {
            uint8_t tmp_pixel = 0;
            fread(&tmp_pixel, sizeof(uint8_t), 1, pData_file);
            if(tmp_pixel >= threshold) {
                 pLabeled_matrix->matrix->data[matrixIndex] = 1;
            } else {
                pLabeled_matrix->matrix->data[matrixIndex] = 0;
            }
            matrixIndex++;
        }
    }
}
