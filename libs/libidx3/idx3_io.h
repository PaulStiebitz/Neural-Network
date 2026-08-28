#ifndef IDX3_IO_H
#define IDX3_IO_H

#include <stdio.h>
#include <stdint.h>
#include "../matrix/matrix.h"

/*
 * Header metadata read from an IDX file.
 *
 * IDX3 layout (images):
 *   Bytes  0-1 : magic zero bytes (0x00 0x00)
 *   Byte   2   : data type code   (0x08 = unsigned byte)
 *   Byte   3   : number of dims   (0x03 for 3-D tensor)
 *   Bytes  4-7 : number of images (big-endian uint32)
 *   Bytes  8-11: rows per image   (big-endian uint32)
 *   Bytes 12-15: cols per image   (big-endian uint32)
 *
 * IDX1 layout (labels):
 *   Bytes  0-1 : magic zero bytes
 *   Byte   2   : data type code   (0x08)
 *   Byte   3   : number of dims   (0x01)
 *   Bytes  4-7 : number of labels (big-endian uint32)
 */
typedef struct {
    uint8_t  zero_bytes[2]; /* Magic zero bytes at the start of every IDX file. */
    uint8_t  bytes;         /* Data type code (0x08 = unsigned byte). */
    uint8_t  dimension;     /* Number of dimensions (1 for labels, 3 for images). */
    uint32_t matrix_count;  /* Total number of matrices / labels in the file. */
    uint32_t matrix_rows;   /* Rows per matrix (1 for IDX1). */
    uint32_t matrix_columns;/* Columns per matrix (1 for IDX1). */
} IDXMetadata;

/* Reverses byte order of a 32-bit integer (big-endian to little-endian). */
uint32_t flip_endian(uint32_t num);

/* Opens the IDX3 image file and IDX1 label file, reads all images and labels,
   and returns a fully populated LabeledMatrixList. Returns NULL on failure. */
LabeledMatrixList *getIDXLabeledMatrixList(const char *pData_path, const char *pLabel_path);

/* Reads the IDX file header and returns the parsed metadata.
   idx_format must be "IDX3" (images) or "IDX1" (labels). */
IDXMetadata readIDXMetadata(FILE *pIDX_file, char idx_format[]);

/* Iterates over all entries in pLabeled_matrix_list and fills each one
   by reading pixel data from pData_File and a label byte from pLabel_file. */
void fillIDXLabeledMatrixList(FILE *pData_File, FILE *pLabel_file, LabeledMatrixList *pLabeled_matrix_list);

/* Reads one label byte and one image (rows x columns pixels) from the respective
   files and stores them in pLabeled_matrix. */
void fillLabeledMatrix(FILE *pData_file, FILE *pLabel_file, LabeledMatrix *pLabeled_matrix);

#endif
