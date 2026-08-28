#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "idx3_io.h"
#include "../matrix/matrix.h"

/*
 * flip_endian, byte-reversal diagram
 *
 * Goal: convert  A B C D to D C B A
 *
 * Step 1: num >> 24: 0 0 0 A      (move byte A to lowest position)
 * Step 2: num << 8:  B C D 0      (shift left; mask 0xff0000 keeps byte 2)
 * Step 3: num >> 8:  0 0 B C      (shift right; mask 0xff00 keeps byte 1)
 * Step 4: num << 24: D 0 0 0      (move byte D to highest position)
 *
 * OR-ing all four partial results gives D C B A.
 */

/* Reverses the byte order of a 32-bit integer (big-endian to little-endian).
   IDX files store multi-byte integers in big-endian order; this function
   converts them to the host's little-endian representation. */
uint32_t flip_endian(uint32_t num) {
    return ((num >> 24) & 0xff)       |  /* byte 3 to byte 0 */
           ((num << 8)  & 0xff0000)   |  /* byte 1 to byte 2 */
           ((num >> 8)  & 0xff00)     |  /* byte 2 to byte 1 */
           ((num << 24) & 0xff000000);   /* byte 0 to byte 3 */
}

/* Opens the IDX3 image file at pData_Path and the IDX1 label file at
   pLabel_Path, reads their headers, allocates a LabeledMatrixList of the
   correct size, and fills it with pixel data and labels.
   Returns NULL if either path is NULL or a file cannot be opened. */
LabeledMatrixList *getIDXLabeledMatrixList(const char *pData_Path, const char *pLabel_Path) {
    if(pData_Path == NULL || pLabel_Path == NULL) {
        return NULL;
    }

    FILE *data_file  = fopen(pData_Path,  "rb");
    FILE *label_file = fopen(pLabel_Path, "rb");
    if(data_file == NULL || label_file == NULL) {
        /* Close whichever file was successfully opened before returning. */
        if(data_file  != NULL) fclose(data_file);
        if(label_file != NULL) fclose(label_file);
        return NULL;
    }

    /* Parse the IDX3 header to obtain image count, rows, and columns. */
    IDXMetadata idx_Data_Metadata = readIDXMetadata(data_file, "IDX3");
    /* Advance the label file's read cursor past its IDX1 header. */
    readIDXMetadata(label_file, "IDX1");

    uint32_t data_matrix_list_length = idx_Data_Metadata.matrix_count;
    uint32_t data_matrix_rows = idx_Data_Metadata.matrix_rows;
    uint32_t data_matrix_columns = idx_Data_Metadata.matrix_columns;

    /* Allocate a list of empty LabeledMatrix structs with the correct dimensions. */
    LabeledMatrixList *labeled_matrix_list = createLabeledMatrixList(data_matrix_list_length, data_matrix_rows, data_matrix_columns);

    /* Read pixel data and labels into every entry of the list. */
    fillIDXLabeledMatrixList(data_file, label_file, labeled_matrix_list);

    fclose(data_file);
    fclose(label_file);
    return labeled_matrix_list;
}

/* Reads the IDX file header at the current file position and returns
   the parsed fields as an IDXMetadata struct.
   idx_format must be "IDX3" (image file) or "IDX1" (label file).
   For IDX3: row and column counts are read and converted from big-endian.
   For IDX1: rows and columns are set to 1 (each label is a scalar). */
IDXMetadata readIDXMetadata(FILE *pIDX_file, char idx_format[]) {
    if(pIDX_file == NULL) {
        IDXMetadata emptyMeta = {0};
        return emptyMeta;
    }

    IDXMetadata idxMetadata = {0};

    size_t size_uint32_t = sizeof(uint32_t);
    size_t size_uint8_t  = sizeof(uint8_t);

    /* Bytes 0-1: magic zero bytes. */
    fread(idxMetadata.zero_bytes, size_uint8_t,  2, pIDX_file);
    /* Byte 2: data type code (0x08 = unsigned byte). */
    fread(&idxMetadata.bytes, size_uint8_t,  1, pIDX_file);
    /* Byte 3: number of dimensions. */
    fread(&idxMetadata.dimension, size_uint8_t,  1, pIDX_file);
    /* Bytes 4-7: item count stored in big-endian. */
    fread(&idxMetadata.matrix_count, size_uint32_t, 1, pIDX_file);

    if(strcmp(idx_format, "IDX3") == 0) {
        /* Bytes 8-11 and 12-15: row and column counts (big-endian). */
        fread(&idxMetadata.matrix_rows, size_uint32_t, 1, pIDX_file);
        fread(&idxMetadata.matrix_columns, size_uint32_t, 1, pIDX_file);
        idxMetadata.matrix_rows = flip_endian(idxMetadata.matrix_rows);
        idxMetadata.matrix_columns = flip_endian(idxMetadata.matrix_columns);
    } else if(strcmp(idx_format, "IDX1") == 0) {
        /* Labels are scalarsm treat each as a 1x1 matrix. */
        idxMetadata.matrix_rows    = 1;
        idxMetadata.matrix_columns = 1;
    }

    /* Single-byte fields do not need endian conversion. */
    idxMetadata.matrix_count = flip_endian(idxMetadata.matrix_count);

    printf("[IDXData] Reading:\n"
           "Dimension: %d\n"
           "Matrix count: %d\n"
           "Matrix rows: %d\n"
           "Matrix columns: %d\n\n",
           idxMetadata.dimension, idxMetadata.matrix_count,
           idxMetadata.matrix_rows, idxMetadata.matrix_columns);

    return idxMetadata;
}

/* Iterates over every slot in pLabeled_matrix_list and fills it by reading
   one label byte and one image worth of pixels from the respective files. */
void fillIDXLabeledMatrixList(FILE *pData_File, FILE *pLabel_file, LabeledMatrixList *pLabeled_matrix_list) {
    if(pData_File == NULL || pLabel_file == NULL || pLabeled_matrix_list == NULL) {
        return;
    }

    for(uint32_t i = 0; i < pLabeled_matrix_list->list_length; i++) {
        LabeledMatrix *current_labeled_matrix = pLabeled_matrix_list->list[i];
        fillLabeledMatrix(pData_File, pLabel_file, current_labeled_matrix);
    }
}

/* Reads one label byte from pLabel_file and stores it in pLabeled_matrix->label.
   Then reads rows * columns pixel bytes from pData_file and stores each as a
   float in pLabeled_matrix->matrix->data (row-major order). */
void fillLabeledMatrix(FILE *pData_file, FILE *pLabel_file, LabeledMatrix *pLabeled_matrix) {
    if(pData_file == NULL || pLabel_file == NULL || pLabeled_matrix == NULL) {
        return;
    }

    uint32_t rows = pLabeled_matrix->matrix->rows;
    uint32_t columns = pLabeled_matrix->matrix->columns;

    /* Read the single-byte class label (0-9 for MNIST). */
    uint8_t tmp_label = 0;
    fread(&tmp_label, sizeof(uint8_t), 1, pLabel_file);
    pLabeled_matrix->label = tmp_label;

    /* Read every pixel row by row and store as float. */
    uint32_t matrixIndex = 0;
    for(uint32_t i = 0; i < rows; i++) {
        for(uint32_t j = 0; j < columns; j++) {
            uint8_t tmp_pixel = 0;
            fread(&tmp_pixel, sizeof(uint8_t), 1, pData_file);
            pLabeled_matrix->matrix->data[matrixIndex] = (float)tmp_pixel;
            matrixIndex++;
        }
    }
}
