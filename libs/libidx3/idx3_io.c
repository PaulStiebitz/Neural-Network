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
Bytes 4–7: Number of images (32-bit big-endian integer)
Bytes 8–11: Number of rows per image (32-bit big-endian integer)
Bytes 12–15: Number of columns per image (32-bit big-endian integer)
Bytes 16+: Raw pixel data as a continuous stream of unsigned bytes

*/

void getIDXdata(const char * path) {
    if(path == NULL) {
        return;
    }
    FILE *file = fopen(path, "rb");
    IDXMeta meta = readIDXMeta(file);

    if(meta.dimension == 0) {
        return;
    }
    printf("dim: %d\n", meta.bytes);
    if(meta.dimension == 1) {
        printf("1D");
    } else {
        printf("3D");
    }
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
    fread(&idxmeta.image_count, size_uint32_t, 1, pFile);
    fread(&idxmeta.image_rows, size_uint32_t, 1, pFile);
    fread(&idxmeta.image_columns, size_uint32_t, 1, pFile);

    // 8 Bit values are already correct!
    idxmeta.image_count = flip_endian(idxmeta.image_count);
    idxmeta.image_rows = flip_endian(idxmeta.image_rows);
    idxmeta.image_columns = flip_endian(idxmeta.image_columns);
    return idxmeta;
}

/*
void readIDX3Data(const char *path) {
        printf("Magic: %d\nImage count: %d\nImage rows: %d\nImage columns: %d",
            magic, image_count, image_rows, image_columns);

        fclose(file);
        free(file);
   return;
}

void readIDX1Data(char *path) {
    return;
}

*/

