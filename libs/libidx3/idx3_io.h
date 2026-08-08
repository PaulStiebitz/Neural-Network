#ifndef IDX3_IO_H
#define IDX3_IO_H

#include <stdint.h>

typedef struct {
    uint16_t zero_bytes;
    uint32_t bytes;
    uint32_t dimension;
    uint32_t image_count;
    uint32_t image_rows;
    uint32_t image_columns;
} IDXMeta;

uint32_t flip_endian(uint32_t num);

void getIDXdata(const char *path);

IDXMeta readIDXMeta(FILE *pFile);

void readIDX3Data(const char *path);

#endif