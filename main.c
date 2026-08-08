#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


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
void readData() {
    FILE *file = fopen("./data/testing/t10k-images.idx3-ubyte", "rb");

    uint32_t magic = 0;
    uint32_t image_count = 0;
    uint32_t image_rows = 0;
    uint32_t image_columns = 0;

    size_t size_uint32_t = sizeof(uint32_t);
    fread(&magic, size_uint32_t, 1, file);
    fread(&image_count, size_uint32_t, 1, file);
    fread(&image_rows, size_uint32_t, 1, file);
    fread(&image_columns, size_uint32_t, 1, file);

    magic = flip_endian(magic);
    image_count = flip_endian(image_count);
    image_rows = flip_endian(image_rows);
    image_columns = flip_endian(image_columns);


    printf("Magic: %d\nImage count: %d\nImage rows: %d\nImage columns: %d",
           magic, image_count, image_rows, image_columns);

    fclose(file);
    free(file);
}

int main(void) {
    readData();
    return 0;
}



