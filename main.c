#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "./libs/matrix/matrix.h"
#include "./libs/libidx3/idx3_io.h"

int main(void) {
    const char dataPath[] = "data/training/train-images.idx3-ubyte";
    Matrix ** matrix_list = getIDXdata(dataPath);

    for(int i = 0; i < 10; i++) {
        printMatrix(matrix_list[i], 28, 28);
    }
    free(matrix_list);
    return 0;
}

