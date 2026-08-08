#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "./libs/matrix/matrix.h"
#include "./libs/libidx3/idx3_io.h"

int main(void) {
    const char trainingDataPath[] = "data/training/train-images.idx3-ubyte";

    MatrixList * matrix_list = getIDXdata(trainingDataPath);
    printMatrixList(matrix_list, 10);
    free(matrix_list);
    return 0;
}

