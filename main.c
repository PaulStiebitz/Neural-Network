#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "./libs/matrix/matrix.h"
#include "./libs/libidx3/idx3_io.h"

int main(void) {
    const char trainingDataPath[] = "data/training/train-images.idx3-ubyte";

    MatrixList * matrix_list = getIDXDataMatrixList(trainingDataPath);
    Matrix * input = MatrixListToVector(matrix_list);

    printMatrixList(matrix_list, 0, "int");
    freeMatrixList(matrix_list);
    free(matrix_list);
    return 0;
}
