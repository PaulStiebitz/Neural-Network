#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "./libs/matrix/matrix.h"
#include "./libs/libidx3/idx3_io.h"

int main(void) {
    const char trainingDataPath[] = "data/training/train-images.idx3-ubyte";
    const char trainingLabelPath[] = "data/training/train-labels.idx1-ubyte";

    LabeledMatrixList * labeled_matrix_list = getIDXLabeledMatrixList(trainingDataPath, trainingLabelPath);

    printLabeledMatrixList(labeled_matrix_list, 100, "int");
    freeLabeledMatrixList(labeled_matrix_list);

    return 0;
}
