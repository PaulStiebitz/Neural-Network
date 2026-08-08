#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "./libs/libidx3/idx3_io.h"

int main(void) {
    const char dataPath[] = "data/training/train-images.idx3-ubyte";
    getIDXdata(dataPath);
    return 0;
}



