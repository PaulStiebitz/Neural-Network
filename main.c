#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "./libs/matrix/matrix.h"
#include "./libs/libidx3/idx3_io.h"
#include "./libs/nn/nn.h"

int main(void) {
    const char trainingDataPath[] = "data/training/train-images.idx3-ubyte";
    const char trainingLabelPath[] = "data/training/train-labels.idx1-ubyte";

    LabeledMatrixList *labeled_matrix_list = getIDXLabeledMatrixList(trainingDataPath, trainingLabelPath);

    /*
    Build Network
    Mnist Networks structure:
    Input Layer:
    784 inputs

    Hidden Layer 1:
    128 Neurons

    Hidden Layer 2:
    64 Neurons

    Output Layer:
    10 Neurons
    */
    uint32_t input_num = 784;
    uint32_t layer_num = 3;
    uint32_t layerConfig = {128, 64, 10};
    NeuralNetwork *mnist_neural_network = buildNeuralNetwork(4, layerConfig);
    // Train Network

    // Test Networke
    printLabeledMatrixList(labeled_matrix_list, 100, "int");
    freeLabeledMatrixList(labeled_matrix_list);

    return 0;
}
