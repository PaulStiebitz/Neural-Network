#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "./libs/matrix/matrix.h"
#include "./libs/libidx3/idx3_io.h"
#include "./libs/nn/nn.h"

int main(void) {
    const char trainingDataPath[] = "data/training/train-images.idx3-ubyte";
    const char trainingLabelPath[] = "data/training/train-labels.idx1-ubyte";

    // Load training data and labels
    printf("[IDXData] Get Training data\n");
    LabeledMatrixList *labeled_matrix_list = getIDXLabeledMatrixList(trainingDataPath, trainingLabelPath);
    //printLabeledMatrixList(labeled_matrix_list, 100, "int");
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
    uint32_t start_input_num = 784;
    uint32_t layer_num = 3;
    uint32_t layerConfig[] = {128, 64, 10};

    printf("[NN] Creating neural network\n");
    NeuralNetwork *mnist_neural_network = createNeuralNetwork(start_input_num, layer_num, layerConfig);
    printNeuralNetwork(mnist_neural_network);
    // Train Network

    // Test Network

    // Free components
    freeLabeledMatrixList(labeled_matrix_list);
    freeNeuralNetwork(mnist_neural_network);

    return 0;
}
