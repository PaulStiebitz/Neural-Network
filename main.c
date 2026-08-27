#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "./libs/matrix/matrix.h"
#include "./libs/libidx3/idx3_io.h"
#include "./libs/nn/nn.h"

int main(void) {
    srand((unsigned int)time(NULL));
    // Training data
    const char trainingDataPath[] = "data/training/train-images.idx3-ubyte";
    const char trainingLabelPath[] = "data/training/train-labels.idx1-ubyte";

    // Test Data
    const char testDataPath[] = "data/testing/t10k-images.idx3-ubyte";
    const char testLabelPath[] = "data/testing/t10k-labels.idx1-ubyte";

    // Load training data and labels
    printf("[IDXData] Get Training data\n");
    LabeledMatrixList *training_labeled_matrix_list = getIDXLabeledMatrixList(trainingDataPath, trainingLabelPath);
    printLabeledMatrixList(training_labeled_matrix_list, 10, "int");

    // Load testing data and labels
    printf("[IDXData] Get Testing data\n");
    LabeledMatrixList *testing_labeled_matrix_list = getIDXLabeledMatrixList(testDataPath, testLabelPath);
    printLabeledMatrixList(testing_labeled_matrix_list, 10, "int");

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
    float learning_rate = 0.001f;

    uint32_t num_epochs = 10;

    printf("[NN] Creating neural network\n");
    NeuralNetwork *mnist_neural_network = createNeuralNetwork(start_input_num, layer_num, layerConfig, learning_rate);

    // Train Network
    for(uint32_t epoch = 0; epoch < num_epochs; epoch++) {
        printf("[NN] Training epoch: %d/%d\n", epoch + 1, num_epochs);
        trainNeuralNetwork(mnist_neural_network, training_labeled_matrix_list);
    }

    // Test Network
    testNeuralNetwork(mnist_neural_network, testing_labeled_matrix_list);
    // Free components
    freeLabeledMatrixList(training_labeled_matrix_list);
    freeLabeledMatrixList(testing_labeled_matrix_list);
    freeNeuralNetwork(mnist_neural_network);

    return 0;
}
