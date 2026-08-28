#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "./libs/matrix/matrix.h"
#include "./libs/libidx3/idx3_io.h"
#include "./libs/nn/nn.h"

int main(void) {
    /* Seed the random number generator with the current time. */
    srand((unsigned int)time(NULL));

    /* File paths for the MNIST training images and labels. */
    const char trainingDataPath[] = "data/training/train-images.idx3-ubyte";
    const char trainingLabelPath[] = "data/training/train-labels.idx1-ubyte";

    /* File paths for the MNIST test images and labels. */
    const char testDataPath[] = "data/testing/t10k-images.idx3-ubyte";
    const char testLabelPath[] = "data/testing/t10k-labels.idx1-ubyte";

    /* Load training images and their labels from the IDX files. */
    printf("[IDXData] Get Training data\n");
    LabeledMatrixList *training_labeled_matrix_list = getIDXLabeledMatrixList(trainingDataPath, trainingLabelPath);
    /* Print the first 10 training images as binary pixel maps. */
    printLabeledMatrixList(training_labeled_matrix_list, 10, "int");

    /* Load test images and their labels from the IDX files. */
    printf("[IDXData] Get Testing data\n");
    LabeledMatrixList *testing_labeled_matrix_list = getIDXLabeledMatrixList(testDataPath, testLabelPath);
    /* Print the first 10 test images as binary pixel maps. */
    printLabeledMatrixList(testing_labeled_matrix_list, 10, "int");

    /*
    MNIST network architecture:
      Input layer  : 784 inputs  (28x28 pixels, flattened)
      Hidden layer 1: 128 neurons (ReLU)
      Hidden layer 2:  64 neurons (ReLU)
      Output layer :  10 neurons (Softmax, one per digit class)
    */
    uint32_t start_input_num = 784;
    uint32_t layer_num = 3;
    uint32_t layerConfig[] = {128, 64, 10};
    float learning_rate = 0.001f;

    /* Number of full passes over the training set. */
    uint32_t num_epochs = 10;

    /* Allocate and initialise the neural network with random weights. */
    printf("[NN] Creating neural network\n");
    NeuralNetwork *mnist_neural_network = createNeuralNetwork(start_input_num, layer_num, layerConfig, learning_rate);

    /* Train the network for the specified number of epochs. */
    for(uint32_t epoch = 0; epoch < num_epochs; epoch++) {
        printf("[NN] Training epoch: %d/%d\n", epoch + 1, num_epochs);
        trainNeuralNetwork(mnist_neural_network, training_labeled_matrix_list);
    }

    /* Evaluate the trained network on the test set and print accuracy. */
    testNeuralNetwork(mnist_neural_network, testing_labeled_matrix_list);

    /* Release all allocated memory. */
    freeLabeledMatrixList(training_labeled_matrix_list);
    freeLabeledMatrixList(testing_labeled_matrix_list);
    freeNeuralNetwork(mnist_neural_network);

    return 0;
}
