#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include "../matrix/matrix.h"

#define EPSILON 1e-15

typedef struct {
    uint32_t num_inputs;
    uint32_t num_neurons;
    Matrix *matrix_weights;
    Vector *vector_biases;
    Vector *vector_preactivation_z;
    Vector *vector_activation_a;
    Vector *vector_error_delta;
} NeuralNetworkLayer;

typedef struct {
    uint32_t layer;
    float learning_rate;
    NeuralNetworkLayer **layer_list;
} NeuralNetwork;

/*
Build the neural network with the folling config array:
pLayers: number of total layers
pLayerConfig: number of neurons in the i-th layer of the config array
*/
NeuralNetworkLayer *createNeuralNetworkLayer(uint32_t pNum_inputs, uint32_t pNum_neurons);
void trainNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *training_labeled_matrix_list);
void testNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *testing_labeled_matrix_list);
void printNeuralNetwork(NeuralNetwork *pNeural_network);
void freeNeuralNetwork(NeuralNetwork *pNeuralNetwork);

NeuralNetwork *createNeuralNetwork(uint32_t pStart_input_num, uint32_t pLayer_num, uint32_t *pLayer_config, float pLearning_rate);
void printNeuralNetworkLayer(NeuralNetworkLayer *pNeural_network_layer);
void freeNeuralNetworkLayer(NeuralNetworkLayer *pNetwork_layer);

void forwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network, Vector *pInitial_matrix_input_vector, uint32_t pMatrix_num, const char *pMode);
void backwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network, Vector *pInitial_matrix_input_vector, uint32_t pMatrix_num);

// Forward Pass
void relu_activation(Vector *pPre_activation, Vector *pReLU_destination);
float reluDerivative(float z);
void softmax(Vector *pActivation, Vector *pReLU_destination);
uint32_t argmax(Vector *pSoftmax_vector);

// Backward Pass
void printCrossEntropyLoss(uint32_t pActual_number, Vector *pActivation);
// Cross Entropy loss derivative + softmax derivative in respect to z[3].
void softmaxCrossEntropyDerivative(uint32_t pActual_number, Vector *pActivation, Vector *pDestination_delta);
// Generic layerDelta calculcation for all layers.
void layerDelta(Matrix *pLast_Weights, Vector *pLast_delta, Vector *pCurrent_z, Vector *pDestination_delta);

// Update weights and biases
Matrix *weightsDerivative(Vector *pDelta, Vector *pPrev_activation);
void updateWeights(float pLearning_rate, Matrix *pCurrent_weights, Vector *pDelta, Vector *pPrev_activation);
void updateBias(float pLearning_rate, Vector *pCurrent_bias, Vector *pCurrent_delta);

#endif