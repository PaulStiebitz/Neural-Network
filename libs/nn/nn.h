#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include "../matrix/matrix.h"

typedef struct {
    uint32_t layer;
    NeuralNetworkLayer *layer_list;
} NeuralNetwork;

typedef struct {
    uint32_t num_inputs;
    uint32_t num_neurons;
    Matrix *matrix_weights;
    Vector *vector_biases;
    Vector vector_preactivation_z;
    Vector vector_activation_a;
    Vector vector_error_delta;
} NeuralNetworkLayer;

/*
Build the neural network with the folling config array:
pLayers: num of total layers
pLayerConfig: the first value indicates the number of inputs.
The remanining array indicitates the total number of neurons for the i-th layer, meanwhile
the index i represents also the index of the array.
*/
NeuralNetwork *buildNeuralNetwork(uint32_t pStart_input_num, uint32_t pLayer_num, uint32_t *pLayer_config);
NeuralNetworkLayer *buildNeuralNetworkLayer(uint32_t pNum_inputs, uint32_t pNum_neurons);

// Free components from memory
void freeNeuralNetwork(NeuralNetwork *pNeuralNetwork);

// Forward Pass
Vector *layerPreActivation(Matrix *pWeights, Vector *pLayer_input, Vector *pLayer_bias);
Vector *relu_activation(Vector *pPre_activation);
Vector *softmax(Vector *pPre_activation);
uint32_t argmax(Vector *pProb);

// Backward Pass
Vector *crossEntropyLoss(Vector *pProb, Vector *pTarget);
// Cross Entropy loss derivative + softmax derivative in respect to z[3].
Vector *ouputLayerDelta(Vector *pProb, Vector *pTarget);
// Generic layerDelta calculcation for all layers.
Vector *layerDelta(Matrix *pWeights, Vector *pLast_delta, Vector *pForward_layer_preactivation_z);
Vector *reluDerivative(Vector *pForward_layer_preactivation_z);
Vector *weightsLossDerivative(Vector *pLast_delta, Vector *pLayer_input);

// Update weights and biases
void updateWeights(Matrix *pCurrent_weights, Matrix *pWeightsLossDerivative);
void updateBias(Vector *pCurrent_bias, Vector *pCurrent_delta);

// Training and testing the neural network
void trainNeuralNetwork();
void testNeuralNetwork();

#endif