#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include "../matrix/matrix.h"

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

void forwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network);
void backwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network);

// Forward Pass
void relu_activation(Vector *pPre_activation, Vector *pReLU_destination);
void softmax(Vector *pActivation, Vector *pReLU_destination);
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

#endif