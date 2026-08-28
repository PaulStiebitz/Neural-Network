#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include "../matrix/matrix.h"

/* Small constant used to clamp activation values before taking log,
   preventing log(0) = -inf in the cross-entropy loss. */
#define EPSILON 1e-15

/* One fully-connected layer of a neural network.
   Stores all per-layer state needed for forward and backward passes. */
typedef struct {
    uint32_t num_inputs;              /* Number of inputs fed into this layer. */
    uint32_t num_neurons;             /* Number of neurons (outputs) in this layer. */
    Matrix  *matrix_weights;          /* Weight matrix W  [num_neurons x num_inputs]. */
    Vector  *vector_biases;           /* Bias vector b    [num_neurons]. */
    Vector  *vector_preactivation_z;  /* Pre-activation z = W*a_prev + b  [num_neurons]. */
    Vector  *vector_activation_a;     /* Activation a = f(z)              [num_neurons]. */
    Vector  *vector_error_delta;      /* Error gradient delta for backprop [num_neurons]. */
} NeuralNetworkLayer;

/* A fully-connected feedforward neural network. */
typedef struct {
    uint32_t            layer;       /* Total number of layers. */
    float               learning_rate;
    NeuralNetworkLayer **layer_list; /* Heap-allocated array of layer pointers. */
} NeuralNetwork;

/* Construction / Destruction */

/* Allocates a NeuralNetworkLayer with random weights and zero biases.
   Returns NULL on any allocation failure. */
NeuralNetworkLayer *createNeuralNetworkLayer(uint32_t pNum_inputs, uint32_t pNum_neurons);

/* Frees all vectors, the weight matrix, and the layer struct itself.
   No-op if pNetwork_layer is NULL. */
void freeNeuralNetworkLayer(NeuralNetworkLayer *pNetwork_layer);

/* Allocates a NeuralNetwork with pLayer_num layers.
   pStart_input_num: number of inputs to the first layer.
   pLayer_config:    array of length pLayer_num giving neuron counts per layer.
   pLearning_rate:   step size used during weight updates.
   Returns NULL on any allocation failure. */
NeuralNetwork *createNeuralNetwork(uint32_t pStart_input_num, uint32_t pLayer_num, uint32_t *pLayer_config, float pLearning_rate);

/* Frees all layers, the layer pointer array, and the network struct itself. */
void freeNeuralNetwork(NeuralNetwork *pNeuralNetwork);

/* Training and Evaluation */

/* Runs one full training epoch over training_labeled_matrix_list.
   Shuffles the data (Fisher-Yates) before each epoch, then performs
   forward + backward pass and weight update for every sample. */
void trainNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *training_labeled_matrix_list);

/* Runs a forward pass on every sample in testing_labeled_matrix_list and
   prints the final classification accuracy. */
void testNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *testing_labeled_matrix_list);

/* Forward Pass/

/* Propagates one sample through the entire network.
   pMode "train": silent.  pMode "test": prints prediction per sample. */
void forwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network,
                 Vector *pInitial_matrix_input_vector, uint32_t pMatrix_num,
                 const char *pMode);

/* Applies ReLU element-wise: a = max(0, z). */
void relu_activation(Vector *pPre_activation, Vector *pReLU_destination);

/* Returns 1 if z > 0, else 0 (derivative of ReLU). */
float reluDerivative(float z);

/* Applies numerically stable Softmax to pActivation and stores the result
   in pSoftmax_destination. */
void softmax(Vector *pActivation, Vector *pSoftmax_destination);

/* Returns the index of the largest element in pSoftmax_vector.
   Returns UINT32_MAX if pSoftmax_vector is NULL. */
uint32_t argmax(Vector *pSoftmax_vector);

/* Backward Pass */

/* Propagates gradients from the output layer back through the network and
   updates all weights and biases in-place. */
void backwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network, Vector *pInitial_matrix_input_vector);

/* Prints the cross-entropy loss for the given sample. */
void printCrossEntropyLoss(uint32_t pActual_number, Vector *pActivation);

/* Computes the combined Softmax + cross-entropy gradient with respect to z
   for the output layer: delta[i] = a[i] - 1 if i == label, else a[i]. */
void softmaxCrossEntropyDerivative(uint32_t pActual_number, Vector *pActivation, Vector *pDestination_delta);

/* Computes the error gradient delta for a hidden layer:
   delta = (W_next^T * delta_next) element-wise-multiplied by ReLU'(z). */
void layerDelta(Matrix *pLast_Weights, Vector *pLast_delta, Vector *pCurrent_z, Vector *pDestination_delta);

/* -- Weight / Bias Updates -- */

/* Allocates and returns the weight gradient matrix dL/dW = delta * a_prev^T. */
Matrix *weightsDerivative(Vector *pDelta, Vector *pPrev_activation);

/* Updates the weight matrix in-place with gradient clipping (L2-norm <= GRADIENT_CLIP):
   W -= learning_rate * scale * dL/dW. */
void updateWeights(float pLearning_rate, Matrix *pCurrent_weights, Vector *pDelta, Vector *pPrev_activation);

/* Updates the bias vector in-place: b -= learning_rate * delta. */
void updateBias(float pLearning_rate, Vector *pCurrent_bias, Vector *pCurrent_delta);

/* Debug Printing */

/* Prints all fields of a single NeuralNetworkLayer. */
void printNeuralNetworkLayer(NeuralNetworkLayer *pNeural_network_layer);

/* Prints every layer of pNeural_network. */
void printNeuralNetwork(NeuralNetwork *pNeural_network);

#endif
