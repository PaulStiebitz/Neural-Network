#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nn.h"
#include "../matrix/matrix.h"

/* Maximum L2-norm allowed for a weight gradient matrix before scaling it down. */
#define GRADIENT_CLIP 1.0f

/* Allocates a NeuralNetworkLayer with pNum_inputs inputs and pNum_neurons neurons.
   Initialises the weight matrix with small random values and all vectors to zero.
   Uses goto for cleanup so every failure path frees whatever was already allocated.
   Returns NULL on any allocation failure. */
NeuralNetworkLayer *createNeuralNetworkLayer(uint32_t pNum_inputs, uint32_t pNum_neurons) {
    NeuralNetworkLayer *neural_network_layer = calloc(1, sizeof(NeuralNetworkLayer));
    if(neural_network_layer == NULL) {
        return NULL;
    }
    neural_network_layer->num_inputs  = pNum_inputs;
    neural_network_layer->num_neurons = pNum_neurons;

    /* Weight matrix: num_neurons rows, num_inputs columns. */
    neural_network_layer->matrix_weights = createMatrix(pNum_neurons, pNum_inputs);
    if(neural_network_layer->matrix_weights == NULL)
        goto cleanup;
    fillMatrixRandom(neural_network_layer->matrix_weights);

    /* Bias vector, one entry per neuron (zero-initialised). */
    neural_network_layer->vector_biases = createVector(pNum_neurons);
    if(neural_network_layer->vector_biases == NULL)
        goto cleanup;

    /* Pre-activation z = W*a + b, computed during forward pass. */
    neural_network_layer->vector_preactivation_z = createVector(pNum_neurons);
    if(neural_network_layer->vector_preactivation_z == NULL)
        goto cleanup;

    /* Post-activation a = f(z), computed during forward pass. */
    neural_network_layer->vector_activation_a = createVector(pNum_neurons);
    if(neural_network_layer->vector_activation_a == NULL)
        goto cleanup;

    /* Error delta used during the backward pass. */
    neural_network_layer->vector_error_delta = createVector(pNum_neurons);
    if(neural_network_layer->vector_error_delta == NULL)
        goto cleanup;

    return neural_network_layer;

    cleanup:
        freeNeuralNetworkLayer(neural_network_layer);
        return NULL;
}

/* Frees all member vectors, the weight matrix, and the layer struct itself.
   No-op if pNetwork_layer is NULL. */
void freeNeuralNetworkLayer(NeuralNetworkLayer *pNetwork_layer) {
    if(pNetwork_layer == NULL) {
        return;
    }
    freeMatrix(pNetwork_layer->matrix_weights);
    freeVector(pNetwork_layer->vector_biases);
    freeVector(pNetwork_layer->vector_preactivation_z);
    freeVector(pNetwork_layer->vector_activation_a);
    freeVector(pNetwork_layer->vector_error_delta);
    free(pNetwork_layer);
}

/* Allocates a NeuralNetwork with pLayer_num fully-connected layers.
   pStart_input_num : number of raw inputs fed to layer 0.
   pLayer_config    : array of pLayer_num neuron counts, one per layer.
   pLearning_rate   : gradient-descent step size.
   On failure all already-created layers are freed before returning NULL. */
NeuralNetwork *createNeuralNetwork(uint32_t pStart_input_num, uint32_t pLayer_num, uint32_t *pLayer_config, float pLearning_rate) {
    NeuralNetwork *neural_network = malloc(sizeof(NeuralNetwork));
    if(neural_network == NULL) {
        return NULL;
    }
    neural_network->layer         = pLayer_num;
    neural_network->learning_rate = pLearning_rate;

    neural_network->layer_list = malloc(pLayer_num * sizeof(NeuralNetworkLayer *));
    if(neural_network->layer_list == NULL) {
        free(neural_network);
        return NULL;
    }

    for(uint32_t i = 0; i < pLayer_num; i++) {
        uint32_t num_neurons = pLayer_config[i];
        /* Layer 0 receives the raw input; every other layer receives the previous layer's output. */
        uint32_t num_inputs  = (i == 0) ? pStart_input_num : pLayer_config[i - 1];
        neural_network->layer_list[i] = createNeuralNetworkLayer(num_inputs, num_neurons);

        if(neural_network->layer_list[i] == NULL) {
            /* Free every successfully created layer before bailing out. */
            for(uint32_t j = 0; j < i; j++) {
                freeNeuralNetworkLayer(neural_network->layer_list[j]);
            }
            free(neural_network->layer_list);
            free(neural_network);
            return NULL;
        }
    }
    return neural_network;
}

/* Frees all layers, the layer pointer array, and the network struct itself.
   No-op if pNeural_network is NULL. */
void freeNeuralNetwork(NeuralNetwork *pNeural_network) {
    if(pNeural_network == NULL) {
        return;
    }
    for(uint32_t i = 0; i < pNeural_network->layer; i++) {
        freeNeuralNetworkLayer(pNeural_network->layer_list[i]);
    }
    free(pNeural_network->layer_list);
    free(pNeural_network);
}

/* Prints all weights, biases, and intermediate vectors of a single layer. */
void printNeuralNetworkLayer(NeuralNetworkLayer *pNeural_network_layer) {
    printf("[NNL] num_inputs: %d\n",  pNeural_network_layer->num_inputs);
    printf("[NNL] num_neurons: %d\n", pNeural_network_layer->num_neurons);
    printf("[NNL] matrix_weights:\n");
    printMatrix(pNeural_network_layer->matrix_weights, "float");
    printf("[NNL] vector_biases:\n");
    printVector(pNeural_network_layer->vector_biases);
    printf("[NNL] vector_preactivation_z:\n");
    printVector(pNeural_network_layer->vector_preactivation_z);
    printf("[NNL] vector_activation_a:\n");
    printVector(pNeural_network_layer->vector_activation_a);
    printf("[NNL] vector_error_delta:\n");
    printVector(pNeural_network_layer->vector_error_delta);
}

/* Prints a summary of every layer in the network. */
void printNeuralNetwork(NeuralNetwork *pNeural_network) {
    printf("[NN] Printing neural network\n");
    printf("[NN] Total layers: %d\n", pNeural_network->layer);
    for(uint32_t i = 0; i < pNeural_network->layer; i++) {
        printf("[NN] Printing Layer: %d\n", i + 1);
        printNeuralNetworkLayer(pNeural_network->layer_list[i]);
    }
    printf("\n");
}

/* Runs a forward pass on every sample in testing_labeled_matrix_list,
   counts correct predictions, and prints the overall accuracy. */
void testNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *testing_labeled_matrix_list) {
    uint32_t correct = 0;
    for(uint32_t matrix_num = 0; matrix_num < testing_labeled_matrix_list->list_length; matrix_num++) {
        Matrix *current_matrix = testing_labeled_matrix_list->list[matrix_num]->matrix;
        /* Flatten the 2-D image into a 1-D input vector. */
        Vector *initial_matrix_input = matrixToVector(current_matrix);
        if(initial_matrix_input == NULL) {
            return;
        }
        forwardPass(testing_labeled_matrix_list->list[matrix_num], pNeural_network, initial_matrix_input, matrix_num, "test");

        /* Compare predicted class (argmax of output layer) against the ground truth. */
        NeuralNetworkLayer *last_layer = pNeural_network->layer_list[pNeural_network->layer - 1];
        uint32_t predicted = argmax(last_layer->vector_activation_a);
        if(predicted == testing_labeled_matrix_list->list[matrix_num]->label) {
            correct++;
        }
        freeVector(initial_matrix_input);
    }
    printf("[Test] Accuracy: %d/%d (%.2f%%)\n", correct, testing_labeled_matrix_list->list_length, 100.0f * (float)correct / (float)testing_labeled_matrix_list->list_length);
}

/* Trains the network for one epoch over training_labeled_matrix_list.
   Applies a Fisher-Yates shuffle before iterating to reduce order bias,
   then performs forward pass + backward pass for every sample. */
void trainNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *training_labeled_matrix_list) {
    /* Fisher-Yates shuffle: randomise sample order for this epoch. */
    uint32_t list_length = training_labeled_matrix_list->list_length;
    for(uint32_t i = list_length - 1; i > 0; i--) {
        uint32_t j = (uint32_t)rand() % (i + 1);
        LabeledMatrix *tmp = training_labeled_matrix_list->list[i];
        training_labeled_matrix_list->list[i] = training_labeled_matrix_list->list[j];
        training_labeled_matrix_list->list[j] = tmp;
    }

    for(uint32_t matrix_num = 0; matrix_num < training_labeled_matrix_list->list_length; matrix_num++) {
        Matrix *current_matrix = training_labeled_matrix_list->list[matrix_num]->matrix;
        /* Flatten the image to a vector for layer input. */
        Vector *initial_matrix_input = matrixToVector(current_matrix);
        if(initial_matrix_input == NULL) {
            return;
        }
        forwardPass(training_labeled_matrix_list->list[matrix_num], pNeural_network, initial_matrix_input, matrix_num, "train");
        backwardPass(training_labeled_matrix_list->list[matrix_num], pNeural_network, initial_matrix_input);
        freeVector(initial_matrix_input);
    }
}

/* Propagates pInitial_matrix_input_vector forward through every layer of the network.
   Layer 0: normalises the raw pixel input to [0, 1] before multiplying by weights.
   Hidden layers: apply ReLU activation.
   Output layer:  applies Softmax; prints the prediction when pMode == "test". */
void forwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network, Vector *pInitial_matrix_input_vector, uint32_t pMatrix_num, const char *pMode) {
    uint32_t num_layers = pNeural_network->layer;
    for(uint32_t i = 0; i < num_layers; i++) {
        NeuralNetworkLayer *current_network_layer  = pNeural_network->layer_list[i];
        Matrix *current_layer_weights = current_network_layer->matrix_weights;
        Vector *current_layer_biases = current_network_layer->vector_biases;
        Vector *current_layer_z = current_network_layer->vector_preactivation_z;
        Vector *current_layer_a = current_network_layer->vector_activation_a;

        if(i == 0) {
            /* Normalise raw pixel values from [0, 255] to [0, 1]. */
            normalizeVector(pInitial_matrix_input_vector, 255.0f);
            matrixTimesVector(current_layer_weights, pInitial_matrix_input_vector, current_layer_z);
        } else {
            /* Use the previous layer's activation as input. */
            Vector *prev_a = pNeural_network->layer_list[i - 1]->vector_activation_a;
            matrixTimesVector(current_layer_weights, prev_a, current_layer_z);
        }

        /* Add bias: z = W*a + b. */
        vectorPlusVector(current_layer_biases, current_layer_z);

        if(i == num_layers - 1) {
            /* Output layer: Softmax produces a probability distribution. */
            softmax(current_layer_z, current_layer_a);
            uint32_t predicted_num = argmax(current_layer_a);
            if(predicted_num == UINT32_MAX) {
                return;
            }
            float predicted_prob = current_layer_a->data[predicted_num];
            uint32_t actual_number  = pLabeled_matrix->label;
            if(strcmp(pMode, "test") == 0) {
                printf("[Pred] %d.Picture Predicted Number: %d Prob: %.3f, Actual number: %d\n", pMatrix_num, predicted_num, predicted_prob, actual_number);
            }
        } else {
            /* Hidden layer: ReLU activation. */
            relu_activation(current_layer_z, current_layer_a);
        }
    }
}

/* Runs the backward pass from the output layer to layer 0.
   Computes per-layer error deltas and updates weights and biases in-place. */
void backwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network, Vector *pInitial_matrix_input_vector) {
    int32_t last_layer = (int32_t)pNeural_network->layer - 1;
    for(int32_t i = last_layer; i >= 0; i--) {
        NeuralNetworkLayer *current_layer = pNeural_network->layer_list[i];
        Vector *current_delta = current_layer->vector_error_delta;

        if(i == last_layer) {
            /* Output layer: combined Softmax + cross-entropy gradient. */
            softmaxCrossEntropyDerivative(pLabeled_matrix->label, current_layer->vector_activation_a, current_delta);
        } else {
            /* Hidden layer: propagate gradient from the next layer. */
            NeuralNetworkLayer *next_layer = pNeural_network->layer_list[i + 1];
            layerDelta(next_layer->matrix_weights, next_layer->vector_error_delta, current_layer->vector_preactivation_z, current_delta);
        }

        /* The activation feeding into this layer: raw input for layer 0, else previous a. */
        Vector *prev_activation = (i == 0) ? pInitial_matrix_input_vector : pNeural_network->layer_list[i - 1]->vector_activation_a;
        updateWeights(pNeural_network->learning_rate, current_layer->matrix_weights, current_delta, prev_activation);
        updateBias(pNeural_network->learning_rate, current_layer->vector_biases, current_delta);
    }
}

/* Applies ReLU element-wise: output[i] = max(0, input[i]).
   No-op if either pointer is NULL or lengths differ. */
void relu_activation(Vector *pPre_activation, Vector *pReLU_destination) {
    if(pPre_activation == NULL || pReLU_destination == NULL) {
        return;
    }
    if(pPre_activation->rows != pReLU_destination->rows) {
        return;
    }
    uint32_t vector_length = pPre_activation->rows;
    for(uint32_t i = 0; i < vector_length; i++) {
        pReLU_destination->data[i] = (pPre_activation->data[i] > 0) ? pPre_activation->data[i] : 0.0f;
    }
}

/* Computes numerically stable Softmax:
   1. Subtract max(z) from every element to prevent overflow.
   2. Exponentiate and normalise by the sum.
   No-op if either pointer is NULL or lengths differ. */
void softmax(Vector *pActivation, Vector *pSoftmax_destination) {
    if(pActivation == NULL || pSoftmax_destination == NULL) {
        return;
    }
    if(pActivation->rows != pSoftmax_destination->rows) {
        return;
    }

    /* Find the maximum value for numerical stability. */
    float max_val = pActivation->data[0];
    for(uint32_t i = 1; i < pActivation->rows; i++) {
        if(pActivation->data[i] > max_val) {
            max_val = pActivation->data[i];
        }
    }

    /* Compute the normalisation denominator. */
    float sum = 0.0f;
    for(uint32_t j = 0; j < pActivation->rows; j++) {
        sum += expf(pActivation->data[j] - max_val);
    }

    /* Write the final probabilities. */
    for(uint32_t i = 0; i < pActivation->rows; i++) {
        pSoftmax_destination->data[i] = expf(pActivation->data[i] - max_val) / sum;
    }
}

/* Returns the index of the largest value in pSoftmax_vector (the predicted class).
   Returns UINT32_MAX if pSoftmax_vector is NULL. */
uint32_t argmax(Vector *pSoftmax_vector) {
    if(pSoftmax_vector == NULL) {
        return UINT32_MAX;
    }
    float max_prob_value = 0.0f;
    uint32_t max_prob_index = 0;
    for(uint32_t i = 0; i < pSoftmax_vector->rows; i++) {
        float current_prob_value = pSoftmax_vector->data[i];
        if(current_prob_value > max_prob_value) {
            max_prob_value = current_prob_value;
            max_prob_index = i;
        }
    }
    return max_prob_index;
}

/* Prints the cross-entropy loss  L = -log(a[pActual_number])  for one sample.
   Clamps the activation to EPSILON before taking the log to avoid -inf. */
void printCrossEntropyLoss(uint32_t pActual_number, Vector *pActivation) {
    if(pActivation == NULL || pActual_number >= pActivation->rows) {
        return;
    }
    double activation_val = pActivation->data[pActual_number];
    if(activation_val < EPSILON) {
        activation_val = EPSILON;
    }
    double loss = -log(activation_val);
    printf("Loss: %.3f\n", loss);
}

/* Computes the gradient of the combined Softmax + cross-entropy loss
   with respect to the pre-activation z of the output layer:
     delta[i] = a[i] - 1   if i == pActual_number  (correct class)
     delta[i] = a[i]        otherwise               (all other classes)
   No-op if either pointer is NULL or lengths differ. */
void softmaxCrossEntropyDerivative(uint32_t pActual_number, Vector *pActivation, Vector *pDestination_delta) {
    if(pActivation == NULL || pDestination_delta == NULL) {
        return;
    }
    if(pActivation->rows != pDestination_delta->rows) {
        return;
    }
    for(uint32_t i = 0; i < pActivation->rows; i++) {
        if(i == pActual_number) {
            pDestination_delta->data[i] = pActivation->data[i] - 1.0f;
        } else {
            pDestination_delta->data[i] = pActivation->data[i];
        }
    }
}

/* Returns the ReLU derivative: 1 if z > 0, else 0. */
float reluDerivative(float z) {
    return (z > 0.0f) ? 1.0f : 0.0f;
}

/* Computes the error delta for a hidden layer:
   delta = (W_next^T * delta_next) element-wise * ReLU'(z_current).
   No-op if any pointer is NULL. */
void layerDelta(Matrix *pLast_Weights, Vector *pLast_delta, Vector *pCurrent_z,    Vector *pDestination_delta) {
    if(pLast_Weights == NULL || pLast_delta == NULL || pCurrent_z == NULL || pDestination_delta == NULL) {
        return;
    }
    /* Backpropagate the gradient through the transposed weight matrix. */
    matrixTransposeTimesVector(pLast_Weights, pLast_delta, pDestination_delta);
    /* Multiply element-wise by the ReLU derivative of the current layer's pre-activation. */
    for(uint32_t i = 0; i < pDestination_delta->rows; i++) {
        pDestination_delta->data[i] *= reluDerivative(pCurrent_z->data[i]);
    }
}

/* Allocates and returns the weight gradient matrix dL/dW = delta * a_prev^T.
   Shape: [delta.rows x pPrev_activation.rows].
   Returns NULL if either pointer is NULL or allocation fails. */
Matrix *weightsDerivative(Vector *pDelta, Vector *pPrev_activation) {
    if(pDelta == NULL || pPrev_activation == NULL) {
        return NULL;
    }
    Matrix *gradient = createMatrix(pDelta->rows, pPrev_activation->rows);
    if(gradient == NULL) {
        return NULL;
    }
    for(uint32_t i = 0; i < pDelta->rows; i++) {
        for(uint32_t j = 0; j < pPrev_activation->rows; j++) {
            uint32_t index = i * gradient->columns + j;
            gradient->data[index] = pDelta->data[i] * pPrev_activation->data[j];
        }
    }
    return gradient;
}

/* Updates pCurrent_weights in-place using gradient descent with gradient clipping.
   The gradient matrix dL/dW is computed, its L2-norm is calculated, and if the
   norm exceeds GRADIENT_CLIP the gradient is scaled down proportionally before
   applying: W -= learning_rate * scale * dL/dW. */
void updateWeights(float pLearning_rate, Matrix *pCurrent_weights, Vector *pDelta, Vector *pPrev_activation) {
    if(pCurrent_weights == NULL || pDelta == NULL || pPrev_activation == NULL) {
        return;
    }
    Matrix *gradient = weightsDerivative(pDelta, pPrev_activation);
    if(gradient == NULL) {
        return;
    }

    /* Compute the L2-norm of the gradient matrix. */
    float norm  = 0.0f;
    uint32_t total = pCurrent_weights->rows * pCurrent_weights->columns;
    for(uint32_t k = 0; k < total; k++) {
        norm += gradient->data[k] * gradient->data[k];
    }
    norm = sqrtf(norm);

    /* Scale down only if the norm exceeds the clip threshold. */
    float scale = (norm > GRADIENT_CLIP) ? (GRADIENT_CLIP / norm) : 1.0f;

    for(uint32_t i = 0; i < pCurrent_weights->rows; i++) {
        for(uint32_t j = 0; j < pCurrent_weights->columns; j++) {
            uint32_t index = i * pCurrent_weights->columns + j;
            pCurrent_weights->data[index] -= pLearning_rate * gradient->data[index] * scale;
        }
    }
    freeMatrix(gradient);
}

/* Updates the bias vector in-place: b[i] -= learning_rate * delta[i].
   No-op if either pointer is NULL or lengths differ. */
void updateBias(float pLearning_rate, Vector *pBias, Vector *pDelta) {
    if(pBias == NULL || pDelta == NULL) {
        return;
    }
    if(pBias->rows != pDelta->rows) {
        return;
    }
    for(uint32_t i = 0; i < pBias->rows; i++) {
        pBias->data[i] -= pLearning_rate * pDelta->data[i];
    }
}
