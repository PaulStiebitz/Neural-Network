#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nn.h"
#include "../matrix/matrix.h"
#define GRADIENT_CLIP 1.0f

NeuralNetworkLayer *createNeuralNetworkLayer(uint32_t pNum_inputs, uint32_t pNum_neurons) {
    size_t neural_network_layer_mem_req = sizeof(NeuralNetworkLayer);
    NeuralNetworkLayer *neural_network_layer = calloc(1, neural_network_layer_mem_req);
    if(neural_network_layer == NULL) {
        return NULL;
    }
    neural_network_layer->num_inputs = pNum_inputs;
    neural_network_layer->num_neurons = pNum_neurons;
    neural_network_layer->matrix_weights = createMatrix(pNum_neurons, pNum_inputs);
    if(neural_network_layer->matrix_weights == NULL)
        goto cleanup;
    fillMatrixRandom(neural_network_layer->matrix_weights);
    neural_network_layer->vector_biases = createVector(pNum_neurons);
    if(neural_network_layer->vector_biases == NULL)
        goto cleanup;

    neural_network_layer->vector_preactivation_z = createVector(pNum_neurons);
    if(neural_network_layer->vector_preactivation_z == NULL)
        goto cleanup;

    neural_network_layer->vector_activation_a = createVector(pNum_neurons);
    if(neural_network_layer->vector_activation_a == NULL)
        goto cleanup;

    neural_network_layer->vector_error_delta = createVector(pNum_neurons);
    if(neural_network_layer->vector_error_delta == NULL)
        goto cleanup;

    return neural_network_layer;

    cleanup:
        freeNeuralNetworkLayer(neural_network_layer);
        return NULL;
}

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

/*
Build the neural network with the folling config array:
pLayers: num of total layers
pLayerConfig: the first value indicates the number of inputs.
The remanining array indicitates the total number of neurons for the i-th layer, meanwhile
the index i represents also the index of the array.
*/
NeuralNetwork *createNeuralNetwork(uint32_t pStart_input_num, uint32_t pLayer_num, uint32_t *pLayer_config, float pLearning_rate) {
    size_t neural_network_mem_req = sizeof(NeuralNetwork);
    NeuralNetwork *neural_network = malloc(neural_network_mem_req);
    if(neural_network == NULL) {
        return NULL;
    }
    neural_network->layer = pLayer_num;
    neural_network->learning_rate = pLearning_rate;
    size_t neural_network_layer_list_mem_req = pLayer_num * sizeof(NeuralNetworkLayer*);
    neural_network->layer_list = malloc(neural_network_layer_list_mem_req);
    if(neural_network->layer_list == NULL) {
        free(neural_network);
        return NULL;
    }

    for(uint32_t i = 0; i < pLayer_num; i++) {
        uint32_t num_neurons = pLayer_config[i];
        if(i == 0) {
            neural_network->layer_list[i] = createNeuralNetworkLayer(pStart_input_num, num_neurons);
        } else {
            uint32_t num_input = pLayer_config[i-1];
            neural_network->layer_list[i] = createNeuralNetworkLayer(num_input, num_neurons);
        }

        if(neural_network->layer_list[i] == NULL) {
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

void printNeuralNetworkLayer(NeuralNetworkLayer *pNeural_network_layer) {
    printf("[NNL] num_inputs: %d\n", pNeural_network_layer->num_inputs);
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

void printNeuralNetwork(NeuralNetwork *pNeural_network) {
    printf("[NN] Printing neural network\n");
    printf("Debugging Layers!!!: %d\n", pNeural_network->layer);
    for(uint32_t i = 0; i < pNeural_network->layer; i++) {
        printf("[NN] Printing Layer: %d\n", i + 1);
        printNeuralNetworkLayer(pNeural_network->layer_list[i]);
    }
    printf("\n");
}

void testNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *testing_labeled_matrix_list) {
    uint32_t correct = 0;
    for(uint32_t matrix_num = 0; matrix_num < testing_labeled_matrix_list->list_length; matrix_num++) {
        Matrix *current_matrix = testing_labeled_matrix_list->list[matrix_num]->matrix;
        Vector *initial_matrix_input = matrixToVector(current_matrix);
        if(initial_matrix_input == NULL) {
            return;
        }
        forwardPass(testing_labeled_matrix_list->list[matrix_num], pNeural_network, initial_matrix_input, matrix_num, "test");
        // Determine Accuracy
        NeuralNetworkLayer *last_layer = pNeural_network->layer_list[pNeural_network->layer - 1];
        uint32_t predicted = argmax(last_layer->vector_activation_a);
        if(predicted == testing_labeled_matrix_list->list[matrix_num]->label) {
            correct++;
        }
        freeVector(initial_matrix_input);
    }
    printf("[Test] Accuracy: %d/%d (%.2f%%)\n", correct, testing_labeled_matrix_list->list_length, 100.0f * (float)correct / (float)testing_labeled_matrix_list->list_length);
}

void trainNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *training_labeled_matrix_list) {
    // Fisher-Yates Shuffle: Change order of pictures in every epoch
    uint32_t list_length = training_labeled_matrix_list->list_length;
    for(uint32_t i = list_length - 1; i > 0; i--) {
        uint32_t j = (uint32_t)rand() % (i + 1);
        LabeledMatrix *tmp = training_labeled_matrix_list->list[i];
        training_labeled_matrix_list->list[i] = training_labeled_matrix_list->list[j];
        training_labeled_matrix_list->list[j] = tmp;
    }

    for(uint32_t matrix_num = 0; matrix_num < training_labeled_matrix_list->list_length; matrix_num++) {
        Matrix *current_matrix = training_labeled_matrix_list->list[matrix_num]->matrix;
        Vector *initial_matrix_input = matrixToVector(current_matrix);
        if(initial_matrix_input == NULL) {
            return;
        }
        forwardPass(training_labeled_matrix_list->list[matrix_num], pNeural_network, initial_matrix_input, matrix_num, "train");
        backwardPass(training_labeled_matrix_list->list[matrix_num], pNeural_network, initial_matrix_input, matrix_num);
        freeVector(initial_matrix_input);
    }
}

void forwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network, Vector *pInitial_matrix_input_vector, uint32_t pMatrix_num, const char *pMode) {
    uint32_t num_layers = pNeural_network->layer;
    for(uint32_t i = 0; i < num_layers; i++) {
        NeuralNetworkLayer *current_network_layer = pNeural_network->layer_list[i];
        Matrix *current_layer_weights = current_network_layer->matrix_weights;
        Vector *current_layer_biases = current_network_layer->vector_biases;
        Vector *current_layer_preactivation_z = current_network_layer->vector_preactivation_z;
        Vector *current_layer_activation_a = current_network_layer->vector_activation_a;
        if(i == 0) {
            normalizeVector(pInitial_matrix_input_vector, 255.0f);
            matrixTimesVector(current_layer_weights, pInitial_matrix_input_vector, current_layer_preactivation_z);
        } else {
            NeuralNetworkLayer *prev_network_layer = pNeural_network->layer_list[i-1];
            Vector *prev_layer_activation_a = prev_network_layer->vector_activation_a;
            matrixTimesVector(current_layer_weights, prev_layer_activation_a, current_layer_preactivation_z);
        }
        vectorPlusVector(current_layer_biases, current_layer_preactivation_z);
        if(i == num_layers - 1) {
            softmax(current_layer_preactivation_z, current_layer_activation_a);
            uint32_t predicted_num = argmax(current_layer_activation_a);
            if(predicted_num == UINT32_MAX) {
                return;
            }
            float predicted_num_prob = current_layer_activation_a->data[predicted_num];
            uint32_t actual_number = pLabeled_matrix->label;
            if(strcmp(pMode, "test") == 0) {
                printf("[Pred] %d.Picture Predicted Number: %d Prob: %.3f, Actual number: %d\n", pMatrix_num, predicted_num, predicted_num_prob, actual_number);
            }
        } else {
            relu_activation(current_layer_preactivation_z, current_layer_activation_a);
        }
    }
}

void backwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network, Vector *pInitial_matrix_input_vector, uint32_t pMatrix_num) {
    int32_t last_layer = (int32_t)pNeural_network->layer - 1;
    for(int32_t i = last_layer; i >= 0; i--) {
        NeuralNetworkLayer *current_layer = pNeural_network->layer_list[i];
        Vector *current_delta = current_layer->vector_error_delta;
        if(i == last_layer) {
            //printf("[Loss] %d.Picture ", pMatrix_num);
            //printCrossEntropyLoss(pLabeled_matrix->label, current_layer->vector_activation_a);
            softmaxCrossEntropyDerivative(pLabeled_matrix->label, current_layer->vector_activation_a, current_delta);
        } else {
            NeuralNetworkLayer *next_layer = pNeural_network->layer_list[i + 1];
            layerDelta(next_layer->matrix_weights, next_layer->vector_error_delta, current_layer->vector_preactivation_z, current_delta);
        }

        Vector *prev_activation;
        if(i == 0) {
            prev_activation = pInitial_matrix_input_vector;
        } else {
            prev_activation = pNeural_network ->layer_list[i - 1]->vector_activation_a;
        }

        updateWeights(pNeural_network->learning_rate, current_layer->matrix_weights, current_delta, prev_activation);
        updateBias(pNeural_network->learning_rate, current_layer->vector_biases, current_delta);
    }
}

void relu_activation(Vector *pPre_activation, Vector *pReLU_destination) {
    if(pPre_activation == NULL || pReLU_destination == NULL) {
        return;
    }

    if(pPre_activation->rows != pReLU_destination->rows) {
        return;
    }

    uint32_t vector_length = pPre_activation->rows;
    for(uint32_t i = 0; i < vector_length; i++) {
        if(pPre_activation->data[i] > 0) {
            pReLU_destination->data[i] = pPre_activation->data[i];
        } else if(pPre_activation->data[i] <= 0) {
            pReLU_destination->data[i] = 0;
        }
    }
}

void softmax(Vector *pActivation, Vector *pSoftmax_destination) {
    if(pActivation == NULL || pSoftmax_destination == NULL) {
        return;
    }
    if(pActivation->rows != pSoftmax_destination->rows) {
        return;
    }
    // substract numerical stability
    float max_val = pActivation->data[0];
    for(uint32_t i = 1; i < pActivation->rows; i++) {
        if(pActivation->data[i] > max_val) {
            max_val = pActivation->data[i];
        }
    }

    float sum = 0.0f;
    for(uint32_t j = 0; j < pActivation->rows; j++) {
        sum += expf(pActivation->data[j] - max_val);
    }

    for(uint32_t i = 0; i < pActivation->rows; i++) {
        pSoftmax_destination->data[i] = expf(pActivation->data[i] - max_val) / sum;
    }
}

uint32_t argmax(Vector *pSoftmax_vector) {
    if(pSoftmax_vector == NULL) {
        return UINT32_MAX;
    }
    float max_prob_value = 0;
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

void printCrossEntropyLoss(uint32_t pActual_number, Vector *pActivation) {
    if(pActivation == NULL || pActual_number >= pActivation->rows) {
        return;
    }
    double activation_val = pActivation->data[pActual_number];
     if (activation_val < EPSILON) {
        activation_val = EPSILON;
    }
    double loss = -log(activation_val);
    printf("Loss: %.3f\n", loss);
}

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

float reluDerivative(float z)
{
    if(z > 0.0f) {
        return 1.0f;
    }
    return 0.0f;
}

void layerDelta(Matrix *pLast_Weights, Vector *pLast_delta, Vector *pCurrent_z, Vector *pDestination_delta) {
    if(pLast_Weights == NULL || pLast_delta == NULL || pCurrent_z == NULL || pDestination_delta == NULL) {
        return;
    }

    matrixTransposeTimesVector(pLast_Weights, pLast_delta, pDestination_delta);

    for(uint32_t i = 0; i < pDestination_delta->rows; i++) {
        pDestination_delta->data[i] *= reluDerivative(pCurrent_z->data[i]);
    }
}

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

void updateWeights(float pLearning_rate, Matrix *pCurrent_weights, Vector *pDelta, Vector *pPrev_activation) {
    if(pCurrent_weights == NULL || pDelta == NULL || pPrev_activation == NULL) {
        return;
    }

    Matrix *gradient = weightsDerivative(pDelta, pPrev_activation);

    if(gradient == NULL) {
        return;
    }

    // Gradient Clipping: L2-Norm des Gradienten berechnen
    float norm = 0.0f;
    uint32_t total = pCurrent_weights->rows * pCurrent_weights->columns;
    for(uint32_t k = 0; k < total; k++) {
        norm += gradient->data[k] * gradient->data[k];
    }
    norm = sqrtf(norm);

    // Skalierungsfaktor: nur kürzen wenn Norm > GRADIENT_CLIP
    float scale = (norm > GRADIENT_CLIP) ? (GRADIENT_CLIP / norm) : 1.0f;

    for(uint32_t i = 0; i < pCurrent_weights->rows; i++) {
        for(uint32_t j = 0; j < pCurrent_weights->columns; j++) {
            uint32_t index = i * pCurrent_weights->columns + j;
            pCurrent_weights->data[index] -= pLearning_rate * gradient->data[index] * scale;
        }
    }

    freeMatrix(gradient);
}

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


