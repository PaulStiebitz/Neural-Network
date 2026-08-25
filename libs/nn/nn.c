#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "../matrix/matrix.h"

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
    printf("[NN] num_inputs: %d\n", pNeural_network_layer->num_inputs);
    printf("[NN] num_neurons: %d\n", pNeural_network_layer->num_neurons);
    printf("[NN] matrix_weights:\n");
    printMatrix(pNeural_network_layer->matrix_weights, "float");
    printf("[NN] vector_biases:\n");
    printVector(pNeural_network_layer->vector_biases);
    printf("[NN] vector_preactivation_z:\n");
    printVector(pNeural_network_layer->vector_preactivation_z);
    printf("[NN] vector_activation_a:\n");
    printVector(pNeural_network_layer->vector_activation_a);
    printf("[NN] vector_error_delta:\n");
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

void trainNeuralNetwork(NeuralNetwork *pNeural_network, LabeledMatrixList *training_labeled_matrix_list) {
    for(uint32_t i = 0; i < training_labeled_matrix_list->list_length; i++) {
        forwardPass(training_labeled_matrix_list->list[i], pNeural_network, i);
        backwardPass(training_labeled_matrix_list->list[i], pNeural_network);
    }
}

void forwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network, uint32_t pMatrix_num) {
    uint32_t num_layers = pNeural_network->layer;
    for(uint32_t i = 0; i < num_layers; i++) {
        NeuralNetworkLayer *current_network_layer = pNeural_network->layer_list[i];
        Matrix *current_layer_weights = current_network_layer->matrix_weights;
        Vector *current_layer_biases = current_network_layer->vector_biases;
        Vector *current_layer_preactivation_z = current_network_layer->vector_preactivation_z;
        Vector *current_layer_activation_a = current_network_layer->vector_activation_a;
        if(i == 0) {
            Vector *initial_matrix_input_vector = matrixToVector(pLabeled_matrix->matrix);
            if(initial_matrix_input_vector == NULL) {
                return;
            }
            normalizeVector(initial_matrix_input_vector, 255.0f);
            matrixTimesVector(current_layer_weights, initial_matrix_input_vector, current_layer_preactivation_z);
            free(initial_matrix_input_vector->data);
            free(initial_matrix_input_vector);
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
            //printMatrix(pLabeled_matrix->matrix, "float");
            printf("[Pred] %d.Picture Predicted Number: %d Prob: %f, Actual number: %d\n", pMatrix_num, predicted_num, predicted_num_prob, actual_number);
        } else {
            relu_activation(current_layer_preactivation_z, current_layer_activation_a);
        }
    }
}

void backwardPass(LabeledMatrix *pLabeled_matrix, NeuralNetwork *pNeural_network) {

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


