#include <stdint.h>
#include <stdio.h>
#include "nn.h"
#include "../matrix/matrix.h"

/*
Build the neural network with the folling config array:
pLayers: num of total layers
pLayerConfig: the first value indicates the number of inputs.
The remanining array indicitates the total number of neurons for the i-th layer, meanwhile
the index i represents also the index of the array.
*/
NeuralNetwork *buildNeuralNetwork(uint32_t pStart_input_num, uint32_t pLayer_num, uint32_t *pLayer_config) {
    size_t NeuralNetwork_mem_req = sizeof(NeuralNetwork);
    NeuralNetwork *neural_network = malloc(NeuralNetwork_mem_req);
    if(neural_network == NULL) {
        return NULL;
    }

    size_t layer_list_mem_req = pLayer_num * sizeof(NeuralNetworkLayer);
    neural_network->layer_list = malloc(layer_list_mem_req);
    if(neural_network->layer_list == NULL) {
        free(neural_network);
        return NULL;
    }

    for(uint32_t i = 0; i < pLayer_num; i++) {
        NeuralNetworkLayer *current_layer = neural_network->layer_list[i];
        uint32_t layer_num_neurons = pLayer_config[i];
        current_layer->num_neurons = layer_num_neurons;
        /*
        The first Hidden Layer doesn't have a previous layer, which means pStart_input_num
        is used for num_inputs, matrix_weights and biases.
         */
        if(i == 0) {
            current_layer->num_inputs = pStart_input_num;
            current_layer->matrix_weights = createMatrix(layer_num_neurons, pStart_input_num);
        } else {
            uint32_t prev_layer_input_num = pLayer_config[i-1];
            current_layer->num_inputs = prev_layer_input_num;
            current_layer->matrix_weights = createMatrix(layer_num_neurons, prev_layer_input_num);
        }

        // Free Neural Network Layer ...
    }
}