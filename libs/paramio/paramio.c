#include "paramio.h"

void exportTrainedParams(NeuralNetwork *pNeuralNetwork, const char *pParamExportPath) {
    if(validNeuralNetwork(pNeuralNetwork) == false) {
        printf("[PARAMIO] exportTrainedParams failed, due neural network null value\n");
        return;
    } else if(pParamExportPath == NULL) {
        printf("[PARAMIO] exportTrainedParams failed, due to pParamExportPath null value");
        return;
    }
    FILE *paramExportFile = fopen(pParamExportPath, "wb");
    if(paramExportFile == NULL) {
        return;
    }
    writeTrainedParams(paramExportFile, pNeuralNetwork);
    fclose(paramExportFile);
}

/* Checks that the network pointer itself, its layer count, and every layer's
 * weight matrix and bias vector are all non-NULL and non-zero before export. */
bool validNeuralNetwork(NeuralNetwork *pNeuralNetwork) {
    if(pNeuralNetwork == NULL || pNeuralNetwork->layer == 0) {
        return false;
    }
    uint32_t layer_num = pNeuralNetwork->layer;
    for(uint32_t i = 0; i < layer_num; i++) {
        NeuralNetworkLayer *current_layer = pNeuralNetwork->layer_list[i];
        if(current_layer == NULL || current_layer->num_inputs == 0 || current_layer->num_neurons == 0) {
            return false;
        }
        if(current_layer->matrix_weights == NULL || current_layer->vector_biases == NULL) {
            return false;
        }
    }
    return true;
}

// File Format
//
// [uint32] layer_count
//
// Repeated for each layer:
//
// [uint32] num_inputs
// [uint32] num_neurons
// [float] weights[num_inputs * num_neurons]
// [float] biases[num_neurons]
void writeTrainedParams(FILE *pParamExportFile, NeuralNetwork *pNeural_network) {
    if(pParamExportFile == NULL || pNeural_network == NULL) {
        return;
    }
    // [uint32] layer_count
    uint32_t neural_network_layer_count = pNeural_network->layer;
    fwrite(&neural_network_layer_count, sizeof(uint32_t), 1, pParamExportFile);
    // Repeated for each layer:
    for(uint32_t i = 0; i < pNeural_network->layer; i++) {
        NeuralNetworkLayer *current_network_layer = pNeural_network->layer_list[i];
        uint32_t num_inputs = current_network_layer->num_inputs;
        uint32_t num_neurons = current_network_layer->num_neurons;
        Matrix *current_layer_weight_matrix = current_network_layer->matrix_weights;
        Vector *current_layer_bias_vector = current_network_layer->vector_biases;
        uint32_t matrix_entries = current_layer_weight_matrix->rows * current_layer_weight_matrix->columns;
        uint32_t vector_entries = current_layer_bias_vector->rows;

        fwrite(&num_inputs, sizeof(uint32_t), 1, pParamExportFile);
        fwrite(&num_neurons, sizeof(uint32_t), 1, pParamExportFile);
        // [float] weights[num_inputs * num_neurons]
        for(uint32_t j = 0; j < matrix_entries; j++) {
            float current_matrix_weight = current_layer_weight_matrix->data[j];
            fwrite(&current_matrix_weight, sizeof(float), 1, pParamExportFile);
        }
        // [float] biases[num_neurons]
        for(uint32_t k = 0; k < vector_entries; k++) {
            float current_vector_bias = current_layer_bias_vector->data[k];
            fwrite(&current_vector_bias, sizeof(float), 1, pParamExportFile);
        }
    }
}
