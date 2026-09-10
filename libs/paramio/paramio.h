#ifndef PARAMIO_H
#define PARAMIO_H

#include <stdio.h>
#include <stdbool.h>
#include "../nn/nn.h"

/* Opens the file at pParamExportPath and writes the trained weights and biases
 * of pNeuralNetwork to it in binary format. Logs an error and returns early if
 * the network is invalid or the path is NULL. */
void exportTrainedParams(NeuralNetwork *pNeuralNetwork, const char *pParamExportPath);

/* Returns true if pNeuralNetwork and all of its layers have non-NULL, non-zero
 * weight matrices and bias vectors; returns false otherwise. */
bool validNeuralNetwork(NeuralNetwork *pNeuralNetwork);

/* Writes the layer count followed by each layer's num_inputs, num_neurons,
 * weight floats and bias floats to pParamExportFile in binary format.
 * Returns early if either pointer is NULL. */
void writeTrainedParams(FILE *pParamExportFile, NeuralNetwork *pNeural_network);


#endif
