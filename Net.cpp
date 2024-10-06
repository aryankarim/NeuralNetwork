#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "Neuron.cpp"

using namespace std;

typedef vector<Neuron> Layer;

class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return recentAverageError; }

private:
    vector<Layer> all_layers;
    double error;
    double recentAverageError;
    double recentAverageSmoothingFactor;
};

Net::Net(const vector<unsigned> &topology)
{

    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        all_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            all_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a neuron!" << endl;
        }

        // bias neuron init
        all_layers.back().back().setOutputVal(1.0);
    }
};

void Net::feedForward(const vector<double> &inputVals)
{

    assert(inputVals.size() == all_layers[0].size() - 1);

    for (unsigned i = 0; i < inputVals.size(); i++)
    {
        all_layers[0][i].setOutputVal(inputVals[i]);
    }

    for (unsigned layerNum = 0; layerNum < all_layers.size(); layerNum++)
    {
        Layer &prevLayer = all_layers[layerNum - 1];
        for (unsigned n = 0; n < all_layers[layerNum].size(); n++)
        {
            all_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const vector<double> &targetVals)
{

    // Calculate overall net error -- RMS of output neuron errors

    Layer &outputLayer = all_layers.back();
    error = 0.0;

    // Recent Average Mesurement

    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1);

    // Calcualte output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers

    for (unsigned layerNum = all_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = all_layers[layerNum];
        Layer &nextLayer = all_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    //  For all layers form outpus to first hidden layer, update connection weight

    for (unsigned layerNum = all_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = all_layers[layerNum];
        Layer &prevLayer = all_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n)
        {
            layer[n].updateInputWeight(prevLayer);
        }
    }
}

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < all_layers.size() - 1; n++)
    {
        resultVals.push_back(all_layers.back()[n].getOutputVal());
    }
};