#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "Connection.cpp"

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; };
    double getOutputVal(void) const { return m_outputVal; };
    void feedForward(const Layer& prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeight(Layer& prevLayer);

private:
    static double eta;
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer& nextLayer);
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double gradient;

    
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;


Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; c++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

void Neuron::updateInputWeight(Layer& prevLayer) {
    for (unsigned n = 0; n < prevLayer.size(); n++)
    {
        Neuron &neuron = prevLayer[n];

        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer& nextLayer) {
    double sum = 0;

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);

}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    // tanh - output range [-1,1]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer& prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs which are out inputs.
    // Include the ias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); n++)
    {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

