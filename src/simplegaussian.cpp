#include "simplegaussian.h"
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"
#include <iostream>

using namespace std;
using std::vector;

SimpleGaussian::SimpleGaussian(System* system, double numberOfHiddenNodes, double numberOfVisibleNodes, vector<double> X, vector<double> Hidden, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double>> w) :
    WaveFunction(system) {
    m_numberOfParameters = 3;
    m_parameters.reserve(3);
    m_parameters.push_back(a_bias);
    m_parameters.push_back(b_bias);
    m_parameters.push_back(w);
    m_system->set
    //m_parameters.push_back(beta);
}

double SimpleGaussian::evaluate(vector<double> X, vector<double> Hidden, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double>> w) {

    double first_sum = 0;
    double prod = 1;
    int M = m_system->m_initialState->getNumberOfHiddenNodes();
    int N = m_system->m_initialState->getNumberOfVisibleNodes();
    for (int i = 0; i < M; i++){
        first_sum -= (X[i]-a_bias[i])*(X[i]-a_bias[i]);
    }
    first_sum /= 2*m_system->m_initialState->getSigmaSquared;

    first_sum = exp(first_sum);

    for (int j = 0; j < N; j++){
        double second_sum = 0;
        for (int i = 0; i < M; i++){
            second_sum += X[i]*w[i][j];
        }
        second_sum /= m_system->m_initialState->getSigmaSquared;

        prod *= 1 + exp(b[j] + second_sum);
    }
    return first_sum*prod;
}


double SimpleGaussian::computeDoubleDerivative(vector<double> X, vector<double> Hidden, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double>> w) {

    // Can probably reuse quantum force term to get the second derivative
    int NumbOfParticles = m_system->m_initialState->getNumberOfParticles;

    int M = m_system->m_initialState->getNumberOfHiddenNodes();
    int N = m_system->m_initialState->getNumberOfVisibleNodes();

    double large_sum = 0;
    for (int j = 0; j < NumbOfParticles; j++){

        for (int k = 0; k < N; k++){

            double second_sum = 0;
            for (int i = 0; i < M; i++){
                second_sum += X[i]*w[i][j];
            }
            second_sum /= m_system->m_initialState->getSigmaSquared;
            double exponentials = exp(b_bias[k] + second_sum);
            double third_sum = 0;

            for (int i = j; i < j+2; i++){
                third_sum += w[i][k]*w[i][k];
            }
            third_sum /= m_system->m_initialState->getSigmaSquared*m_system->m_initialState->getSigmaSquared;

            large_sum += third_sum*exponentials /( (1 + exponentials)*(1+exponentials));
        }
    }
    return large_sum + (3*NumbOfParticles / m_system->m_initialState->getSigmaSquared);
}


