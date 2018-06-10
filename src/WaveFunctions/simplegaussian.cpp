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

SimpleGaussian::SimpleGaussian(System* system):
    WaveFunction(system) {
    m_numberOfParameters = 3;
    m_parameters.reserve(3);
}

double SimpleGaussian::evaluate(double GibbsValue,vector<double> X, vector<double> Hidden, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double>> w) {

    double first_sum = 0;
    double prod = 1;
    int N = m_system->getNumberOfHiddenNodes();
    int M = m_system->getNumberOfVisibleNodes();
    for (int i = 0; i < M; i++){
        first_sum += (X[i]-a_bias[i])*(X[i]-a_bias[i]);
    }
    first_sum /= 2*m_system->getSigma_squared();

    first_sum = exp(-first_sum*GibbsValue);
    for (int j = 0; j < N; j++){
        double second_sum = 0;
        for (int i = 0; i < M; i++){
            second_sum += X[i]*w[i][j];
        }
        second_sum /= m_system->getSigma_squared();

        prod *= 1 + exp(b_bias[j] + second_sum);
    }
    if(GibbsValue==0.5) {return first_sum*sqrt(prod); cout<<"ehi"<<endl;}

    return first_sum*prod;
}


double SimpleGaussian::computeDoubleDerivative(double GibbsValue, vector<double> X, vector<double> Hidden, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double>> w) {

    int N = m_system->getNumberOfHiddenNodes();
    int M = m_system->getNumberOfVisibleNodes();

    vector <double> argument(N);

    double firstsum  = 0.0;
    double secondsum = 0.0;
    double kinetic   = 0.0;

    double temp2;
    double temp3;
    double sum;

    for(int j = 0; j < N; j++){
        sum = 0;

        for(int i = 0; i < M; i++){
            sum += X[i] * w[i][j] / m_system->getSigma_squared();
        }

        argument[j] = exp( - b_bias[j] - sum);
    }
    for(int i = 0 ; i < M;i++){
        temp2 = 0;
        temp3 = 0;
        for(int j = 0; j < N; j++){
            double expon = 1.0 + argument[j];
            temp2 += w[i][j] * 1.0 / (expon);
            temp3 += w[i][j] * w[i][j] * argument[j] / (expon*expon);
        }
        firstsum = - ( X[i] - a_bias[i] ) + temp2;
        firstsum /= m_system->getSigma_squared();

        secondsum = temp3 / ( m_system->getSigma_squared() * m_system->getSigma_squared() )
                    - 1.0 / m_system->getSigma_squared();

        kinetic += firstsum * firstsum*GibbsValue*GibbsValue + secondsum*GibbsValue;
    }
    return kinetic;
}


std::vector<double> SimpleGaussian::QuantumForce(double GibbsValue, vector<double> X, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double>> w) {
    //Function to comput the Quantum Force for the Importance Sampling method

    int N = m_system->getNumberOfHiddenNodes();
    int M = m_system->getNumberOfVisibleNodes();

    vector <double> QuantumForce(M);
    vector <double> argument(N);
    vector <double> temp2(M);

    double sum;

    for (int j = 0; j < N; j++){
        sum = 0;
        for (int i = 0; i < M; i++){
            sum += X[i] * w[i][j] / m_system->getSigma_squared();
        }
        argument[j] = b_bias[j] + sum;
    }

    for(int i = 0; i < M; i++){
        temp2[i] = 0;
        for(int j = 0; j < N; j++){
            double temp4 = exp( - argument[j] );
            double expon = 1 + temp4;
            temp2[i] += w[i][j] / ( expon );

        }
        QuantumForce[i] = 2 * ( - (X[i] - a_bias[i] ) + temp2[i] ) * GibbsValue / m_system->getSigma_squared();

    }

    return QuantumForce;
}

