#include "randomuniform.h"
#include <iostream>
#include <cassert>
#include "../Math/random.h"
#include "../particle.h"
#include "../system.h"
#include <cmath>
#include <random>
#include <vector>

using std::cout;
using std::endl;

using namespace std;

RandomUniform::RandomUniform(System* system, int numberOfParticles, int numberOfDimensions, int numberOfHiddenNodes, int numberOfVisibleNodes, double sigma, vector<double>& m_X, vector<double>& m_Hidden, vector<double>& m_a_bias, vector<double>& m_b_bias, vector<std::vector<double>>& m_w, double timeStep, /*double learningRate,*/ int numberOfParameters)   :
        InitialState(system) {
    assert(numberOfHiddenNodes > 0 && numberOfVisibleNodes > 0);

    m_system->setSigma(sigma);
    m_system->setSigma_squared(sigma*sigma);
    m_system->setNumberOfHiddenNodes(numberOfHiddenNodes);
    m_system->setNumberOfVisibleNodes(numberOfVisibleNodes);

    m_system->setNumberOfParticles(numberOfParticles);
    m_system->setNumberOfDimensions(numberOfDimensions);
    m_system->setTimeStep(timeStep);
    m_system->setSqrtTimeStep(sqrt(timeStep));

    setupInitialState(m_X, m_Hidden, m_a_bias, m_b_bias, m_w); //delete m_hidden
}

void RandomUniform::setupInitialState(vector<double>& m_X, vector<double>& m_Hidden, vector<double>& m_a_bias, vector<double>& m_b_bias, vector<std::vector<double>>& m_w) {

//    double sigma_0 = 0.001;
    double sigma_0 = 0.5;
    int M = m_system->getNumberOfVisibleNodes();
    int N= m_system->getNumberOfHiddenNodes();

    for (int i=0; i < M; i++) {
     m_X[i]=(Random::nextDouble()-0.5);
    }

    for(int i=0; i<M; i++){
        m_a_bias[i]=Random::nextGaussian(0,sigma_0);
    }

    for(int i=0; i<N; i++){
        m_b_bias[i]=Random::nextGaussian(0,sigma_0);
    }

    for(int i=0; i<M; i++){
        for(int j=0;j<N; j++){
            m_w[i][j]=Random::nextGaussian(0,sigma_0);
        }
    }

//    m_a_bias[0]=0.000352794;
//    m_a_bias[1]= -0.00107663;
//    m_a_bias[2]= 0.000186011;
//    m_a_bias[3]= 0.00075935;

//    m_b_bias[0]= 0.00158981;
//    m_b_bias[1]= -0.000396733;

//    m_w[0][0]=-0.000488378;
//    m_w[0][1]=-0.00200291;
//    m_w[1][0]=-0.00131297;
//    m_w[1][1]=0.000953404;
//    m_w[2][0]=0.000882991;
//    m_w[2][1]=-0.000605703;
//    m_w[3][0]=-0.00258305;
//    m_w[3][1]=0.00118713;

//    m_X[0]= 0.126214;
//    m_X[1]=0.281379;
//    m_X[2]=-0.460602;
//    m_X[3]=0.183972;

}


