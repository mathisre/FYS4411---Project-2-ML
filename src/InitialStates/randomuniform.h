#pragma once
#include "initialstate.h"
using namespace std;

class RandomUniform : public InitialState {
public:
    RandomUniform(System* system, int numberOfParticles, int numberOfDimensions, int numberOfHiddenNodes, int numberOfVisibleNodes, double sigma, vector<double>& m_X, vector<double>& m_Hidden, vector<double>& m_a_bias, vector<double>& m_b_bias, vector<std::vector<double>>& m_w, double timeStep,int numberOfParameters);
    void setupInitialState(vector<double> &m_X, vector<double> &m_Hidden, vector<double> &m_a_bias, vector<double> &m_b_bias, vector<std::vector<double> > &m_w);
  //  void setupInitialStateWithInteraction();

};

