#pragma once
#include "hamiltonian.h"
#include <vector>
using namespace std;

class HarmonicOscillator : public Hamiltonian {
public:
    HarmonicOscillator(System* system, double omega);
    double computeLocalEnergy(double GibbsValue, bool interaction, vector<double> X, vector<double> Hidden, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double> > w);

    double omega() const;
    void setOmega(const double &omega);

    double computeInteractionPotential();
    double computePotentialEnergy(vector<double> X);
private:
    double m_omega = 0;
};

