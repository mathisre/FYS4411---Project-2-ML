#pragma once
#include <vector>
using namespace std;

class Hamiltonian {
public:
    Hamiltonian(class System* system);
    virtual double computeLocalEnergy(double GibbsValue, bool interaction, vector<double> X, vector<double> Hidden, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double> > w)=0;

protected:
    class System* m_system = nullptr;
};

