#pragma once
#include <vector>


class WaveFunction {
public:
    WaveFunction(class System* system);
    int     getNumberOfParameters() { return m_numberOfParameters; }
    std::vector<double> getParameters() { return m_parameters; }
    virtual double evaluate(double GibbsValue, std::vector<double> X, std::vector<double> Hidden, std::vector<double> a_bias, std::vector<double> b_bias, std::vector<std::vector<double>> w) = 0;
    virtual double computeDoubleDerivative(double GibbsValue, std::vector<double> X, std::vector<double> Hidden, std::vector<double> a_bias, std::vector<double> b_bias, std::vector<std::vector<double> > w)=0;
    virtual std::vector<double> QuantumForce(double GibbsValue, std::vector<double> X, std::vector<double> a_bias, std::vector<double> b_bias, std::vector<std::vector<double> > w)=0;

    void setParameters(const std::vector<double> &parameters);

protected:
    int     m_numberOfParameters = 0;
    std::vector<double> m_parameters = std::vector<double>();
    class System* m_system = nullptr;
};

