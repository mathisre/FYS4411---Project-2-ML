#pragma once
#include <vector>
#include "../particle.h"

class InitialState {
public:
    InitialState(class System* system);
    virtual void setupInitialState(std::vector<double> &m_X, std::vector<double> &m_Hidden, std::vector<double> &m_a_bias, std::vector<double> &m_b_bias, std::vector<std::vector<double> > &m_w) = 0;
    std::vector<class Particle> getParticles() { return m_particles; }


protected:
    class System* m_system = nullptr;
    class Random* m_random = nullptr;
    std::vector<Particle> m_particles; //= std::vector<Particle>();

//    std::vector<double> m_X=std::vector<double>();
//    std::vector<double> m_Hidden=std::vector<double>();
//    std::vector<double> m_a_bias=std::vector<double>();
//    std::vector<double> m_b_bias=std::vector<double>();
//    std::vector<std::vector<double>> m_w=std::vector<std::vector<double>>;

};

