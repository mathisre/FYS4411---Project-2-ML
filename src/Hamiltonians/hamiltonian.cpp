#include "hamiltonian.h"
#include "../WaveFunctions/simplegaussian.h"
#include "../system.h"
#include "../particle.h"
#include "vector"
#include <iostream>

using namespace std;

Hamiltonian::Hamiltonian(System* system) {
    m_system = system;
}


