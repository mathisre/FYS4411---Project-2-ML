#!/bin/bash
g++ -std=c++11 -o3 -o prog.x main.cpp sampler.cpp system.cpp particle.cpp Hamiltonians/hamiltonian.cpp Hamiltonians/harmonicoscillator.cpp InitialStates/initialstate.cpp InitialStates/randomuniform.cpp Math/random.cpp WaveFunctions/simplegaussian.cpp WaveFunctions/wavefunction.cpp

