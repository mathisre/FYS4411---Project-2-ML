#include <iostream>
#include <random>
#include <cmath>
#include "system.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/simplegaussian.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "Math/random.h"
#include <chrono>
#include <string>
#include <vector>

using namespace std;

int main(){


    int numberOfParticles   = 2;
    int numberOfDimensions  = 2;
    int numberOfHiddenNodes = 1;

    string method = "MetropolisBruteForce";
//    string method = "MetropolisImportance";
//    string method = "Gibbs";

    // Interaction: put "false" to consider non-interacting fermions
    bool interaction = true;

    double learning_rate    = 0.2;
    double timeStep         = 0.5;      // Importance sampling time step
    double stepLength       = 0.5;      // Metropolis step length.
    double sigma            = 1.00;     // Sigma from energy function
    double omega            = 1;        // Oscillator frequency.

    int TotalNumberOfCycles = 300;      // Number of SGD cycles

    int numberOfSteps       = (int) 300000;   // NUmber of Monte Carlo steps per SGD cycle
    double equilibration    = 0.2;            // Amount of the total steps used for equilibration

    int numberOfVisibleNodes = numberOfDimensions*numberOfParticles;    
    int numberOfParameters = numberOfVisibleNodes + numberOfHiddenNodes
            + numberOfVisibleNodes * numberOfHiddenNodes;

    // Vectors for variational parameters
    std::vector<double> X(numberOfVisibleNodes);
    std::vector<double> Hidden=std::vector<double>(numberOfHiddenNodes);
    std::vector<double> a_bias=std::vector<double>(numberOfVisibleNodes);
    std::vector<double> b_bias=std::vector<double>(numberOfHiddenNodes);
    std::vector<std::vector<double>> w(numberOfVisibleNodes, vector<double>(numberOfHiddenNodes));//=std::vector<std::vector<double>>();
    vector <double> Gradient(numberOfParameters);

    System* system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setInitialState             (new RandomUniform(system, numberOfParticles, numberOfDimensions, numberOfHiddenNodes, numberOfVisibleNodes, sigma, X, Hidden, a_bias, b_bias, w, timeStep,numberOfParameters));
    system->setWaveFunction             (new SimpleGaussian(system));
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);

    // Data files for SGD cycle and final MC run
    string filename_cycle_data;
    string finalFilename;
    if (interaction == true)
    {
        // SGD data file
        if( method == "MetropolisBruteForce" ) filename_cycle_data =  "../data/bruCycleDataI_" + to_string(stepLength) + "_n_" + to_string(learning_rate) +  "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
        if( method == "MetropolisImportance" ) filename_cycle_data = "../data/impCycleDataI_" + to_string(timeStep) + "_n_" + to_string(learning_rate)+ "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
        if( method == "Gibbs"                ) filename_cycle_data =  "../data/gibCycledataI2_s_" + to_string(sigma) +"_n_" + to_string(learning_rate)+  "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";

        // Instantaneous energy data file (of bigger run after SGD)
        if( method == "MetropolisBruteForce" ) finalFilename =  "../data/finalBruCycleDataI_" + to_string(stepLength) + "_n_" + to_string(learning_rate)+ "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
        if( method == "MetropolisImportance" ) finalFilename = "../data/finalImpCycleDataI_" + to_string(timeStep) + "_n_" + to_string(learning_rate)+ "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
        if( method == "Gibbs"                ) finalFilename =  "../data/finalGibCycledataI_s_" + to_string(sigma) +"_n_" + to_string(learning_rate)+ "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
    }
    else
    {
        // SGD data file
        if( method == "MetropolisBruteForce" ) filename_cycle_data =  "../data/bruCycleDataNoI_" + to_string(stepLength) + "_n_" + to_string(learning_rate) +  "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
        if( method == "MetropolisImportance" ) filename_cycle_data = "../data/impCycleDataNoI_" + to_string(timeStep) + "_n_" + to_string(learning_rate)+ "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
        if( method == "Gibbs"                ) filename_cycle_data =  "../data/gibCycledataNoI_s_" + to_string(sigma) + "_n_" + to_string(learning_rate)+  "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";

        // Instantaneous energy data file (of bigger run after SGD)
        if( method == "MetropolisBruteForce" ) finalFilename =  "../data/finalBruCycleDataNoI_" + to_string(stepLength) + "_n_" + to_string(learning_rate)+ "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
        if( method == "MetropolisImportance" ) finalFilename = "../data/finalImpCycleDataNoI_" + to_string(timeStep) + "_n_" + to_string(learning_rate)+ "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";
        if( method == "Gibbs"                ) finalFilename =  "../data/finalGibCycledataNoI_s_" + to_string(sigma) +"_n_" + to_string(learning_rate)+ "_Np_" + to_string(numberOfParticles) + "_Nd_" + to_string(numberOfDimensions) + "_NH_" + to_string(numberOfHiddenNodes) +  "_w_" + to_string(omega) +  ".dat";

    }

    system->openFile(filename_cycle_data);
    auto start = std::chrono::system_clock::now();
    for(int cycles = 0; cycles < TotalNumberOfCycles; cycles ++){
        system->setLearningRate           (learning_rate);
        system->setNumberOfParameters     (numberOfParameters);
        system->runMetropolisSteps        (method, Gradient,numberOfSteps,interaction, X, Hidden, a_bias, b_bias, w);
        system->StochasticGradientDescent (Gradient,X,a_bias,b_bias,w);
        system->printOut                  (cycles);
        system->writeToFile(X,a_bias,b_bias,w);
    }

    if (interaction == false){
        system->openDataFile                (finalFilename);
        int finalNumberOfSteps = 1.5e+6;
        system->runMetropolisSteps        (method, Gradient,finalNumberOfSteps,interaction, X, Hidden, a_bias, b_bias, w);
        system->printOut                  (TotalNumberOfCycles);
        system->writeToFile(X,a_bias,b_bias,w);
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << " Computation time = " << diff.count() / 60.0 << " min\n" << endl; //display run time
    return 0;
}


