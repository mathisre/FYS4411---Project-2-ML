#include <iostream>
#include <cmath>
#include <vector>
#include "sampler.h"
#include "system.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"
#include <string>
#include <fstream>
#include <iomanip>
using std::cout;
using std::endl;
std::ofstream ofile;


Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
}

void Sampler::sample(double GibbsValue, bool acceptedStep, bool interaction, vector<double> X, vector<double> Hidden, vector<double> a_bias, vector<double> b_bias, vector<std::vector<double>> w) {
    //Function to sample the data
    if (m_stepNumber == 0) {
        m_cumulativeEnergy          = 0;
        m_cumulativeEnergySquared   = 0;
        m_cumulativeWFderiv         = 0;
        m_cumulativeWFderivMultEloc = 0;
    }


    m_energy = m_system->getHamiltonian()->computeLocalEnergy(GibbsValue, interaction,X,Hidden,a_bias,b_bias,w);

    if ( (double)getStepNumber() >= m_system->getEquilibrationFraction() * getNumberOfMetropolisSteps() ){
        //sample if the system is at equilibrium
        if ( acceptedStep == true ){
            // Sampling of energy moved to metropolisstep
            m_acceptedNumber++;
            //cout<<"accepted"<<m_acceptedNumber<<endl;
        }

        //m_energy = m_system->getHamiltonian()->computeLocalEnergy(interaction,X,Hidden,a_bias,b_bias,w);

        m_cumulativeEnergy          += m_energy;
        // cout<<"---"<<m_energy<<endl;
        m_cumulativeEnergySquared   += m_energy * m_energy;

        vector<double> temp  (getDimensionOfGradient());
        vector<double> temp2 (getDimensionOfGradient());
        vector<double> G     (getDimensionOfGradient());

        G     = m_system->GradientParameters(GibbsValue, X,a_bias,b_bias,w);
        temp  = m_system->getCumulativeGradient();
        temp2 = m_system->getCumulativeEnGradient();

        //        cout<<"grad"<<getDimensionOfGradient()<<endl;
        //        cout<<temp.size()<<endl;

        for(int i = 0; i < getDimensionOfGradient(); i++){

            temp  [i] += G[i];
            temp2 [i] += m_energy * G[i];

        }

        m_system->setCumulativeGradient   (temp);
        m_system->setCumulativeEnGradient (temp2);

        //        m_cumulativeWFderiv         += m_WFderiv;
        //        m_cumulativeWFderivMultEloc += m_WFderivMultELoc;

        // Sometimes crashes
        //        m_system->oneBodyDensity();
        //       cout<<"1"<<endl;
    }

    m_stepNumber++;
    //cout<<m_cumulativeEnergy<<endl;
}

void Sampler::printOutputToTerminal(int cycle) {
    int     np    = m_system->getNumberOfParticles();
    int     nd    = m_system->getNumberOfDimensions();
    int     ms    = m_system->getNumberOfMetropolisSteps();
    int     p     = m_system->getNumberOfParameters();
    int     v     = m_system->getNumberOfVisibleNodes();
    int     h     = m_system->getNumberOfHiddenNodes();
    double  ef    = m_system->getEquilibrationFraction();
    double  ms_eq = ms - ef * ms;

    std::vector<double> pa = m_system->getWaveFunction()->getParameters();

    ofile.close();

    if(cycle == 0) {

        cout << endl;
        cout << "  -- System info -- " << endl;
        cout << " Number of particles  : " << np << endl;
        cout << " Number of dimensions : " << nd << endl;
        cout << " Number of visible nodes : " << v <<endl;
        cout << " Number of hidden nodes : " << h <<endl;
        cout << " Number of Metropolis steps run : 10^" << std::log10(ms) << endl;
        cout << " Number of equilibration steps  : 10^" << std::log10(std::round(ms*ef)) << endl;
        cout << endl;
        cout << "  -- Wave function parameters -- " << endl;
        cout << " Number of parameters : " << p << endl;
        //    for (int i=0; i < p; i++) {
        //        cout << " Parameter " << i+1 << " : " << pa.at(i) << endl;
        //    }
        cout << endl;
    }

    cout<<endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_energy << endl;
    cout << " St. dev: " << sqrt(m_cumulativeEnergySquared - m_energy*m_energy) / sqrt(ms_eq) << endl;
    cout << " Acceptance ratio: " << (double)m_acceptedNumber/ms_eq << endl;
    cout << " Number of cycle: "  <<cycle<<endl;
    cout << endl;

}



void Sampler::computeAverages(vector<double> &Gradient) {

    double frac = m_system->getNumberOfMetropolisSteps() * ( 1 - m_system->getEquilibrationFraction() );

    m_energy = m_cumulativeEnergy / (frac);
    m_cumulativeEnergySquared /= frac;

    vector<double> temp  (getDimensionOfGradient());
    vector<double> temp2 (getDimensionOfGradient());

    temp  = m_system->getCumulativeGradient();
    temp2 = m_system->getCumulativeEnGradient();

    double sum1 = 0;
    double sum2 = 0;

    for(int i = 0; i < m_system->getNumberOfParameters(); i++){

        sum1 = temp[i] / frac;
        sum2 = temp2[i] / frac;

        Gradient[i] = 2 * ( sum2 - m_energy * sum1 );
    }

}


void Sampler::openDataFile(std::string filename){
    if (filename != "0") ofile.open(filename);

    ofile << setprecision(12)<<fixed;
    ofile << setw(5)<<fixed;

}


void Sampler::writeToFile(){
    if (ofile.is_open()) ofile << m_energy << endl;
}

//getters and setters
int Sampler::getStepNumber() const
{
    return m_stepNumber;
}

int Sampler::getNumberOfMetropolisSteps() const
{
    return m_numberOfMetropolisSteps;
}

double Sampler::getWFderivMultELoc() const
{
    return m_WFderivMultELoc;
}

void Sampler::setWFderivMultELoc(double WFderivMultELoc)
{
    m_WFderivMultELoc = WFderivMultELoc;
}

double Sampler::getCumulativeWF() const
{
    return m_cumulativeWF;
}

void Sampler::setCumulativeWF(double cumulativeWF)
{
    m_cumulativeWF = cumulativeWF;
}

double Sampler::getWFderiv() const
{
    return m_WFderiv;
}

void Sampler::setWFderiv(double WFderiv)
{
    m_WFderiv = WFderiv;
}

double Sampler::getCumulativeWFderiv() const
{
    return m_cumulativeWFderiv;
}

void Sampler::setCumulativeWFderiv(double cumulativeWFderiv)
{
    m_cumulativeWFderiv = cumulativeWFderiv;
}

double Sampler::getCumulativeWFderivMultEloc() const
{
    return m_cumulativeWFderivMultEloc;
}

void Sampler::setCumulativeWFderivMultEloc(double cumulativeWFderivMultEloc)
{
    m_cumulativeWFderivMultEloc = cumulativeWFderivMultEloc;
}

int Sampler::getAcceptedNumber() const
{
    return m_acceptedNumber;
}

void Sampler::setAcceptedNumber(int acceptedNumber)
{
    m_acceptedNumber = acceptedNumber;
}

void Sampler::setStepNumber(int stepNumber)
{
    m_stepNumber = stepNumber;
}

void Sampler::setEnergy(double energy)
{
    m_energy = energy;
}

void Sampler::updateEnergy(double dE){
    m_energy += dE;
}

double Sampler::getCumulativeEnergy() const
{
    return m_cumulativeEnergy;
}

double Sampler::getCumulativeEnergySquared() const
{
    return m_cumulativeEnergySquared;
}

void Sampler::setCumulativeEnergySquared(double cumulativeEnergySquared)
{
    m_cumulativeEnergySquared = cumulativeEnergySquared;
}

int Sampler::getDimensionOfGradient() const
{
    return m_dimensionOfGradient;
}

void Sampler::setDimensionOfGradient(int dimensionOfGradient)
{
    m_dimensionOfGradient = dimensionOfGradient;
}
