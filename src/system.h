#pragma once
#include <vector>
#include <string>


class System {
public:
    bool metropolisStepImportance   (double GibbsValue, std::vector<double> &X, std::vector<double> Hidden, std::vector<double> a_bias, std::vector<double> b_bias, std::vector<std::vector<double>> w);
    bool metropolisStepBruteForce   (double GibbsValue, std::vector<double> &X, std::vector<double> Hidden, std::vector<double> a_bias, std::vector<double> b_bias, std::vector<std::vector<double> > w);
    void runMetropolisSteps         (std::string method, std::vector<double>&Gradient, int numberOfMetropolisSteps, bool interaction, std::vector<double> X, std::vector<double> Hidden, std::vector<double> a_bias, std::vector<double> b_bias, std::vector<std::vector<double> > w);
    void setNumberOfParticles       (int numberOfParticles);
    void setNumberOfDimensions      (int numberOfDimensions);
    void setStepLength              (double stepLength);
    void setEquilibrationFraction   (double equilibrationFraction);
    void setHamiltonian             (class Hamiltonian* hamiltonian);
    void setWaveFunction            (class WaveFunction* waveFunction);
    void setInitialState            (class InitialState* initialState);
    void openDataFile               (std::string filename);
    void printOut                   (int cycle);
    double computedistance          (int i);
    double computedistanceABS       (int i, int j);
    void writeToFile(std::vector<double> X, std::vector<double>& a_bias, std::vector<double>& b_bias, std::vector<std::vector<double>>& w);
    void openFile(std::string filename);
   std::vector<std::vector<double>>    computematrixdistance(std::vector<double> &X);
    class WaveFunction*             getWaveFunction()   { return m_waveFunction; }
    class Hamiltonian*              getHamiltonian()    { return m_hamiltonian; }
    class Sampler*                  getSampler()        { return m_sampler; }
    std::vector<class Particle>&     getParticles()      { return m_particles; }
    int getNumberOfParticles()          { return m_numberOfParticles; }
    int getNumberOfDimensions()         { return m_numberOfDimensions; }
    int getNumberOfMetropolisSteps()    { return m_numberOfMetropolisSteps; }
    double getEquilibrationFraction()   { return m_equilibrationFraction; }

//    double gradientDescent(double initialAlpha, std::string filename, int maxIterations);

    double getinteractionSize() const;
    void setinteractionSize(double interactionSize);

    double getTimeStep() const;
    void setTimeStep(double timeStep);

    double getSqrtTimeStep() const;
    void setSqrtTimeStep(double sqrtTimeStep);

    std::vector<std::vector<double> > getDistanceMatrix() const;
    void setDistanceMatrix(const std::vector<std::vector<double> > &distanceMatrix);
   void updateDistanceMatrix(std::vector<double> X, int randparticle);
    double getDistanceMatrixij(int i, int j) const;

    double getPsiOld() const;
    void setPsiOld(double psiOld);

    double findEnergyDerivative();

     std::vector<double> getQuantumForce() const;
    void setQuantumForce(const  std::vector<double> &QuantumForce);
    void updateQuantumForce( std::vector<std::vector<double>> deltaQuantumForce, bool subtract);


    std::vector<int> getHistogram() const;
    void setHistogram();

    void printOneBodyDensity(std::string filename);

    int getBins() const;
    void setBins(int bins);

    double getBucketSize() const;
    void setBucketSize(double bucketSize);
        void oneBodyDensity();


        int getNumberOfVisibleNodes() const;
        void setNumberOfVisibleNodes(int numberOfVisibleNodes);

        int getNumberOfHiddenNodes() const;
        void setNumberOfHiddenNodes(int numberOfHiddenNodes);

        double getSigma() const;
        void setSigma(double sigma);

        double getSigma_squared() const;
        void setSigma_squared(double sigma_squared);

        double getLearningRate() const;
        void setLearningRate(double learningRate);

        void StochasticGradientDescent(std::vector<double> &a_bias, std::vector<double> &b_bias, std::vector<std::vector<double> > &w);
        int getNumberOfParameters() const;
        void setNumberOfParameters(int numberOfParameters);


        void StochasticGradientDescent(std::vector<double> Gradient,std::vector<double> X, std::vector<double> &a_bias, std::vector<double> &b_bias, std::vector<std::vector<double> > &w);
        std::vector<double> GradientParameters(double GibbsValue, std::vector<double> X, std::vector<double> &a_bias, std::vector<double> &b_bias, std::vector<std::vector<double> > &w);
        std::vector<double> getCumulativeGradient() const;
        void setCumulativeGradient(const std::vector<double> &cumulativeGradient);

        std::vector<double> getCumulativeEnGradient() const;
        void setCumulativeEnGradient(const std::vector<double> &cumulativeEnGradient);

        std::vector<double> getGradient() const;

        void setGradient(const std::vector<double> &Gradient);

        std::vector<double> getEnGradient_average() const;
        void setEnGradient_average(const std::vector<double> &EnGradient_average);

        bool GibbsSampling(std::vector<double> &X, std::vector<double> Hidden, std::vector<double> a_bias, std::vector<double> b_bias, std::vector<std::vector<double> > w);
        int computeIndex(int index);
private:
        int                             m_numberOfParticles = 0;
        int                             m_numberOfDimensions = 0;
    int                             m_numberOfMetropolisSteps = 0;
    double                          m_equilibrationFraction = 0;
    double                          m_psiOld = 0;
    double                          m_stepLength = 0.1;
    double                          m_interactionSize = 0;
    double                          m_timeStep = 0;
    double                          m_sqrtTimeStep = 0;
    int                             m_bins = 0;
    double                          m_bucketSize = 0;

    int m_numberOfVisibleNodes =0;
    int m_numberOfHiddenNodes=0;
    double m_sigma =0;
    double m_sigma_squared =0;

double m_learningRate=0;
int m_numberOfParameters=0;

    std::vector<int>                m_histogram;
    std::vector<double> m_QuantumForce;
    class WaveFunction*             m_waveFunction = nullptr;
    class Hamiltonian*              m_hamiltonian = nullptr;
    class InitialState*             m_initialState = nullptr;
    class Sampler*                  m_sampler = nullptr;
    class conjugateGradient*        m_conjugateGradient = nullptr;
    //class Random*                   m_random = nullptr;
    std::vector<class Particle>     m_particles = std::vector<class Particle>();
    std::vector<std::vector<double>>     m_distanceMatrix = std::vector<std::vector<double>>();

    std::vector <double> m_cumulativeGradient=std::vector<double>();
    std::vector <double> m_cumulativeEnGradient=std::vector<double>();
    std::vector <double> m_Gradient=std::vector<double>();
    std::vector <double> m_EnGradient_average=std::vector<double>();

};

