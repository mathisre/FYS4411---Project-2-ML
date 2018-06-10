# FYS4411---Project-2-ML

Inside src lies the program used to compute the ground state energy a systems of 1 or 2 fermions interacting and non in a harmonic potential using machine learning and neural networks. The program has three settings, to use standard Metropolis sampling, Importance samplinng or Gibbs sampling. This is setby commenting and uncommenting a line in main.cpp. Parameters such as number of particles, number of dimensions, number of Monte carlo steps, number of SGD cycles, sigma, etc, are set in main.cpp. If you wish to study non-interacting fermions, make sure to set  bool interaction = false. Alter sigma in RandomUniform:setupInitialState to set the spread of the distribution of initial parameter. For our studies we have used sigma=0.5. 

The program can be compiled using ./compile.sh and then run using ./prog.x. By default there are no command line arguments, but this is can be easily changed.

Moreover, inside src (folder "blocking") there is a python code to perform the blocking method to compute the statistical error on a dataset which takes into account the correlations between data. It requires as input that dataset in the function loadtxt().

In folder "figures", there are all the plots made for the report.

In folder "data", there are some results which we find important i.e. the ground state energy computed with the optimal parameters after (300 SGD cycles) in the interacting case with importance sampling and brute force metropolis. We also insert as examples some data about the non-interacting case.

