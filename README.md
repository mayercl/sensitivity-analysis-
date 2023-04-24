# Sensitivity analysis of the Hannay model 
by Caleb Mayer, Olivia Walch, Daniel B. Forger, and Kevin Hannay.

This code is used to generate the results and figures in "Impact of light schedules and model parameters on the circadian outcomes of individuals" by Caleb Mayer, Olivia Walch, Daniel B. Forger, and Kevin Hannay. We examine interactions between the parameters in the Hannay model and different lightning schedules. 

The code used to generate the main text figures are in Jupyter notebooks (.ipynb), with a different notebook for every main text figure. Additionally, we include a notebook to run all of the main text figures at once. Helper functions and modules are included in .py files.  

## Overview 
After installing Jupyter, you can run the notebooks individually in order to generate the results for the individual figures. Parameter values (both parameters in the model and others) can be interactively changed to see the effects on the results. 

For more questions or details, see the notebook descriptions below or contact mayercl@umich.edu. 

## The notebooks 
We provide a brief description for each of the Jupyter notebooks included. 

clean_figure1.ipynb: this notebook generates a methods/schematic figure. This consists of actograms of the light schedules, a (rescaled) schematic of what parameter perturbations can look like, and example model outputs under parameter perturbations. 

clean_figure2.ipynb: this notebook generates a first hessian-based sensitivity figure. We compute the Frobenius norm of a hessian matrix (see the manuscript and/or hessian_normalized.py for details) to define an overall sensitivity measurement, as well as computing the prinipal eigenvectors of this hessian as individual parameter sensitivity metrics. 

clean_figure3.ipynb: this notebook looks at model output changes from one-at-a-time parameter perturbations. Performing this analysis entails visualizing model output changes for different light schedules when each parameter is perturbed in turn.

clean_figure4.ipynb: this notebook examines the application of the hessian-based sensitivity metric to different light levels. To generate this figure, we compare the overall and individual sensitivity metrics from lower light levels ranging from 0 to 950 lux for the six schedules of interest. 

clean_figure5.ipynb: this notebook performs a parameter recovery analysis. This consists of determining the extent to which the actual value of parameters can be recovered given a noisy light schedule. 

clean_sensitivity_all.ipynb: this notebooks combines the results from code and visualizations in figures 1-5 into one larger notebook. 
