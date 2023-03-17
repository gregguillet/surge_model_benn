# An open-source object-oriented model for investigating glacier surges

This repository contains a python implementation of the Benn et al (2019) model for mass-enthalpy balance of glacier surges.

The original model used in the paper is in MATLAB thus hampering open-access.
The aim of this project is reimplement the mass-enthalpy balance model in an object-oriented fashion providing open flexibility, readability and ease of use.

This repository contains two-files:
- *glacier_surge_model.py* contains all the physics including:
  - Different formulations of the glacier drainage system - refer to original publication for more:
    - Efficient, fully connected network
    - Single component drainage system
  - Different basal sliding relationships:
    - Weertman-type
    - Zoet and Iverson (2020)
    - Beaud et al (2022)

- *glacier_surge_run_plots.py*:
  - Quiver plots
  - Phase state plots
  - Time sereies

The model is calibrated to reproduce figures available in the original Benn et al. (2019) paper.

Different models can be called as _Model_ objects within the *PLOTS* file in order to explore the effect of different paramter values onto model results and performances.

