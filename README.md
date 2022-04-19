# Project PROMAP
Project PROMAP (acronym for Probabilistic Prediction of Material Properties) is a feasibility study which seeks to merge Artificial Intelligence (AI) tools with Bayesian statistics to yield probabilistic estimates of target features under scarse data/information. The context of this study is with regards to the following Nuclear material properties: 1) Creep rupture; and 2) Tensile properties. This repository contains the MATLAB codes required to perform such analysis.

## Instructions:
The following main MATLAB files are to be executed in the following order:
<details>
<summary> 1) PROMAP_NIMS_data_enchancement.m </summary> 
This file executes the method of generating synthetic data from the processed material property data from National Institute for Materials Science (NIMS).
</details>
<details>
<summary> 2) PROMAP_ANN_Training.m </summary> 
This file executes the training of the various Artificial Neural Network (ANN) surrogate models with the synthetic data and computes the R2-score of the corresponding ANN relative to the experimental data.
</details>
<details>
<summary> 3) PROMAP_AdaptiveBMS.m </summary> 
This file executes the method of Adaptive Bayesian Model Selection to generate probabilistic estimates of the key target features by the set of Artificial Neural Networks (ANNs).
</details>

The "Additional Files" folder contains the following codes which can be executed at the interest of the users. 
<details>
<summary> 1) PROMAP_NIMS_data_processing.m </summary> 
This file executes the method of processing the material property data obtained from the National Institute for Materials Science (NIMS).
</details>
<details>
<summary> 2) PROMAP_INCEFA_data_processing.m </summary> 
This file executes the method of processing the material fracture data obtained from the INcreasing safety in NPPs by Covering gaps in Environmental Fatigue Assessment (INCEFA) project funded by H2020 (an EU funding programme for research and innovation).
</details>

The above 2 files also serve as a tutorial on the "cleaning" and processing of raw data-set.

## Reference(s):
* A. Lye, N. Prinja, and E. Patelli (2022). Probabilistic AI for Prediction of Material Properties (PROMAP). *In the Proceedings of the Advanced Nuclear Skills and Innovation Campus Showcase 2022*. doi: [10.13140/RG.2.2.32427.31527](https://doi.org/10.13140/RG.2.2.32427.31527)

* A. Lye, N. Prinja, and E. Patelli (2022). ANSIC Case Study - Project PROMAP. *In the Proceedings of the Advanced Nuclear Skills and Innovation Campus Showcase 2022*. doi: [10.13140/RG.2.2.12632.16645](https://doi.org/10.13140/RG.2.2.12632.16645)

* A. Lye, N. Prinja, and E. Patelli (2022). Probabilistic Artificial Intelligence Prediction of Material Properties for Nuclear Reactor Designs. *In the Proceedings of the 32nd European Safety and Reliability Conference, 1*, Dublin, Ireland. doi: []()

## Author:
* Name: Adolphus Lye
* Contact: adolphus.lye@liverpool.ac.uk
* Affiliation: Insitute for Risk and Uncertainty, University of Liverpool
