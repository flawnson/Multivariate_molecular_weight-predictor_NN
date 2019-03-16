# multivariate_molecular_weight_predictor_NN

## Overview
Project Maverick is a molecular weight predictor, not unlike one I've made previously, although this project showed significant improvements, due in part to the structural improvements made to this model. Maverick receives 6 different molecular properties including

1. Polar Area
2. Complexity Score
3. Heavy CNT
4. Number of Hydrogen Bond Donors
5. Number of Hydrogen Bond Acceptors
6. Rotatable Bonds

To predict the molecular weight of a given molecule. The neural network model is small and trained on a large dataset over a relatively high number of epochs

## Challenges
Challenges were mainly attributed to deciding whether or not normalization techniques were necessary, and eventually creating a UDF to return a molecular weight.

## Sample Images
![](https://www.wikihow.com/images/thumb/4/4f/Calculate-Molecular-Weight-Step-4.jpg/aid4627737-v4-728px-Calculate-Molecular-Weight-Step-4.jpg)

## Applications
Project Maverick has potential uses in chemistry, where molecular weights are usually calculated, but this type of model is better suited for predicting the molecular weight of hypothetical molecules.

## Future
Improving clarity by using pandas, matplotlib, and seaborn.

Upwards and onwards, always and only! :rocket:
