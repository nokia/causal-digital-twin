## `repository:` Causal Digital Twin

This repository contains implementation code needed to reproduce the data generation process of the framework described in the:

# `paper:` Towards Building a Digital Twin of Complex System using Causal Modelling
by Luka Jakovljevic<sup>1,2</sup>, Dimitre Kostadinov<sup>1</sup>, Armen Aghasaryan<sup>1</sup> and Themis Palpanas<sup>2</sup>

<sup>1</sup>*Nokia Bell Labs, France*\
<sup>2</sup>*University of Paris, France*

that is presented on **The 10th International Conference on Complex Networks and their Applications**\
*in Madrid, Spain (November 30 - December 2, 2021)* ["complexnetworks.org"](https://complexnetworks.org/)

and appears in [**Volume 1015 of Springer - Studies in Computational Intelligence Series**](https://link.springer.com/chapter/10.1007/978-3-030-93409-5_40) 📘

contact: luka.jakovljevic@nokia.com

## `usage:` Mimicking Faulty System Behavior

Complex systems, such as **5G** telecommunication networks, generate thousands of information about the system state each minute.
The described causal model is able to capture the dependencies between observable **network alerts** but more importantly,
to mimic the system **faulty behavior**, by encoding the appearance, propagation and persistence of faults 🤖

Why is this important? Because such a model can assist network experts in **reasoning on the state of real system**, given partial observations.
Furthermore, it could allows generating **labelled synthetic alerts**, in order to benchmark causal discovery and
network diagnosis techniques, to ensure that they will work with unseen real data with similar characteristics.
Lastly, to create previously unseen faulty scenarios in the system (**counterfactual reasoning**) 🧠

![alt text](https://github.com/nokia/causal-digital-twin/blob/main/causal_digital_twin.JPG)

## `get started:` Content of the repository
folder `datasets` contains samples of synthetic alerts with known causal relations

module `causal_digital_twin.py` contains functions that construct a Causal Model

notebook `Demo_one_dataset.ipynb` demonstrates the creation of one dataset (one Causal Model i.e. input for a Digital Twin)

notebook `Generate_all_datasets.ipynb` allows simulation of all datasets used in the paper

## Important `functions`:
`generate_DAG (...)` generates a random DAG of desired size and edge probability 

`parametrize_DAG (...)` parametrizes DAG with SCM probabilities described in paper

`time_series (...)` builds Causal Model (SCM) and synthesizes time series of desired length as depicted on Figure 2 in paper

## `Python:` Required Packages
* numpy
* pandas
* networkx
* matplotlib.pyplot
* random
* sklearn

## Licence
Please cite the above mentioned paper when using the framework.

Code is published under 
**BSD 3-Clause License**<br>
(for more info read ["LICENSE file"](https://github.com/nokia/causal-digital-twin/blob/main/LICENSE))

Copyright (c) 2021, Nokia<br>
All rights reserved.
