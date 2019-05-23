# KIVA - Collaborative filtering recommeder system for microfinances
A recommender system for [kiva.org](https://www.kiva.org/) micro-loans based on collaborative filtering.

- [About](#About)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Additional info](#Additional-info)

## About

- TODO

The project consists of three [jupyter notebooks]:
* kiva-cf
    - Initial data cleanup and formatting.
    - Introductory exploration of implicit framework capabilities and application to the KIVA dataset.
* kiva-cf-polara
    - Translation of done work to polara framework.
    - Data reformatting.
    - Creation of an evaluation environment.
* result-analysis
    - Analysis and visualization of evaluation results used for interpretation.

## Installation

To start the notebook locate inside the project root directory and run:

`jupyter notebook`


## Dataset

Full data snapshot can be found [here](https://build.kiva.org/docs/data/snapshots).

## Additional info

All needed csv files should be located in /additional-kiva-snapshot directory.

All pickle files should be located in /pickle directory.

Evaluation results are located in /eval_results directory.

[//]: # (Links)
[jupyter notebooks]: https://github.com/jupyter/notebook