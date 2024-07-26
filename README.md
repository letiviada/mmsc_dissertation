# MSc Mathematical Modelling and Scientific Computing, 2024


This repository contains all the codes used to complete my dissertation. The project involves Differential-Algebraic Equations (DAE) systems, with several 'easy' examples. The repository is organized into folders, each containing specific examples.

## Table of Contents
1. [Introduction](#introduction)
2. [Folder Structure](#folder-structure)

## Introduction
This project contains the codes used for solving a Differential-Algebraic Equations (DAE) system defining a multiscale model for filtration. It contains a study of the different variables for different parameter regimes and a pipeline to use machine learning for filtration. Studies the accuracy of three regression different models: polynomial, randoom forest, and gradeint boosting and how the sample size of training affects the estimation. The sampling is done using ranodom sampling and latin hypercube. Moreover, different optimization problems are studied and machine learning of these implemented. We have been able too demonstrate the efficiency of machine learning methods too study forward and backward problems. These methods simplify the computational cost significantly, as a task that could take hours is solved within seconds.

## Folder Structure
The repository is organized as follows:
```bash
├── multiscale
│   ├── __init__.py
│   ├── check_data.py
│   ├── combine_results.py
│   ├── figures
│   ├── macroscale
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── macro_main.py
│   │   ├── macro_model.py
│   │   ├── macro_plots.py
│   │   └── macro_solver.py
│   ├── microscale
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── micro_compute.py
│   │   ├── micro_functions.py
│   │   └── micro_main.py
│   ├── params_plots.sh
│   ├── performance_indicators
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── perfo_indicators.py
│   │   └── performance_indicators.py
│   ├── performance_plots.py
│   ├── plotting
│   │   ├── __init__.py
│   │   ├── create.py
│   │   ├── plots_data_ml.py
│   │   ├── plots_for_conductance.py
│   │   ├── plots_for_outputs.py
│   │   ├── plots_varying_alpha_beta.py
│   │   ├── save.py
│   │   └── style.py
│   ├── results
│   ├── run_models.sh
│   ├── sweep_params_plots.py
│   └── utils
│       ├── __init__.py
│       ├── get_functions.py
│       ├── get_k_and_j.py
│       ├── interpolate_functions.py
│       └── load_and_save.py
├── regression
│   ├── __init__.py
│   ├── data_plots.py
│   ├── figures
│   ├── model_plots.py
│   ├── model_plots2.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── choose_model.py
│   │   ├── model_eval.py
│   │   └── model_train.py
│   ├── models_gradient_boosting
│   ├── models_polynomial
│   ├── models_random_forest
│   ├── optimization
│   │   ├── __init__.py
│   │   ├── opt_ml_model.py
│   │   ├── opt_throughput
│   │   └── opt_time
│   ├── sample_size_study
│   ├── __init__.py
│   │   ├── sample_plots.py
│   │   ├── sample_size.py
│   └── utils
│       ├── __init__.py
│       ├── get_ratios.py
│       ├── help_functions.py
│       ├── obtain_and_clean_data.py
│       ├── process_data_throughput.py
│       ├── save_and_open_model.py
│       └── treat_data.py
└──



```