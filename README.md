# MSc Mathematical Modelling and Scientific Computing, 2024


This repository contains all the codes used to complete my dissertation. The project involves Differential-Algebraic Equations (DAE) systems, with several 'easy' examples. The repository is organized into folders, each containing specific examples.

## Table of Contents
1. [Introduction](#introduction)
2. [Folder Structure](#folder-structure)
3. [Examples](#examples)
    1. [Example 1: Simple DAE System](#example-1-simple-dae-system)
    2. [Example 2: DAE system with two differentiable variables](#example-2-dae-system-with-two-differentiable-variables)
4. [Trouton Model](#trouton-model)


## Introduction
This project contains the codes used for solving various Differential-Algebraic Equations (DAE) systems. 

## Folder Structure
The repository is organized as follows:
```bash
code_dissertation/
│
├── dae_examples/
│ ├── dae_easy/
│ ├── dae_example2/
│ 
├── trouton_model/
│
└── tensor_code/
```
## Examples

### Example 1: Simple DAE system
This example involves solving a simple DAE system. The system is defined as follows:

$$
\begin{align*}
\frac{dx}{dt} &= \frac{1}{2}, \\
 z-2x &= 0.
\end{align*}
$$

### Example 2: DAE system with two differentiable variables
This example involves solving a medium complexity DAE system. The system is defined as follows:

$$
\begin{align*}
\frac{dx}{dt} &= \frac{1}{2}, \\
\frac{dy}{dt} &= y, \\
 z-2x &= 0.
\end{align*}
$$


## Trouton Model
This systems defines the sheet thickness ($h$), and axial velocity ($u$) of a glass sheet when it is streched. The system is defined as follows:

$$
\begin{align*}
\frac{\partial h}{\partial t} + \frac{\partial (uh)}{\partial x} &= 0 \\
\frac{\partial }{\partial x}\left (h\frac{\partial u}{\partial x}\right) &= 0\\
h(t=0,x) &= 1, \\
h(t,x=0) &= 1, \\
u(t,x=0) &= 1, \\
u(t,x=1) &= 10. \\
\end{align*}
$$
