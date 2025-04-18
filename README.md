<h2 align="center">Groundwater flow equation: Finite-difference forward, adjoint, and gradient operators with PyTorch wrapper</h2>

This repository contains the code for implementing the finite-difference
forward, adjoint, and gradient operators for the groundwater flow equation.
The operators are implemented via
[Devito](https://github.com/devitocodes/devito), a finite-difference
domain-specific language for solving partial differential equations that
generates optimized C code depending on the target architecture.


## Installation

Clone the repository and install the package in editable mode.
```bash
# Create a new conda environment (optional).
conda create --name groundwater "python<=3.12"
conda activate groundwater

# Clone the repository and install the package in your Python environment.
git clone ttps://github.com/alisiahkoohi/groundwater
cd groundwater/
pip install -e .
```

Visit the [Devito installation
guide](https://www.devitoproject.org/download.html) for more information
on setting up the environment variables to fully utilize the
parallelization capabilities of Devito.

## Questions

Please contact [alisk@rice.edu](mailto:alisk@rice.edu) for questions.

## Authors

Ali Siahkoohi and Mathias Louboutin
