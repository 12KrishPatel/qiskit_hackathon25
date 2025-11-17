from qiskit import Aer
from qiskit.visualization import plot_circuit_layout, plot_histogram
from qiskit.algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Build QUBO 

