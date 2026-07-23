import yaml
import os

def load_parameters():
    path = os.path.join(os.path.dirname(__file__), 'parameters.yml')
    with open(path) as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
    return params

import numpy as np
def rotate(axis, theta):
    c, s = np.cos(np.pi/180 * theta), np.sin(np.pi/180 * theta)

    if axis == 0:
        mat = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 1:
        mat = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif axis == 2:
        mat = [[c, -s, 0], [s, c, 0], [0, 0, 1]]

    return np.array(mat)

from . import backend
from .backend import GPU_AVAILABLE, use_gpu, use_cpu, using_gpu, asnumpy, asarray
from .coils import LinearQuadrupole, QuadrupoleCoils
from .mot import SixBeam, GratingMOT
from .beams import UniformBeam, GaussianBeam
from .dipole import DipoleTrap, OpticalLattice
from .levels import LevelDynamics
from .integration import Solver, integrate, generate_initial_conditions
from . import diagnostics
from . import recapture
from .api import (atom, rubidium87, six_beam_mot, dipole_trap, thermal_cloud,
                  set_backend, Experiment, ATOMS)
