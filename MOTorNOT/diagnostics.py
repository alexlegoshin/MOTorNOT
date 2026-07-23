''' Ensemble diagnostics for MOTorNOT simulations: temperature, cloud size,
    and velocity/position distributions, plus simple matplotlib plots.

    Velocities are in m/s, positions in m. Mass may be given either as a value
    in kg or as an `atom` dict (mass in amu) via `atom_mass`.
'''
import numpy as np
from scipy.constants import k as kB, physical_constants
amu = physical_constants['atomic mass constant'][0]


def atom_mass(atom):
    ''' Mass in kg from an `atom` dict (mass in amu). '''
    return atom['mass'] * amu


def temperature(V, mass):
    ''' Kinetic temperature from velocities.

        Args:
            V (ndarray): velocities, shape (N,) for 1D or (N, d) for d axes.
            mass (float): atomic mass in kg.
        Returns:
            float: T = m <v^2> / (d * kB), averaging over atoms and the d axes.
    '''
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        v2 = V**2
    else:
        v2 = np.sum(V**2, axis=1)
        return mass * np.mean(v2) / (V.shape[1] * kB)
    return mass * np.mean(v2) / kB


def temperature_per_axis(V, mass):
    ''' Per-axis temperatures T_i = m <v_i^2> / kB. Returns array of length d. '''
    V = np.atleast_2d(np.asarray(V, dtype=float))
    return mass * np.mean(V**2, axis=0) / kB


def rms_radius(X, center=None):
    ''' RMS cloud radius (scalar) and per-axis RMS size. '''
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if center is None:
        center = X.mean(axis=0)
    d = X - center
    per_axis = np.sqrt(np.mean(d**2, axis=0))
    radial = np.sqrt(np.mean(np.sum(d**2, axis=1)))
    return radial, per_axis


def maxwell_boltzmann_speed_pdf(v, T, mass):
    ''' 3D Maxwell-Boltzmann speed distribution, for overlaying on histograms. '''
    v = np.asarray(v, dtype=float)
    return (mass / (2 * np.pi * kB * T))**1.5 * 4 * np.pi * v**2 \
        * np.exp(-mass * v**2 / (2 * kB * T))


# ---------------- plotting (matplotlib, headless-friendly) ----------------
def plot_velocity_distribution(V, mass, path=None, bins=50, overlay_mb=True):
    ''' Histogram of speed |v| with optional Maxwell-Boltzmann overlay. '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    V = np.atleast_2d(np.asarray(V, dtype=float))
    speed = np.linalg.norm(V, axis=1)
    T = temperature(V, mass)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(speed, bins=bins, density=True, alpha=0.7, color='steelblue',
            label='simulation')
    if overlay_mb:
        vv = np.linspace(0, speed.max() * 1.05, 300)
        ax.plot(vv, maxwell_boltzmann_speed_pdf(vv, T, mass), 'r-', lw=2,
                label='Maxwell-Boltzmann, T=%.1f uK' % (T * 1e6))
    ax.set_xlabel('speed |v| (m/s)')
    ax.set_ylabel('probability density')
    ax.set_title('Velocity distribution (T = %.2f uK)' % (T * 1e6))
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return T


def plot_position_distribution(X, path=None, bins=50):
    ''' Histograms of x, y, z positions (mm). '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    X = np.atleast_2d(np.asarray(X, dtype=float))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, lab) in enumerate(zip(axes, 'xyz')):
        ax.hist(X[:, i] * 1e3, bins=bins, density=True, alpha=0.7, color='seagreen')
        ax.set_xlabel('%s (mm)' % lab)
        ax.set_ylabel('density')
    fig.suptitle('Position distribution')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_temperature_evolution(times, temperatures, path=None):
    ''' Temperature (uK) vs time (ms). '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = np.asarray(times); temperatures = np.asarray(temperatures)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times * 1e3, temperatures * 1e6, 'g-')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('temperature (uK)')
    ax.set_title('Cloud temperature vs time')
    ax.grid(True, ls='--', lw=0.5)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


def temperature_series(V_over_time, mass):
    ''' Temperature at each timestep for a trajectory array of shape
        (n_steps, n_atoms, 3) as produced by MOTorNOT.integration.Solver.V. '''
    V_over_time = np.asarray(V_over_time, dtype=float)
    return np.array([temperature(V_over_time[i], mass)
                     for i in range(V_over_time.shape[0])])
