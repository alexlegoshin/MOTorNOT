''' Classical recapture of a MOT ensemble into an optical dipole trap.

    When the MOT beams and magnetic field are switched off, each atom flies
    ballistically in the conservative dipole potential U(x). An atom is
    *captured* if its total mechanical energy is below the trap escape
    threshold (U -> 0 far from the trap), i.e.

        E = 1/2 m |v|^2 + U(x) < 0.

    Since U is conservative this energy is conserved, so the energy criterion
    gives the captured fraction directly (and is fully GPU-accelerated for
    large ensembles). `simulate_recapture` additionally integrates the
    trajectories with the backend-aware fixed-step integrator so the retained
    fraction, oscillation and post-release temperature can be inspected on the
    same device.

    The quantum (tunnelling / bound-state) treatment is deliberately left out
    here: for depths of many uK and thermal atoms the classical energy
    criterion is the physically dominant effect.
'''
import numpy as np
from scipy.constants import k as kB, physical_constants
from MOTorNOT.integration import integrate
from MOTorNOT.backend import get_array_module, asnumpy, asarray, using_gpu
from MOTorNOT import diagnostics as dg
amu = physical_constants['atomic mass constant'][0]


def total_energy(trap, X, V):
    ''' Mechanical energy of each atom (J). GPU-accelerated for CuPy inputs. '''
    xp = get_array_module(X, V)
    X = xp.atleast_2d(X); V = xp.atleast_2d(V)
    KE = 0.5 * trap.mass * xp.sum(V**2, axis=1)
    return KE + trap.potential(X)


def bound_mask(trap, X, V):
    ''' Boolean mask of atoms with E < 0 (bound in the trap). '''
    return total_energy(trap, X, V) < 0


def capture_fraction(trap, X, V):
    ''' Fraction of atoms captured (energy criterion) and the mask. '''
    mask = bound_mask(trap, X, V)
    xp = get_array_module(mask)
    return float(xp.mean(mask.astype(float))), mask


def thermal_cloud(atom, N, temperature, sigma_r, center=(0, 0, 0), rng=None,
                  device='auto'):
    ''' Generate a Gaussian spatial cloud at a given temperature.

        Args:
            atom (dict): atom parameters (mass in amu).
            N (int): number of atoms.
            temperature (float): cloud temperature (K).
            sigma_r (float): RMS cloud radius per axis (m).
            device ('auto'|'cpu'|'gpu'): where to place the returned arrays;
                'auto' follows the active backend so downstream integration can
                run on the GPU.
        Returns:
            (X, V): position and velocity arrays, shape (N, 3).
    '''
    rng = rng if rng is not None else np.random.default_rng()
    m = atom['mass'] * amu
    X = rng.normal(0.0, sigma_r, size=(N, 3)) + np.asarray(center, dtype=float)
    V = rng.normal(0.0, np.sqrt(kB * temperature / m), size=(N, 3))
    if device == 'gpu' or (device == 'auto' and using_gpu()):
        return asarray(X), asarray(V)
    return X, V


def simulate_recapture(trap, X, V, duration, dt, region=None, sample=1):
    ''' Integrate the ensemble in the dipole potential and report capture.

        Runs on the GPU when X, V are CuPy arrays (see `thermal_cloud`).

        Args:
            trap: a DipoleTrap / OpticalLattice instance.
            X, V (ndarray): initial positions and velocities (N, 3).
            duration (float): hold/observation time (s).
            dt (float): fixed timestep (s).
            region (float): retention radius (m); atoms within this distance of
                the trap centre at the end are "retained". Defaults to 5*waist.
            sample (int): store every `sample`-th step of the trajectory.
        Returns:
            dict with time array, trajectory arrays X (frames,N,3) and V,
            energy-based trapped mask & fraction, retained mask & fraction,
            and temperatures of the full and trapped ensembles.
    '''
    xp = get_array_module(X, V)
    X = xp.atleast_2d(X).astype(float)
    V = xp.atleast_2d(V).astype(float)
    frac_E, mask_E = capture_fraction(trap, X, V)

    t, Xt, Vt = integrate(trap.acceleration, X, V, duration, dt, sample=sample)

    if region is None:
        region = 5 * trap.waist
    dist_final = xp.linalg.norm(Xt[-1] - xp.asarray(trap.center), axis=1)
    mask_retained = dist_final < region

    T_all = dg.temperature(V, trap.mass)
    T_trapped = dg.temperature(V[mask_E], trap.mass) if bool(mask_E.any()) else 0.0
    return {
        't': t, 'X': Xt, 'V': Vt,
        'trapped_mask': mask_E, 'trapped_fraction': frac_E,
        'retained_mask': mask_retained,
        'retained_fraction': float(xp.mean(mask_retained.astype(float))),
        'T_all': T_all, 'T_trapped': T_trapped,
    }
