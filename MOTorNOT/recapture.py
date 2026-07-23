''' Classical recapture of a MOT ensemble into an optical dipole trap.

    When the MOT beams and magnetic field are switched off, each atom flies
    ballistically in the conservative dipole potential U(x). An atom is
    *captured* if its total mechanical energy is below the trap escape
    threshold (U -> 0 far from the trap), i.e.

        E = 1/2 m |v|^2 + U(x) < 0.

    Since U is conservative this energy is conserved, so the energy criterion
    gives the captured fraction directly. `simulate_recapture` additionally
    integrates the trajectories (reusing MOTorNOT.integration.solve) so the
    retained fraction, oscillation and post-release temperature can be
    inspected. The quantum (tunnelling / bound-state) treatment is deliberately
    left out here -- for depths of many uK and thermal atoms the classical
    energy criterion is the physically dominant effect.
'''
import numpy as np
from scipy.constants import k as kB, physical_constants
from MOTorNOT.integration import solve
from MOTorNOT import diagnostics as dg
amu = physical_constants['atomic mass constant'][0]


def total_energy(trap, X, V):
    ''' Mechanical energy of each atom (J). '''
    X = np.atleast_2d(X); V = np.atleast_2d(V)
    KE = 0.5 * trap.mass * np.sum(V**2, axis=1)
    return KE + trap.potential(X)


def bound_mask(trap, X, V):
    ''' Boolean mask of atoms with E < 0 (bound in the trap). '''
    return total_energy(trap, X, V) < 0


def capture_fraction(trap, X, V):
    ''' Fraction of atoms captured (energy criterion) and the mask. '''
    mask = bound_mask(trap, X, V)
    return float(np.mean(mask)), mask


def thermal_cloud(atom, N, temperature, sigma_r, center=(0, 0, 0), rng=None):
    ''' Generate a Gaussian spatial cloud at a given temperature.

        Args:
            atom (dict): atom parameters (mass in amu).
            N (int): number of atoms.
            temperature (float): cloud temperature (K).
            sigma_r (float): RMS cloud radius per axis (m).
        Returns:
            (X, V): position and velocity arrays, shape (N, 3).
    '''
    rng = rng if rng is not None else np.random.default_rng()
    m = atom['mass'] * amu
    X = rng.normal(0.0, sigma_r, size=(N, 3)) + np.asarray(center, dtype=float)
    V = rng.normal(0.0, np.sqrt(kB * temperature / m), size=(N, 3))
    return X, V


def simulate_recapture(trap, X, V, duration, dt=None, region=None):
    ''' Integrate the ensemble in the dipole potential and report capture.

        Args:
            trap: a DipoleTrap / OpticalLattice instance.
            X, V (ndarray): initial positions and velocities (N, 3).
            duration (float): hold/observation time (s).
            dt (float): optional fixed timestep for the output grid.
            region (float): retention radius (m); atoms within this distance of
                the trap centre at the end are "retained". Defaults to
                5 * waist.
        Returns:
            dict with time array, trajectory arrays X (steps,N,3) and V,
            energy-based trapped mask & fraction, retained mask & fraction,
            and temperatures of the full and trapped ensembles.
    '''
    X = np.atleast_2d(X).astype(float)
    V = np.atleast_2d(V).astype(float)
    frac_E, mask_E = capture_fraction(trap, X, V)

    y, Xt, Vt, t, events = solve(trap.acceleration, X, V, duration, dt=dt)

    if region is None:
        region = 5 * trap.waist
    dist_final = np.linalg.norm(Xt[-1] - trap.center, axis=1)
    mask_retained = dist_final < region

    T_all = dg.temperature(V, trap.mass)
    T_trapped = dg.temperature(V[mask_E], trap.mass) if mask_E.any() else 0.0
    return {
        't': t, 'X': Xt, 'V': Vt,
        'trapped_mask': mask_E, 'trapped_fraction': frac_E,
        'retained_mask': mask_retained, 'retained_fraction': float(np.mean(mask_retained)),
        'T_all': T_all, 'T_trapped': T_trapped,
    }
