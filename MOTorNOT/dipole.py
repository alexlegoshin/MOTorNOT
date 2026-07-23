''' Optical dipole traps for MOTorNOT.

    Implements the far-detuned optical dipole potential in the standard
    two-level form (Grimm, Weidemueller & Ovchinnikov, Adv. At. Mol. Opt.
    Phys. 42, 95 (2000), Eqs. 9-10):

        U(r) = -(3 pi c^2 / (2 w0^3)) * Gamma * (1/(w0 - w) + 1/(w0 + w)) * I(r)

    where w0 is the atomic resonance (angular) frequency, w is the laser
    (angular) frequency, Gamma the natural linewidth (rad/s) and I(r) the
    intensity. For a red-detuned laser (w < w0) the potential is negative
    (attractive), so atoms are drawn to high intensity. The counter-rotating
    term 1/(w0 + w) is kept so the sign and magnitude stay correct even for
    the large detunings typical of dipole traps.

    Atom data is taken from the same `atom` dict used elsewhere in MOTorNOT:
        wavelength : resonance wavelength in nm
        gamma      : natural linewidth / 2pi in Hz
        mass       : mass in amu
'''
import numpy as np
import attr
from scipy.constants import c, k as kB, physical_constants
from MOTorNOT.backend import get_array_module, asnumpy
amu = physical_constants['atomic mass constant'][0]


def ac_stark_prefactor(atom):
    ''' Returns the function I -> U(J) mapping intensity (W/m^2) to dipole
        potential energy (Joules) for the given atom and laser, as a closure
        capturing (omega0, Gamma). Laser frequency is supplied separately. '''
    omega0 = 2 * np.pi * c / (atom['wavelength'] * 1e-9)
    Gamma = 2 * np.pi * atom['gamma']
    return omega0, Gamma


@attr.s
class DipoleTrap:
    ''' Base class for a single-frequency optical dipole trap.

        Args:
            atom (dict): atom parameters (wavelength [nm], gamma [Hz], mass [amu])
            wavelength (float): laser wavelength in metres
            power (float): laser power in Watts
            waist (float): 1/e^2 beam waist radius w0 in metres
            axis (int): propagation axis (0=x, 1=y, 2=z)
            center (array-like): trap focus position in metres
    '''
    atom = attr.ib()
    wavelength = attr.ib(converter=float)
    power = attr.ib(converter=float)
    waist = attr.ib(converter=float)
    axis = attr.ib(converter=int, default=2)
    center = attr.ib(converter=lambda x: np.asarray(x, dtype=float),
                     default=np.zeros(3))

    def __attrs_post_init__(self):
        self.omega0 = 2 * np.pi * c / (self.atom['wavelength'] * 1e-9)
        self.omega = 2 * np.pi * c / self.wavelength
        self.Gamma = 2 * np.pi * self.atom['gamma']
        self.mass = self.atom['mass'] * amu
        self.zR = np.pi * self.waist**2 / self.wavelength   # Rayleigh range
        # detuning bookkeeping (rad/s); negative => red detuned
        self.detuning = self.omega - self.omega0
        # U(J) = coeff * I(W/m^2)
        self._coeff = -(3 * np.pi * c**2) / (2 * self.omega0**3) * self.Gamma \
            * (1.0 / (self.omega0 - self.omega) + 1.0 / (self.omega0 + self.omega))

    # ----- geometry helpers -----
    def _cylindrical(self, X):
        ''' Returns (longitudinal, radial) coordinates relative to the focus. '''
        xp = get_array_module(X)
        X = xp.atleast_2d(X) - xp.asarray(self.center)
        longitudinal = X[:, self.axis]
        others = [i for i in range(3) if i != self.axis]
        radial = xp.sqrt(X[:, others[0]]**2 + X[:, others[1]]**2)
        return longitudinal, radial

    def intensity(self, X):
        ''' Gaussian-beam intensity (W/m^2) at position(s) X (shape (N,3)). '''
        xp = get_array_module(X)
        z, r = self._cylindrical(X)
        w = self.waist * xp.sqrt(1 + (z / self.zR)**2)
        return (2 * self.power / (np.pi * w**2)) * xp.exp(-2 * r**2 / w**2)

    # ----- potential and forces -----
    def potential(self, X):
        ''' Dipole potential energy in Joules. '''
        return self._coeff * self.intensity(X)

    def potential_uK(self, X):
        ''' Dipole potential in units of micro-Kelvin (U / kB, x1e6). '''
        return self.potential(X) / kB * 1e6

    def depth(self):
        ''' Trap depth |U(focus)| in Joules (positive number). '''
        return float(abs(asnumpy(self.potential(self.center.reshape(1, 3)))[0]))

    def depth_uK(self):
        return self.depth() / kB * 1e6

    def force(self, X, h=1e-9):
        ''' F = -grad U, central finite difference (N,3). '''
        xp = get_array_module(X)
        X = xp.atleast_2d(X).astype(float)
        F = xp.zeros(X.shape)
        for i in range(3):
            dX = xp.zeros(X.shape)
            dX[:, i] = h
            F[:, i] = -(self.potential(X + dX) - self.potential(X - dX)) / (2 * h)
        return F

    def acceleration(self, X, V=None):
        ''' Acceleration (m/s^2); V accepted and ignored (conservative force)
            so this plugs directly into MOTorNOT.integration.solve/Solver. '''
        return self.force(X) / self.mass

    def trap_frequencies(self):
        ''' Harmonic trap (angular) frequencies (rad/s) at the focus,
            (radial, axial), from the curvature of a Gaussian beam trap. '''
        U0 = self.depth()
        omega_r = np.sqrt(4 * U0 / (self.mass * self.waist**2))
        omega_z = np.sqrt(2 * U0 / (self.mass * self.zR**2))
        return omega_r, omega_z


@attr.s
class OpticalLattice(DipoleTrap):
    ''' 1D optical lattice formed by retro-reflecting the trap beam along its
        propagation axis. The standing wave multiplies the running-wave
        intensity by 4*cos^2(k z), giving wells spaced by lambda/2. '''

    def intensity(self, X):
        xp = get_array_module(X)
        z, r = self._cylindrical(X)
        w = self.waist * xp.sqrt(1 + (z / self.zR)**2)
        envelope = (2 * self.power / (np.pi * w**2)) * xp.exp(-2 * r**2 / w**2)
        k = 2 * np.pi / self.wavelength
        return envelope * 4 * xp.cos(k * z)**2

    def trap_frequencies(self):
        ''' (radial, axial) angular frequencies. Axial confinement is set by the
            lattice curvature at an antinode: omega_z = k*sqrt(2 U0 / m). '''
        U0 = self.depth()
        k = 2 * np.pi / self.wavelength
        omega_r = np.sqrt(4 * U0 / (self.mass * self.waist**2))
        omega_z = k * np.sqrt(2 * U0 / self.mass)
        return omega_r, omega_z
