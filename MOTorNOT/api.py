''' High-level, batteries-included API for MOTorNOT.

    The lower-level classes (SixBeam, GratingMOT, DipoleTrap, Solver, ...) stay
    available for full control, but most workflows -- cool a cloud in a MOT,
    release it into a dipole trap, read out the temperature -- are covered by
    the helpers here:

        import MOTorNOT as mn

        atom = mn.rubidium87()
        mot  = mn.six_beam_mot(atom, power=15e-3, detuning=-0.5)   # detuning in linewidths
        trap = mn.dipole_trap(atom, wavelength=1064e-9, power=2.0, waist=25e-6)

        exp = mn.Experiment(atom, mot=mot, trap=trap)
        exp.load_cloud(N=5000, temperature=300e-6, sigma_r=0.3e-3)
        exp.cool(duration=5e-3, dt=1e-6)          # Doppler cooling in the MOT
        result = exp.recapture(duration=2e-3, dt=1e-6)
        print(exp.temperature(), result['trapped_fraction'])

    Everything runs on the GPU automatically when CuPy + a CUDA device are
    present; call `mn.set_backend('cpu')` to force the CPU.
'''
import numpy as np
from scipy.constants import k as kB, physical_constants
from MOTorNOT.mot import SixBeam
from MOTorNOT.coils import LinearQuadrupole
from MOTorNOT.dipole import DipoleTrap, OpticalLattice
from MOTorNOT.integration import integrate
from MOTorNOT import diagnostics, recapture, backend
amu = physical_constants['atomic mass constant'][0]


# ---------------------------------------------------------------- atoms -----
# Common laser-cooled alkalis. gamma is the natural linewidth / 2pi in Hz,
# wavelength the cooling-transition wavelength in nm, Isat in mW/cm^2.
ATOMS = {
    'Rb87': {'mass': 87, 'gamma': 6.0666e+6, 'wavelength': 780.241, 'gF': 0.5, 'Isat': 1.67},
    'Rb85': {'mass': 85, 'gamma': 6.0666e+6, 'wavelength': 780.241, 'gF': 1.0 / 3, 'Isat': 1.67},
    'Na23': {'mass': 23, 'gamma': 9.79e+6, 'wavelength': 589.158, 'gF': 0.5, 'Isat': 6.26},
    'K39':  {'mass': 39, 'gamma': 6.035e+6, 'wavelength': 766.701, 'gF': 0.5, 'Isat': 1.75},
    'Cs133': {'mass': 133, 'gamma': 5.234e+6, 'wavelength': 852.347, 'gF': 0.25, 'Isat': 1.10},
}


def atom(name='Rb87'):
    ''' Return a fresh copy of a built-in atom parameter dict. '''
    if name not in ATOMS:
        raise KeyError('unknown atom %r; available: %s' % (name, list(ATOMS)))
    return dict(ATOMS[name])


def rubidium87():
    ''' Convenience shortcut for the Rb-87 D2 atom dict. '''
    return atom('Rb87')


# --------------------------------------------------------------- builders ---
def six_beam_mot(atom, power=15e-3, radius=10e-3, detuning=-0.5,
                 B_gradient=0.1, handedness=-1):
    ''' Build a ready-to-use six-beam MOT with a (GPU-friendly) linear
        quadrupole field.

        Args:
            atom (dict): atom parameters (see `atom`).
            power (float): power per beam (W).
            radius (float): beam radius (m).
            detuning (float): laser detuning in units of the natural linewidth
                (negative = red). E.g. -0.5 means -Gamma/2.
            B_gradient (float): magnetic field gradient B0 (T/m).
            handedness (int): polarization handedness; the default (-1) together
                with B_gradient>0 gives a 3D-restoring trap.
        Returns:
            a configured SixBeam MOT with `.atom` set.
    '''
    linewidth = 2 * np.pi * atom['gamma']
    field = LinearQuadrupole(B0=B_gradient, offset=0.0).field
    mot = SixBeam(power=power, radius=radius, detuning=detuning * linewidth,
                  handedness=handedness, field=field)
    mot.atom = atom
    return mot


def dipole_trap(atom, wavelength=1064e-9, power=1.0, waist=25e-6, axis=2,
                lattice=False):
    ''' Build a Gaussian dipole trap (or a 1D optical lattice if lattice=True). '''
    cls = OpticalLattice if lattice else DipoleTrap
    return cls(atom=atom, wavelength=wavelength, power=power, waist=waist, axis=axis)


def thermal_cloud(atom, N, temperature, sigma_r, center=(0, 0, 0), seed=None):
    ''' Generate a Gaussian cloud of N atoms at a given temperature. Arrays are
        placed on the active backend (GPU when enabled). '''
    rng = np.random.default_rng(seed)
    return recapture.thermal_cloud(atom, N, temperature, sigma_r,
                                   center=center, rng=rng)


def set_backend(name='auto'):
    ''' Select the compute backend: 'gpu', 'cpu', or 'auto'. Returns the active
        array module. '''
    name = name.lower()
    if name == 'gpu':
        return backend.use_gpu()
    if name == 'cpu':
        return backend.use_cpu()
    if name == 'auto':
        return backend.use_gpu() if backend.GPU_AVAILABLE else backend.use_cpu()
    raise ValueError("backend must be 'gpu', 'cpu' or 'auto'")


# ------------------------------------------------------------- experiment ---
class Experiment:
    ''' A small orchestrator tying a MOT and a dipole trap together around a
        shared atomic ensemble. Positions/velocities are kept on the active
        backend so the whole sequence can run on the GPU.

        Note: the semiclassical force model has no photon-recoil heating, so MOT
        cooling drives the temperature monotonically down rather than settling
        at the Doppler limit -- treat `cool` as a capture/compression stage.
    '''

    def __init__(self, atom, mot=None, trap=None):
        self.atom = atom
        self.mot = mot
        self.trap = trap
        self.mass = atom['mass'] * amu
        self.X = None
        self.V = None

    def load_cloud(self, N, temperature, sigma_r, center=(0, 0, 0), seed=None):
        ''' Create an initial thermal cloud. '''
        self.X, self.V = thermal_cloud(self.atom, N, temperature, sigma_r,
                                       center=center, seed=seed)
        return self

    def set_cloud(self, X, V):
        ''' Use an externally supplied cloud. '''
        self.X, self.V = X, V
        return self

    def temperature(self):
        ''' Current cloud temperature (K). '''
        return diagnostics.temperature(self.V, self.mass)

    def rms_size(self):
        ''' Current cloud RMS radius (m) and per-axis sizes. '''
        return diagnostics.rms_radius(self.X)

    def cool(self, duration, dt, sample=10):
        ''' Evolve the cloud in the MOT for `duration`. Updates the stored
            positions/velocities to the final state and returns
            (times, temperatures) sampled along the way. '''
        if self.mot is None:
            raise ValueError('no MOT configured')
        t, X, V = integrate(self.mot.acceleration, self.X, self.V,
                            duration, dt, sample=sample)
        self.X, self.V = X[-1], V[-1]
        return t, diagnostics.temperature_series(V, self.mass)

    def recapture(self, duration, dt, **kwargs):
        ''' Release the current cloud into the dipole trap and report capture. '''
        if self.trap is None:
            raise ValueError('no dipole trap configured')
        return recapture.simulate_recapture(self.trap, self.X, self.V,
                                             duration, dt, **kwargs)
