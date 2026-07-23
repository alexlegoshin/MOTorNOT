''' Internal-state (hyperfine level) rate-equation dynamics for an alkali MOT.

    A minimal but physically consistent 3-level model for the D2 cooling cycle:

        g2  : F=2 ground state  (bright; addressed by the cooling laser)
        e   : F'=3 excited state
        g1  : F=1 ground state  (dark to the cooling laser)

    Processes:
        cooling laser  g2 <-> e   at stimulated rate  W = (Gamma/2) s / (1+(2 delta/Gamma)^2)
        spontaneous    e -> g2    with branching (1-b)   (cycling)
                       e -> g1    with branching  b       (off-resonant leak to dark)
        repumper       g1 -> g2   at rate  R_rep

    With b -> 0 the model reduces to a saturated two-level system whose
    steady-state scattering rate Gamma*Ne equals the textbook
    (Gamma/2) s / (1 + s + (2 delta/Gamma)^2). Turning the repumper off lets
    the dark leak pump every atom into g1, killing the fluorescence -- the
    physical reason a repumper is required.

    Populations are fractions summing to 1: y = [Ng2, Ne, Ng1].
'''
import numpy as np
import attr
from scipy.integrate import solve_ivp


@attr.s
class LevelDynamics:
    gamma = attr.ib(converter=float)              # linewidth / 2pi, Hz
    s = attr.ib(converter=float)                  # saturation parameter I/Isat
    detuning = attr.ib(converter=float)           # laser detuning, rad/s
    branch_dark = attr.ib(converter=float, default=1e-3)   # e -> g1 branching b
    repump_rate = attr.ib(converter=float, default=0.0)    # g1 -> g2, 1/s

    def __attrs_post_init__(self):
        self.Gamma = 2 * np.pi * self.gamma       # rad/s

    @property
    def W(self):
        ''' Stimulated (absorption/emission) rate on the cooling transition. '''
        return (self.Gamma / 2) * self.s / (1 + (2 * self.detuning / self.Gamma)**2)

    def rhs(self, t, y):
        Ng2, Ne, Ng1 = y
        W, G, b, Rr = self.W, self.Gamma, self.branch_dark, self.repump_rate
        dNe = W * (Ng2 - Ne) - G * Ne
        dNg2 = -W * (Ng2 - Ne) + G * (1 - b) * Ne + Rr * Ng1
        dNg1 = G * b * Ne - Rr * Ng1
        return [dNg2, dNe, dNg1]

    def evolve(self, times, y0=(1.0, 0.0, 0.0)):
        ''' Integrate populations over `times`. Returns array (len(times), 3). '''
        times = np.asarray(times, dtype=float)
        r = solve_ivp(self.rhs, (times[0], times[-1]), list(y0),
                      t_eval=times, rtol=1e-8, atol=1e-12)
        return r.y.T

    def steady_state(self):
        ''' Steady-state populations [Ng2, Ne, Ng1] from the null space of the
            rate matrix with the normalisation Ng2+Ne+Ng1 = 1. '''
        W, G, b, Rr = self.W, self.Gamma, self.branch_dark, self.repump_rate
        # d/dt y = A y ; rows: [Ng2, Ne, Ng1]
        A = np.array([
            [-W,            W + G * (1 - b),  Rr],
            [ W,           -(W + G),          0.0],
            [ 0.0,          G * b,           -Rr],
        ])
        # replace one equation with normalisation
        M = A.copy()
        M[2, :] = 1.0
        rhs = np.array([0.0, 0.0, 1.0])
        return np.linalg.solve(M, rhs)

    def scattering_rate(self, populations):
        ''' Photon scattering rate Gamma * Ne (1/s). '''
        populations = np.atleast_2d(populations)
        return self.Gamma * populations[:, 1]

    def scattering_rate_two_level(self):
        ''' Textbook saturated two-level scattering rate, for validation. '''
        return (self.Gamma / 2) * self.s / (1 + self.s + (2 * self.detuning / self.Gamma)**2)
