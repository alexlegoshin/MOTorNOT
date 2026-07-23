# MOTorNOT

A compact simulation library for **laser cooling and trapping of neutral atoms**.
It models the semiclassical scattering force in a magneto-optical trap (MOT),
optical **dipole traps** and lattices, hyperfine **level dynamics with a
repumper**, and the **recapture** of a MOT cloud into a dipole trap — with the
same code running on the **CPU or an NVIDIA GPU** automatically.

```python
import MOTorNOT as mn

atom = mn.rubidium87()
mot  = mn.six_beam_mot(atom, power=15e-3, detuning=-1.0)      # detuning in linewidths
trap = mn.dipole_trap(atom, wavelength=1064e-9, power=3.0, waist=25e-6)

exp = mn.Experiment(atom, mot=mot, trap=trap)
exp.load_cloud(N=5000, temperature=300e-6, sigma_r=0.3e-3)    # 300 µK cloud
exp.cool(duration=4e-3, dt=2e-6)                              # Doppler cooling in the MOT
result = exp.recapture(duration=1e-3, dt=2e-6)                # release into the dipole trap

print("T = %.1f µK, recaptured %.0f%%"
      % (exp.temperature()*1e6, 100*result['trapped_fraction']))
```

---

## Features

- **Six-beam and grating MOTs** with a proper σ± / m_F scattering-force model,
  Zeeman shifts, and real coil or linear-quadrupole magnetic fields.
- **Optical dipole traps** (Gaussian beam) and **1D optical lattices** with the
  correct AC-Stark potential (red-detuned ⇒ attractive), trap depth and
  radial/axial trap frequencies.
- **Hyperfine level dynamics** (F=2 / F′ / F=1) as rate equations, including a
  **repumper** — turn it off and the fluorescence dies, as it should.
- **Ensemble diagnostics**: temperature (total & per-axis), cloud size,
  velocity/position distributions, and ready-made plots.
- **MOT → dipole recapture** with an exact energy-based capture criterion plus
  full trajectory integration.
- **Transparent GPU acceleration** via [CuPy](https://cupy.dev): if a CUDA GPU
  is present the heavy array math runs on it; otherwise everything falls back to
  NumPy. No code changes, no duplicated logic.

## Installation

```bash
# CPU (NumPy) — always works
pip install -e .

# Optional GPU acceleration (NVIDIA CUDA 12.x). The [ctk] extra pulls in the
# CUDA headers CuPy needs to compile kernels:
pip install "cupy-cuda12x[ctk]"
```

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `plotly`, `attrs`,
`pyyaml`. Python ≥ 3.9 (developed and tested on 3.12).

## Quick start

The high-level `Experiment` orchestrator is the fastest way in, but every piece
is usable on its own.

### Atoms

```python
atom = mn.rubidium87()          # or mn.atom('Rb85'), 'Na23', 'K39', 'Cs133'
```

An atom is a plain dict: `mass` (amu), `gamma` (linewidth/2π, Hz),
`wavelength` (nm), `gF`, `Isat` (mW/cm²). Build your own to model any species.

### Magneto-optical trap

```python
mot = mn.six_beam_mot(atom, power=15e-3, radius=10e-3,
                      detuning=-1.0,      # in units of the natural linewidth
                      B_gradient=0.1)     # T/m
a = mot.acceleration(X, V)                # X, V are (N, 3) arrays, in SI units
```

### Dipole trap and lattice

```python
trap    = mn.dipole_trap(atom, wavelength=1064e-9, power=2.0, waist=25e-6)
lattice = mn.dipole_trap(atom, wavelength=1064e-9, power=2.0, waist=25e-6,
                         lattice=True)

trap.depth_uK()            # trap depth in µK  (negative potential ⇒ positive depth)
trap.trap_frequencies()   # (ω_radial, ω_axial) in rad/s
U = trap.potential(X)     # potential energy (J); trap.force(X) = -∇U
```

### Level dynamics with a repumper

```python
from MOTorNOT import LevelDynamics
ld = LevelDynamics(gamma=atom['gamma'], s=2.0,
                   detuning=-0.5*2*3.14159*atom['gamma'],
                   branch_dark=1e-3, repump_rate=1e6)
ld.steady_state()          # [F=2, excited, F=1] populations
ld.scattering_rate(ld.steady_state())
```

### Diagnostics

```python
from MOTorNOT import diagnostics as dg
dg.temperature(V, mass)                       # K
dg.plot_velocity_distribution(V, mass, path='v.png')
dg.plot_temperature_evolution(times, temps, path='T.png')
```

### Recapture

```python
from MOTorNOT import recapture as rc
frac, mask = rc.capture_fraction(trap, X, V)                    # exact, E<0 criterion
res = rc.simulate_recapture(trap, X, V, duration=2e-3, dt=1e-6) # + trajectories
```

### Low-level integration

```python
from MOTorNOT import integrate, Solver
# backend-aware fixed-step RK4 (GPU-capable, handles velocity-dependent forces):
t, X, V = integrate(mot.acceleration, X0, V0, duration=3e-3, dt=1e-6)
# or scipy's adaptive solver on the CPU:
sol = Solver(mot.acceleration, X0, V0).run(3e-3, dt=1e-6)
```

## GPU acceleration

The numeric kernels are *array-agnostic*: they dispatch to CuPy or NumPy based
on where their input arrays live. Practically:

```python
mn.set_backend('auto')     # 'gpu', 'cpu', or 'auto' (default)
mn.GPU_AVAILABLE           # True if a usable CUDA GPU + CuPy were found
mn.backend.device_name()   # e.g. 'NVIDIA GeForce GTX 1650'
```

When the GPU backend is active, `thermal_cloud`/`Experiment` place arrays on the
device and the force evaluation, `integrate`, dipole potentials, diagnostics and
the recapture energy criterion all run there. CPU and GPU results are identical
to floating-point precision. The benefit grows with ensemble size; the GPU pays
a kernel-launch overhead that dominates for small `N`.

Runs on the CPU regardless (no GPU code path is required): the adaptive
`solve_ivp` integrator, the elliptic-integral `Coil`/`QuadrupoleCoils` fields,
and root finding. Use `LinearQuadrupole` for a fully GPU-resident MOT.

## Physics notes & limitations

- Forces are **semiclassical** (mean scattering rate); there is no photon-recoil
  momentum diffusion, so MOT cooling drives the temperature monotonically down
  rather than settling at the Doppler limit. Treat `cool` as a capture/
  compression stage, not a thermometer of the Doppler floor.
- The dipole potential uses the standard two-level AC-Stark form (Grimm,
  Weidemüller & Ovchinnikov 2000) including the counter-rotating term.
- Recapture is treated **classically** (energy criterion + ballistic motion in
  the conservative potential); tunnelling / bound-state effects are negligible
  for the many-µK depths and thermal atoms considered here.

## Module map

| Module | Contents |
|---|---|
| `mot.py` | `MOT`, `SixBeam`, `GratingMOT`, `PyramidMOT` scattering-force models |
| `beams.py` | `UniformBeam`, `GaussianBeam`, `DiffractedBeam`, `PyramidBeam` |
| `coils.py` | `Coil`, `QuadrupoleCoils`, `LinearQuadrupole` magnetic fields |
| `dipole.py` | `DipoleTrap`, `OpticalLattice` |
| `levels.py` | `LevelDynamics` hyperfine rate equations + repumper |
| `diagnostics.py` | temperature, cloud size, distributions, plots |
| `recapture.py` | MOT→dipole recapture, `thermal_cloud`, capture fraction |
| `integration.py` | `Solver` (scipy), `integrate` (backend-aware RK4) |
| `backend.py` | CPU/GPU array-module selection |
| `api.py` | high-level helpers: `atom`, `six_beam_mot`, `dipole_trap`, `Experiment` |

## License

MIT. Original MOT/beam/coil framework by Robert Fasano; dipole-trap, level,
diagnostics, recapture and GPU layers added subsequently.
