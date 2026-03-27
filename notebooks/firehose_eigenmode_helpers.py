"""Helpers for 1-D CGL firehose-eigenmode experiments with Dedalus.

Provides five groups of functionality:

1. **CGL closure** -- analytical Chew-Goldberger-Low pressure law in both
   NumPy (``cgl_pressures_np``) and torch (``cgl_pressures_torch``) variants.
   All other CGL-aware functions delegate to these two.
2. **Linear analysis** -- flux Jacobian via torch autograd, eigenvalue
   decomposition for a single Fourier mode (``build_linear_operator``).
3. **Nonlinear runner** -- ``run_firehose_case`` builds a 1-D Dedalus IVP
   seeded by the fastest-growing firehose eigenmode, with built-in spectral
   dealiasing filter (on by default) and configurable stopping guards
   (Egrow ceiling, high-k spectral quality, density floor).
4. **ML surrogate** -- MLP model, synthetic data generators, training loop
   with optional physics-informed losses (tangent, flux-Jacobian,
   growth-rate matching).
5. **Diagnostics** -- lightweight spectral helpers for post-run analysis.

Designed for parameter campaigns over (rho0, Bx0) with the firehose
instability as the target physics.

B_floor convention
------------------
The linear-analysis / Dedalus routines default to ``B_floor=1e-6`` (tight
floor, negligible regularisation for well-resolved fields).  The ML loss
and inference routines default to ``B_floor=1e-3`` (softer floor for
numerically stable autograd gradients during training).  Callers can
override both.
"""

from __future__ import annotations

import time

import dedalus.public as d3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ── CGL closure ──────────────────────────────────────────────────────────


def cgl_pressures_np(rho, Bx, By, Bz, gamma_par=3.0, gamma_perp=2.0, B_floor=1e-6):
    """CGL parallel / perpendicular pressures (NumPy).
    returns p_par, p_perp, B2, Bmag
    """
    rho = np.asarray(rho, dtype=np.float64)
    Bx = np.asarray(Bx, dtype=np.float64)
    By = np.asarray(By, dtype=np.float64)
    Bz = np.asarray(Bz, dtype=np.float64)
    rho_safe = np.maximum(rho, 1e-12)
    B2 = Bx**2 + By**2 + Bz**2
    Bmag = np.sqrt(B2 + B_floor**2)
    p_par = rho_safe**gamma_par / Bmag**(gamma_par - 1.0)
    p_perp = rho_safe * Bmag**(gamma_perp - 1.0)
    return p_par, p_perp, B2, Bmag


def cgl_total_energy_density_np(rho, mx, my, mz, Bx, By, Bz,
                                gamma_par=3.0, gamma_perp=2.0, B_floor=1e-6):
    r"""CGL total energy density: kinetic + magnetic + internal ($p_\parallel/2 + p_\perp$)."""
    rho_safe = np.maximum(np.asarray(rho, dtype=np.float64), 1e-12)
    mx, my, mz = (np.asarray(v, dtype=np.float64) for v in (mx, my, mz))
    Bx, By, Bz = (np.asarray(v, dtype=np.float64) for v in (Bx, By, Bz))
    p_par, p_perp, B2, _ = cgl_pressures_np(rho_safe, Bx, By, Bz,
                                             gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor)
    return 0.5 * (mx**2 + my**2 + mz**2) / rho_safe + 0.5 * B2 + 0.5 * p_par + p_perp


def cgl_pressures_torch(rho, Bx, By, Bz, gamma_par=3.0, gamma_perp=2.0, B_floor=1e-6):
    """CGL parallel / perpendicular pressures (torch, differentiable).

    Mirrors :func:`cgl_pressures_np` but operates on torch tensors with
    full autograd support.  This is the **single source of truth** for
    the CGL pressure law on the torch side; other functions
    (:func:`_flux_torch`, :func:`pressures_nn_aniso`) delegate here.

    Parameters
    ----------
    rho, Bx, By, Bz : Tensor
        Scalar or broadcastable tensors of density and magnetic-field
        components.
    gamma_par, gamma_perp : float
        CGL polytropic indices (default 3.0 and 2.0).
    B_floor : float
        Regularisation floor added inside ``sqrt(B**2 + B_floor**2)``.

    Returns
    -------
    (p_par, p_perp, B2, Bmag) : tuple of Tensors
    """
    rho_safe = torch.clamp(rho, min=1e-10)
    B2 = Bx**2 + By**2 + Bz**2
    Bmag = torch.sqrt(B2 + B_floor**2)
    p_par = rho_safe**gamma_par / Bmag**(gamma_par - 1.0)
    p_perp = rho_safe * Bmag**(gamma_perp - 1.0)
    return p_par, p_perp, B2, Bmag


# ── Linear operator ─────────────────────────────────────────────────────


def _flux_torch(Q, gamma_par=3.0, gamma_perp=2.0, B_floor=1e-6):
    """1-D CGL conservative flux vector ``F(Q)`` (torch, differentiable).

    Delegates the pressure law to :func:`cgl_pressures_torch`, then
    assembles the 7-component flux for
    ``Q = (rho, mx, my, mz, Bx, By, Bz)``.

    Used by :func:`build_linear_operator` to obtain the linearised
    flux Jacobian via ``torch.autograd.functional.jacobian``.
    """
    rho_safe = torch.clamp(Q[0], min=1e-10)
    mx, my, mz = Q[1], Q[2], Q[3]
    Bx, By, Bz = Q[4], Q[5], Q[6]
    ux, uy, uz = mx / rho_safe, my / rho_safe, mz / rho_safe

    p_par, p_perp, B2, Bmag = cgl_pressures_torch(
        rho_safe, Bx, By, Bz, gamma_par, gamma_perp, B_floor)

    # Anisotropic pressure tensor: P_ij = p_perp delta_ij + (p_par - p_perp) b_i b_j
    dp = p_par - p_perp
    bx, by, bz = Bx / Bmag, By / Bmag, Bz / Bmag
    Pxx = p_perp + dp * bx * bx
    Pxy = dp * bx * by
    Pxz = dp * bx * bz

    return torch.stack([
        mx,                                           # mass flux
        mx * ux + Pxx + 0.5 * B2 - Bx * Bx,          # x-momentum flux
        my * ux + Pxy - Bx * By,                      # y-momentum flux
        mz * ux + Pxz - Bx * Bz,                      # z-momentum flux
        0.0 * Bx,                                      # dBx/dt = 0 (div B = 0)
        ux * By - uy * Bx,                             # By induction
        ux * Bz - uz * Bx,                             # Bz induction
    ])


def build_linear_operator(rho0, Bx0, k_phys, mu_visc, eta,
                           gamma_par=3.0, gamma_perp=2.0, B_floor=1e-6):
    """Linearised CGL operator for a single Fourier mode.

    Computes the flux Jacobian ``A = dF/dQ`` at the equilibrium state
    ``Q0 = (rho0, 0, 0, 0, Bx0, 0, 0)`` via torch autograd, then forms
    the linear operator ``L = -i k A - k² D`` where ``D`` is the
    diffusion matrix (viscosity + resistivity).

    Parameters
    ----------
    rho0, Bx0 : float
        Equilibrium density and x-magnetic field.
    k_phys : float
        Physical wavenumber.
    mu_visc, eta : float
        Viscosity and resistivity.

    Returns
    -------
    dict
        Keys: ``A``, ``L``, ``eigvals``, ``eigvecs``,
        ``transverse_fraction``, ``best_idx``, ``best_sigma``, ``best_vec``.
        The *best* mode has the largest growth rate, with transverse
        polarisation fraction as a tie-breaker to prefer firehose-like
        modes over sound-like ones.
    """
    Qeq = torch.tensor([rho0, 0.0, 0.0, 0.0, Bx0, 0.0, 0.0], requires_grad=True)

    def flux_of_Q(Q):
        return _flux_torch(Q, gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor)

    A = torch.autograd.functional.jacobian(flux_of_Q, Qeq).detach().cpu().numpy()
    diffusive = np.diag([0.0, (4.0 / 3.0) * mu_visc / rho0, mu_visc / rho0,
                         mu_visc / rho0, eta, eta, eta])
    L = (-1j * k_phys) * A - (k_phys**2) * diffusive
    eigvals, eigvecs = np.linalg.eig(L)
    transverse_fraction = (np.linalg.norm(eigvecs[[2, 3, 5, 6], :], axis=0)
                           / np.linalg.norm(eigvecs, axis=0))
    best_idx = int(np.argmax(eigvals.real + 1e-12 * transverse_fraction))
    return {
        "A": A, "L": L, # linearized flux Jacobian and operator
        "eigvals": eigvals, "eigvecs": eigvecs, # eigenvalues and eigenvectors
        "transverse_fraction": transverse_fraction, # for tie-breaking
        "best_idx": best_idx, # index of best firehose-like mode
        "best_sigma": eigvals[best_idx], # growth rate + i * frequency of best mode
        "best_vec": eigvecs[:, best_idx], # eigenvector of best mode
    }


# ── Initial-condition construction ───────────────────────────────────────


def make_real_eigenmode_profile(eigvec, x, k_phys, amplitude, phase_index=5):
    """Convert a complex eigenvector to a real-valued spatial perturbation.

    Rotates the phase so that component *phase_index* is as real as possible.
    """
    eigvec = np.asarray(eigvec, dtype=np.complex128)
    anchor = eigvec[phase_index]
    if np.abs(anchor) < 1e-14:
        anchor = eigvec[np.argmax(np.abs(eigvec))]
    phased_vec = eigvec * np.exp(-1j * np.angle(anchor))
    phased_vec = phased_vec / np.max(np.abs(phased_vec))
    mode = np.exp(1j * k_phys * x)
    return amplitude * np.real(phased_vec[:, None] * mode[None, :])


def build_seed_state_arrays(case, gamma_par=3.0, gamma_perp=2.0, B_floor=1e-6):
    """Build initial-condition arrays for a firehose case (no Dedalus solve).

    Accepts the same *case* dict as ``run_firehose_case``.
    Returns dict with x, field arrays, and firehose margin.
    """
    N = int(case.get("N", 512))
    Lx = float(case.get("Lx", 12.0))
    rho0, Bx0 = float(case["rho0"]), float(case["Bx0"])
    mode_number = int(case["mode_number"])
    amplitude = float(case["amplitude"])

    x = np.linspace(-Lx / 2.0, Lx / 2.0, N, endpoint=False)
    k_phys = 2.0 * np.pi * mode_number / Lx
    lin = build_linear_operator(rho0, Bx0, k_phys, float(case["mu_visc"]), float(case["eta"]),
                                gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor)
    pert = make_real_eigenmode_profile(lin["best_vec"], x, k_phys, amplitude)

    rho_arr = rho0 + pert[0]
    mx_arr, my_arr, mz_arr = pert[1], pert[2], pert[3]
    Bx_arr = Bx0 + pert[4]
    By_arr, Bz_arr = pert[5], pert[6]

    p_par, p_perp, B2, _ = cgl_pressures_np(rho_arr, Bx_arr, By_arr, Bz_arr,
                                             gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor)
    return {
        "x": x,
        "rho": rho_arr, "mx": mx_arr, "my": my_arr, "mz": mz_arr,
        "Bx": Bx_arr, "By": By_arr, "Bz": Bz_arr,
        "margin": p_par - p_perp - B2,
    }


# ── Spectral diagnostics ────────────────────────────────────────────────


def spectral_resolution_metrics(field_x, top_mode_start=0.8):
    """(top_power_fraction, peak_mode_fraction, peak_mode_index) near Nyquist."""
    centered = np.asarray(field_x, dtype=np.float64) - np.mean(field_x)
    coeff = np.fft.rfft(centered)
    if coeff.size <= 1:
        return 0.0, 0.0, 0
    power = np.abs(coeff) ** 2
    total_power = float(np.sum(power[1:]))
    nyquist_mode = coeff.size - 1
    if total_power <= 1e-30 or nyquist_mode <= 0:
        return 0.0, 0.0, 0
    start_mode = min(nyquist_mode, max(1, int(np.floor(top_mode_start * nyquist_mode))))
    top_power_fraction = float(np.sum(power[start_mode:]) / total_power)
    peak_mode = int(np.argmax(power[1:]) + 1)
    peak_mode_fraction = float(peak_mode / nyquist_mode)
    return top_power_fraction, peak_mode_fraction, peak_mode


def high_k_fraction(field_x, frac=0.33):
    """Fraction of spectral power in the top *frac* of modes."""
    coeff = np.fft.rfft(np.asarray(field_x) - np.mean(field_x))
    power = np.abs(coeff) ** 2
    cutoff = max(1, int(np.floor((1.0 - frac) * coeff.size)))
    return float(power[cutoff:].sum() / max(power.sum(), 1e-30))


def spectral_mode_summary(field_x, Lx_local, top_n=8):
    """DataFrame of the *top_n* most energetic Fourier modes."""
    coeff = np.fft.rfft(np.asarray(field_x) - np.mean(field_x))
    power = np.abs(coeff) ** 2
    mode_numbers = np.arange(coeff.size)
    order = np.argsort(power[1:])[::-1] + 1 if coeff.size > 1 else np.array([], dtype=int)
    nyquist = max(coeff.size - 1, 1)
    rows = [
        {"mode": int(mode_numbers[i]),
         "mode_fraction_of_nyquist": float(mode_numbers[i] / nyquist),
         "wavelength": float(Lx_local / mode_numbers[i]),
         "power_fraction": float(power[i] / max(power.sum(), 1e-30))}
        for i in order[:top_n]
    ]
    return pd.DataFrame(rows)


def nearest_snapshot(result, target_t):
    """Snapshot from *result* closest to *target_t*."""
    snaps = result["snapshots"]
    return snaps[int(np.argmin([abs(s["t"] - target_t) for s in snaps]))]


# ── Nonlinear Dedalus runner ─────────────────────────────────────────────

FIELD_NAMES = ("rho", "mx", "my", "mz", "Bx", "By", "Bz")


def run_firehose_case(
    rho_or_fields, mx_init=None, my_init=None, mz_init=None,
    Bx_init=None, By_init=None, Bz_init=None,
    *,
    Lx,
    mu_visc,
    eta,
    gamma_par=3.0,
    gamma_perp=2.0,
    B_floor=1e-6,
    t_final=0.5,
    cfl_safety=0.05,
    dt_initial=5e-6,
    dt_max=1e-4,
    sample_every=10,
    print_every=50,
    max_steps=20000,
    wall_time_limit=120.0,
    rho_floor=1e-10,
    max_state_norm=15.0,
    max_energy_growth_factor=None,
    max_high_k_power=0.10,
    filter_fraction=0.67,
    filter_cadence=5,
):
    """Run a 1-D CGL firehose experiment from prescribed initial fields.

    Parameters
    ----------
    rho_or_fields : array_like or dict
        Either the density array, or a dict with keys from ``FIELD_NAMES``
        (``"rho"``, ``"mx"``, ``"my"``, ``"mz"``, ``"Bx"``, ``"By"``,
        ``"Bz"``).  When a dict is passed the remaining six positional
        arguments are ignored.
    mx_init, my_init, mz_init, Bx_init, By_init, Bz_init : array_like, optional
        Individual field arrays (used only when *rho_or_fields* is not a dict).
    Lx : float
        Domain length.
    mu_visc, eta : float
        Viscosity and resistivity.

    Returns
    -------
    dict
        ``"t"``  -- (n_t,) snapshot times.
        ``"x"``  -- (n_x,) grid positions.
        ``"rho"``, ``"mx"``, ``"my"``, ``"mz"``, ``"Bx"``, ``"By"``, ``"Bz"``
            -- each (n_t, n_x) field history.
        ``"status"`` -- str, reason the simulation ended.
    """
    # ── accept dict or individual arrays ─────────────────────────────────
    if isinstance(rho_or_fields, dict):
        init_arrays = [
            np.asarray(rho_or_fields[name], dtype=np.float64).ravel()
            for name in FIELD_NAMES
        ]
    else:
        if any(a is None for a in (mx_init, my_init, mz_init,
                                    Bx_init, By_init, Bz_init)):
            raise TypeError(
                "When rho_or_fields is not a dict, all seven field arrays "
                "must be provided as positional arguments."
            )
        init_arrays = [
            np.asarray(a, dtype=np.float64).ravel()
            for a in (rho_or_fields, mx_init, my_init, mz_init,
                      Bx_init, By_init, Bz_init)
        ]

    N = init_arrays[0].size
    if not all(a.size == N for a in init_arrays):
        raise ValueError("All initial-condition arrays must have the same length.")

    Lx = float(Lx)
    mu_visc = float(mu_visc)
    eta = float(eta)

    filter_on = filter_cadence > 0 and 0.0 < filter_fraction < 1.0

    # ── Dedalus domain ───────────────────────────────────────────────────
    xcoord = d3.Coordinate("x")
    dist   = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.RealFourier(xcoord, size=N, bounds=(-Lx / 2.0, Lx / 2.0), dealias=3 / 2)
    x  = dist.local_grid(xbasis, scale=1)
    dx = lambda A: d3.Differentiate(A, xcoord)

    rho = dist.Field(name="rho", bases=xbasis)
    mx  = dist.Field(name="mx",  bases=xbasis)
    my  = dist.Field(name="my",  bases=xbasis)
    mz  = dist.Field(name="mz",  bases=xbasis)
    Bx  = dist.Field(name="Bx",  bases=xbasis)
    By  = dist.Field(name="By",  bases=xbasis)
    Bz  = dist.Field(name="Bz",  bases=xbasis)
    state_fields = [rho, mx, my, mz, Bx, By, Bz]

    # ── CGL closure (Dedalus symbolic fields) ──────────────────────────
    # NOTE: This duplicates the CGL pressure formula from cgl_pressures_np /
    # cgl_pressures_torch, but uses Dedalus symbolic Field objects which
    # cannot share code with the NumPy / torch implementations.
    rho_safe = rho + rho_floor
    ux, uy, uz = mx / rho_safe, my / rho_safe, mz / rho_safe
    B2   = Bx * Bx + By * By + Bz * Bz
    Bmag = np.sqrt(B2 + B_floor**2)
    p_par  = rho_safe**gamma_par / Bmag**(gamma_par - 1.0)
    p_perp = rho_safe * Bmag**(gamma_perp - 1.0)
    dp = p_par - p_perp
    bx, by, bz = Bx / Bmag, By / Bmag, Bz / Bmag
    Pxx, Pxy, Pxz = p_perp + dp * bx * bx, dp * bx * by, dp * bx * bz

    Frho = mx
    Fmx  = mx * ux + Pxx + 0.5 * B2 - Bx * Bx
    Fmy  = my * ux + Pxy - Bx * By
    Fmz  = mz * ux + Pxz - Bx * Bz
    FBy  = ux * By - uy * Bx
    FBz  = ux * Bz - uz * Bx
    tau_xx = (4.0 / 3.0) * mu_visc * dx(ux)
    tau_xy = mu_visc * dx(uy)
    tau_xz = mu_visc * dx(uz)

    problem = d3.IVP(state_fields, namespace=locals())
    problem.add_equation("dt(rho) = -dx(Frho)")
    problem.add_equation("dt(mx)  = -dx(Fmx - tau_xx)")
    problem.add_equation("dt(my)  = -dx(Fmy - tau_xy)")
    problem.add_equation("dt(mz)  = -dx(Fmz - tau_xz)")
    problem.add_equation("dt(Bx)  = eta*dx(dx(Bx))")
    problem.add_equation("dt(By)  = -dx(FBy) + eta*dx(dx(By))")
    problem.add_equation("dt(Bz)  = -dx(FBz) + eta*dx(dx(Bz))")

    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = t_final

    # ── set initial conditions ───────────────────────────────────────────
    if np.min(init_arrays[0]) <= rho_floor:
        raise RuntimeError("Initial density is non-positive or below rho_floor.")

    for fld, arr in zip(state_fields, init_arrays):
        fld["g"] = arr

    # ── CFL ──────────────────────────────────────────────────────────────
    dealias_scale = xbasis.dealias
    u_vec = dist.VectorField(xcoord, name="u_vec", bases=xbasis)
    u_vec.change_scales(dealias_scale)
    CFL = d3.CFL(solver, initial_dt=min(dt_initial, dt_max), cadence=1,
                 safety=cfl_safety, max_dt=dt_max, min_dt=1e-8,
                 max_change=1.1, min_change=0.5, threshold=0.01)
    CFL.add_velocity(u_vec)
    cf_expr = np.sqrt(
        0.5 * (gamma_par + gamma_perp) * ((p_par + 2.0 * p_perp) / 3.0) / (rho + rho_floor)
        + B2 / (rho + rho_floor))
    flow = d3.GlobalFlowProperty(solver, cadence=sample_every)
    flow.add_property(np.abs(mx / (rho + rho_floor)) + cf_expr, name="char_speed")

    # ── reference diagnostics ────────────────────────────────────────────
    Eperp0 = float(np.mean(init_arrays[5]**2 + init_arrays[6]**2))
    Etot0  = float(np.mean(cgl_total_energy_density_np(
        *init_arrays,
        gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor)))

    # ── snapshot storage ─────────────────────────────────────────────────
    snap_t = []
    snap_fields = {name: [] for name in FIELD_NAMES}
    status    = "finished"
    wall_t0   = time.time()

    print(f"[start] N={N} Lx={Lx:.2f} "
          f"mu={mu_visc:.2e} eta={eta:.2e} filter={'on' if filter_on else 'off'}")

    # ── main loop ────────────────────────────────────────────────────────
    while solver.proceed:
        for fld in state_fields:
            fld.change_scales(dealias_scale)
        u_vec["g"][0] = mx["g"] / np.maximum(rho["g"], rho_floor)

        dt = CFL.compute_timestep()
        solver.step(dt)

        # spectral dealiasing
        if filter_on and solver.iteration % filter_cadence == 0:
            for fld in state_fields:
                fld.change_scales(filter_fraction)
                fld.change_scales(1)

        if not all(np.all(np.isfinite(fld["g"])) for fld in state_fields):
            status = "non-finite"
            break

        for fld in state_fields:
            fld.change_scales(1)

        # extract arrays
        arrs = [np.array(fld["g"]) for fld in state_fields]
        ra, mxa, mya, mza, Bxa, Bya, Bza = arrs

        # diagnostics for stopping guards
        Eperp  = float(np.mean(Bya**2 + Bza**2))
        Egrow  = Eperp / max(Eperp0, 1e-30)
        s_norm = float(max(np.max(np.abs(a)) for a in arrs))
        hk_pf, _, _ = spectral_resolution_metrics(Bya)

        # snapshot
        do_snap = (len(snap_t) == 0
                   or solver.iteration % sample_every == 0
                   or not solver.proceed)
        if do_snap:
            snap_t.append(float(solver.sim_time))
            for name, arr in zip(FIELD_NAMES, arrs):
                snap_fields[name].append(arr.copy())

        if solver.iteration % print_every == 0 or len(snap_t) <= 1:
            print(f"  iter={solver.iteration:6d} t={solver.sim_time:.4f} dt={dt:.2e} "
                  f"Egrow={Egrow:.3f} rho=[{np.min(ra):.4f},{np.max(ra):.4f}] "
                  f"|By|max={np.max(np.abs(Bya)):.3e} hk={hk_pf:.3f}")

        # ── stopping guards ──────────────────────────────────────────────
        if max_energy_growth_factor is not None and Egrow >= max_energy_growth_factor:
            status = "Egrow ceiling"
            break
        if hk_pf >= max_high_k_power:
            status = "high-k ceiling"
            break
        if np.min(ra) <= 5.0 * rho_floor:
            status = "density floor"
            break
        if s_norm > max_state_norm:
            status = "state norm ceiling"
            break
        if solver.iteration >= max_steps:
            status = "max steps"
            break
        if time.time() - wall_t0 > wall_time_limit:
            status = "wall time"
            break

    t_end = snap_t[-1] if snap_t else 0.0
    print(f"[done] status={status} t={t_end:.4f} snapshots={len(snap_t)}")

    result = {
        "t": np.array(snap_t),
        "x": np.asarray(x).ravel().copy(),
        "status": status,
    }
    for name in FIELD_NAMES:
        result[name] = np.array(snap_fields[name])   # (n_t, n_x)

    return result


def run_eigenmode_firehose_case(case, gamma_par=3.0, gamma_perp=2.0, B_floor=1e-6):
    """Convenience wrapper: build eigenmode IC from *case* dict, then run.

    Accepts the same *case* dict as the old interface (rho0, Bx0,
    mode_number, amplitude, mu_visc, eta, …) and returns the (t, x) field
    dictionary produced by ``run_firehose_case``, with extra keys
    ``"sigma"`` and ``"linear"`` from the eigenmode analysis.
    """
    seed = build_seed_state_arrays(case, gamma_par=gamma_par,
                                   gamma_perp=gamma_perp, B_floor=B_floor)
    Lx = float(case.get("Lx", 12.0))
    k_phys = 2.0 * np.pi * int(case["mode_number"]) / Lx
    lin = build_linear_operator(
        float(case["rho0"]), float(case["Bx0"]), k_phys,
        float(case["mu_visc"]), float(case["eta"]),
        gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor)

    # Forward simulation params that aren't field arrays or eigenmode-specific
    _skip = {"rho0", "Bx0", "mode_number", "amplitude", "N", "label"}
    sim_kw = {k: case[k] for k in case if k not in _skip}
    sim_kw.setdefault("Lx", Lx)

    result = run_firehose_case(
        seed,
        gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor,
        **sim_kw,
    )
    result["sigma"] = lin["best_sigma"]
    result["linear"] = lin
    return result


# ── ML surrogate utilities ───────────────────────────────────────────────

NN_INPUT_FEATURES = ("rho", "ux", "uy", "uz", "Bx", "By", "Bz")


# ── Rotation augmentation ────────────────────────────────────────────────


def random_rotation_matrices_np(n_samples, rng=None):
    """Sample *n_samples* uniformly random 3x3 rotation matrices (NumPy)."""
    if rng is None:
        rng = np.random.default_rng()
    q = rng.normal(size=(n_samples, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y),
    ], axis=1).reshape(n_samples, 3, 3)


def rotate_state_vectors_np(X_raw, rotation_matrices):
    """Rotate velocity (cols 1-3) and B-field (cols 4-6) with the same rotation."""
    X_rot = X_raw.copy()
    X_rot[:, 1:4] = np.einsum('nij,nj->ni', rotation_matrices, X_raw[:, 1:4])
    X_rot[:, 4:7] = np.einsum('nij,nj->ni', rotation_matrices, X_raw[:, 4:7])
    return X_rot


def random_rotation_matrices_torch(n_samples, device, dtype):
    """Sample *n_samples* uniformly random 3x3 rotation matrices (torch)."""
    q = torch.randn(n_samples, 4, device=device, dtype=dtype)
    q = q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=1)
    mats = torch.stack([
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y),
    ], dim=1)
    return mats.reshape(n_samples, 3, 3)


def rotate_state_vectors_torch(X_batch):
    """On-the-fly rotation augmentation for a training batch (torch)."""
    rotations = random_rotation_matrices_torch(
        X_batch.shape[0], device=X_batch.device, dtype=X_batch.dtype
    )
    X_rot = X_batch.clone()
    X_rot[:, 1:4] = torch.einsum('nij,nj->ni', rotations, X_batch[:, 1:4])
    X_rot[:, 4:7] = torch.einsum('nij,nj->ni', rotations, X_batch[:, 4:7])
    return X_rot


# ── MLP model ────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """Compact MLP surrogate: (rho, ux, uy, uz, Bx, By, Bz) -> (log p_par, log p_perp).

    After training, call ``set_normalization`` to embed normalization statistics
    into the model so that ``predict_np`` / ``predict_t`` work without explicit
    norm arrays.  The buffers move automatically with ``.to(device)`` / ``.double()``.
    """

    def __init__(self, n_inputs=7, n_outputs=2, hidden=64, n_hidden=3, dropout=0.10):
        super().__init__()
        layers = []
        in_dim = n_inputs
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_outputs))
        self.net = nn.Sequential(*layers)
        # Normalization buffers — populated by set_normalization()
        self.register_buffer('x_mean', None)
        self.register_buffer('x_std',  None)
        self.register_buffer('y_mean', None)
        self.register_buffer('y_std',  None)

    def forward(self, x):
        return self.net(x)

    def set_normalization(self, x_mean, x_std, y_mean, y_std):
        """Store normalization stats as buffers (move with model to any device/dtype)."""
        def _t(a):
            return torch.tensor(np.asarray(a, dtype=np.float32).ravel())
        self.register_buffer('x_mean', _t(x_mean))
        self.register_buffer('x_std',  _t(x_std))
        self.register_buffer('y_mean', _t(y_mean))
        self.register_buffer('y_std',  _t(y_std))

    def predict_t(self, X_t):
        """Raw-input → (p_par, p_perp) as a torch tensor.

        *X_t* is an un-normalised ``(..., 7)`` tensor of state vectors.
        Returns ``(..., 2)`` tensor of pressures (not log-pressures).
        Normalization buffers must have been set via ``set_normalization``.
        """
        x_mean = self.x_mean.to(device=X_t.device, dtype=X_t.dtype)
        x_std  = self.x_std.to(device=X_t.device,  dtype=X_t.dtype)
        y_mean = self.y_mean.to(device=X_t.device, dtype=X_t.dtype)
        y_std  = self.y_std.to(device=X_t.device,  dtype=X_t.dtype)
        Xn = (X_t - x_mean) / x_std
        y_norm = self.net(Xn)
        return torch.exp(y_norm * y_std + y_mean)

    def predict_np(self, X_raw_np, device=None):
        """Raw-input → (p_par, p_perp) as a NumPy array.

        *X_raw_np* is an ``(n, 7)`` float32 array of un-normalised state vectors.
        Returns ``(n, 2)`` float32 NumPy array of pressures.
        """
        if device is None:
            device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        X_raw_np = np.asarray(X_raw_np, dtype=np.float64)
        with torch.no_grad():
            X_t = torch.tensor(X_raw_np, dtype=model_dtype, device=device)
            result = self.predict_t(X_t)
        return result.cpu().numpy()


# ── Unified NN / CGL closure ────────────────────────────────────────────


def pressures_nn_aniso(rho, u1, u2, u3, B1, B2, B3,
                       net=None, use_cgl=False,
                       gamma_par=3.0, gamma_perp=2.0, B_floor=1e-3):
    """Return (p_par, p_perp) from either exact CGL or a trained NN closure.

    When *use_cgl* is False, *net* must be an ``MLP`` whose normalization
    statistics have been set via ``net.set_normalization(...)``.
    """
    if use_cgl:
        p_par, p_perp, _, _ = cgl_pressures_torch(
            rho, B1, B2, B3, gamma_par, gamma_perp, B_floor)
        return p_par, p_perp
    if net is None:
        raise ValueError("`net` must be provided when use_cgl=False")
    X = torch.stack([rho, u1, u2, u3, B1, B2, B3], dim=-1)
    y = net.predict_t(X)
    return y[..., 0], y[..., 1]


def flux_mhd_aniso(Q, net=None, use_cgl=False,
                   gamma_par=3.0, gamma_perp=2.0, B_floor=1e-3):
    """1-D anisotropic MHD conservative flux ``F(Q)`` (torch, differentiable).

    Computes the same 7-component flux as :func:`_flux_torch` but can use
    either the analytical CGL closure (*use_cgl=True*) or a trained neural
    network (*net*) for the pressure law.  Used by the physics-informed
    loss functions and by :func:`dispersion_from_jacobian`.

    Note: *B_floor* defaults to 1e-3 (rather than 1e-6 in the linear-analysis
    routines) to ensure smoother gradients during ML training.
    """
    rho = Q[0]; m1 = Q[1]; m2 = Q[2]; m3 = Q[3]
    B1  = Q[4]; B2  = Q[5]; B3  = Q[6]
    rho_safe = torch.clamp(rho, min=1e-8)
    u1, u2, u3 = m1 / rho_safe, m2 / rho_safe, m3 / rho_safe
    p_par, p_perp = pressures_nn_aniso(
        rho_safe, u1, u2, u3, B1, B2, B3,
        net=net, use_cgl=use_cgl,
        gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor,
    )
    Bsq  = B1**2 + B2**2 + B3**2
    Bmag = torch.sqrt(Bsq + B_floor**2)
    bx, by, bz = B1 / Bmag, B2 / Bmag, B3 / Bmag
    dp = p_par - p_perp
    Pxx = p_perp + dp * bx * bx
    Pxy = dp * bx * by
    Pxz = dp * bx * bz
    return torch.stack([
        m1,
        m1 * u1 + Pxx + 0.5 * Bsq - B1 * B1,
        m1 * u2 + Pxy - B1 * B2,
        m1 * u3 + Pxz - B1 * B3,
        u1 * B1 - u1 * B1,
        u1 * B2 - u2 * B1,
        u1 * B3 - u3 * B1,
    ])


def dispersion_from_jacobian(rho0, u10, u20, u30, B1_0, B2_0, B3_0,
                              net=None, use_cgl=False, ks=None,
                              device=None,
                              gamma_par=3.0, gamma_perp=2.0, B_floor=1e-3):
    """Compute flux-Jacobian eigenvalues at an equilibrium state.

    Builds the 7x7 flux Jacobian ``A = dF/dQ`` using either the CGL closure
    or a trained neural-network closure, then returns eigenvalues and
    dispersion curves ``omega = k * lambda`` for a list of wavenumbers.

    Parameters
    ----------
    rho0, u10, u20, u30, B1_0, B2_0, B3_0 : float
        Equilibrium state (primitive variables).
    net : MLP or None
        Trained closure model (ignored when *use_cgl=True*).
    ks : list of float
        Wavenumbers at which to evaluate omega = k * eigenvalues.

    Returns
    -------
    (A0, eigvals, omegas, Q0) : tuple of Tensors
    """
    if ks is None:
        ks = [0.1, 1.0, 10.0]
    if device is None:
        if net is not None:
            device = next(net.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q0 = torch.tensor(
        [rho0, rho0 * u10, rho0 * u20, rho0 * u30, B1_0, B2_0, B3_0],
        dtype=torch.double, device=device, requires_grad=True,
    )
    if net is not None:
        model_d = net.to(device).double()
        model_d.eval()
    else:
        model_d = None

    def F_of_Q(Q):
        return flux_mhd_aniso(
            Q, net=model_d, use_cgl=use_cgl,
            gamma_par=gamma_par, gamma_perp=gamma_perp, B_floor=B_floor,
        )

    A0 = torch.autograd.functional.jacobian(F_of_Q, Q0)
    eigvals = torch.linalg.eigvals(A0)
    omegas = torch.outer(
        torch.tensor(ks, dtype=torch.double, device=device), eigvals)
    return A0.detach(), eigvals.detach(), omegas.detach(), Q0.detach()


# ── Physics-informed loss utilities ──────────────────────────────────────


def sample_collocation_Q(n, rho_range=(0.5, 2.0), Bmag_range=(0.5, 2.0),
                         device='cpu', dtype=torch.float64):
    """Sample conserved-variable collocation states for physics losses.

    Returns an ``(n, 7)`` tensor of ``Q = (rho, 0, 0, 0, Bx, By, Bz)``
    with zero momentum.  Half the states are axis-aligned (B along x),
    the other half have randomly oriented B fields.
    """
    rho = torch.empty(n, dtype=dtype, device=device).uniform_(*rho_range)
    Bmag = torch.empty(n, dtype=dtype, device=device).uniform_(*Bmag_range)

    n_axis = n // 2
    n_rand = n - n_axis

    # Axis-aligned
    Bx_a = Bmag[:n_axis]
    By_a = torch.zeros(n_axis, dtype=dtype, device=device)
    Bz_a = torch.zeros(n_axis, dtype=dtype, device=device)

    # Random orientation
    dirs = torch.randn(n_rand, 3, dtype=dtype, device=device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    Bx_r = Bmag[n_axis:] * dirs[:, 0]
    By_r = Bmag[n_axis:] * dirs[:, 1]
    Bz_r = Bmag[n_axis:] * dirs[:, 2]

    zeros = torch.zeros(n, dtype=dtype, device=device)
    return torch.stack([
        rho, zeros, zeros, zeros,
        torch.cat([Bx_a, Bx_r]),
        torch.cat([By_a, By_r]),
        torch.cat([Bz_a, Bz_r]),
    ], dim=1)


def cgl_log_pressure_jacobian(Q, gamma_par=3.0, gamma_perp=2.0, B_floor=1e-3):
    r"""Analytical 2×7 Jacobian of log p_par, log p_perp w.r.t. Q.

    Parameters
    ----------
    Q : Tensor, shape ``(7,)``
        Conserved state ``(rho, mx, my, mz, Bx, By, Bz)``.

    Returns
    -------
    Tensor, shape ``(2, 7)``
    """
    rho = Q[0]
    Bx, By, Bz = Q[4], Q[5], Q[6]
    B2f = Bx ** 2 + By ** 2 + Bz ** 2 + B_floor ** 2

    J = torch.zeros(2, 7, dtype=Q.dtype, device=Q.device)
    # d(log p_par)/dQ
    J[0, 0] = gamma_par / rho
    J[0, 4] = -(gamma_par - 1.0) * Bx / B2f
    J[0, 5] = -(gamma_par - 1.0) * By / B2f
    J[0, 6] = -(gamma_par - 1.0) * Bz / B2f
    # d(log p_perp)/dQ
    J[1, 0] = 1.0 / rho
    J[1, 4] = (gamma_perp - 1.0) * Bx / B2f
    J[1, 5] = (gamma_perp - 1.0) * By / B2f
    J[1, 6] = (gamma_perp - 1.0) * Bz / B2f
    return J


def _ml_log_pressures_from_Q(Q, net):
    """Log-pressures from conserved state Q via trained ML model.

    Parameters
    ----------
    Q : Tensor, shape ``(7,)``
        Should have ``requires_grad=True`` when used inside Jacobian
        computations.

    Returns
    -------
    Tensor, shape ``(2,)`` — ``(log p_par, log p_perp)``.
    """
    rho = torch.clamp(Q[0], min=1e-8)
    ux, uy, uz = Q[1] / rho, Q[2] / rho, Q[3] / rho
    X = torch.stack([rho, ux, uy, uz, Q[4], Q[5], Q[6]])
    pressures = net.predict_t(X)
    return torch.log(torch.clamp(pressures, min=1e-30))


def closure_tangent_loss(net, Q_batch,
                         gamma_par=3.0, gamma_perp=2.0, B_floor=1e-3):
    r"""Closure tangent matching loss (Method 1).

    Computes the 2×7 Jacobian of (log p_par, log p_perp) w.r.t. the conserved 
    # state **Q** for both the ML closure and
    analytical CGL, returning the mean squared Frobenius-norm difference:

    L = (1/N)*\sum_{i=1}^{N}| J^{ML}(Q_i) - J^{CGL}(Q_i)|^2

    Uses ``create_graph=True`` so gradients propagate to model parameters
    (second-order w.r.t. model weights).  Cost: 2 backward passes per
    collocation point.
    """
    n = Q_batch.shape[0]
    total = torch.tensor(0.0, dtype=Q_batch.dtype, device=Q_batch.device)

    for i in range(n):
        Q_i = Q_batch[i].detach().requires_grad_(True)
        J_cgl = cgl_log_pressure_jacobian(Q_i, gamma_par, gamma_perp, B_floor)
        J_ml = torch.autograd.functional.jacobian(
            lambda q: _ml_log_pressures_from_Q(q, net), Q_i,
            create_graph=True,
        )
        total = total + torch.sum((J_ml - J_cgl) ** 2)

    return total / n


def flux_jacobian_loss(net, Q_batch,
                       gamma_par=3.0, gamma_perp=2.0, B_floor=1e-3):
    r"""Full 7×7 flux-Jacobian matching loss (Method 2).

    .. math::

        L = (1/N)*\sum_{i=1}^{N} (dF/dQ|_{ML}(Q_i) - dF/dQ|_{CGL}(Q_i))^2

    Cost: 7 backward passes with ``create_graph=True`` per point
    (≈3.5× the cost of :func:`closure_tangent_loss`).
    """
    n = Q_batch.shape[0]
    total = torch.tensor(0.0, dtype=Q_batch.dtype, device=Q_batch.device)

    for i in range(n):
        Q_i = Q_batch[i].detach().requires_grad_(True)
        A_ml = torch.autograd.functional.jacobian(
            lambda q: flux_mhd_aniso(q, net=net, use_cgl=False,
                                     gamma_par=gamma_par,
                                     gamma_perp=gamma_perp,
                                     B_floor=B_floor),
            Q_i, create_graph=True,
        )
        with torch.no_grad():
            A_cgl = torch.autograd.functional.jacobian(
                lambda q: flux_mhd_aniso(q, net=None, use_cgl=True,
                                         gamma_par=gamma_par,
                                         gamma_perp=gamma_perp,
                                         B_floor=B_floor),
                Q_i,
            )
        total = total + torch.sum((A_ml - A_cgl) ** 2)

    return total / n


def growth_rate_loss(net, Q_batch, k=1.0,
                     gamma_par=3.0, gamma_perp=2.0, B_floor=1e-3,
                     temperature=5.0):
    r"""Growth-rate matching loss via logsumexp soft-max (Method 3).

    Compares the soft-max of :math:`|Im}(\omega)|` between ML
    and CGL flux Jacobians at each collocation state.

    .. warning::

        Differentiating through eigenvalue decomposition is numerically
        fragile near degenerate eigenvalues and complex-conjugate
        crossings.  Prefer :func:`closure_tangent_loss` or
        :func:`flux_jacobian_loss` unless growth-rate accuracy is the
        dominant concern.

    Parameters
    ----------
    k : float
        Wavenumber at which to evaluate :math:`\omega = k \lambda`.
    temperature : float
        Sharpness of the logsumexp soft-max.  Higher values approximate
        the hard max more closely.
    """
    n = Q_batch.shape[0]
    total = torch.tensor(0.0, dtype=Q_batch.dtype, device=Q_batch.device)

    for i in range(n):
        Q_i = Q_batch[i].detach().requires_grad_(True)

        A_ml = torch.autograd.functional.jacobian(
            lambda q: flux_mhd_aniso(q, net=net, use_cgl=False,
                                     gamma_par=gamma_par,
                                     gamma_perp=gamma_perp,
                                     B_floor=B_floor),
            Q_i, create_graph=True,
        )
        with torch.no_grad():
            A_cgl = torch.autograd.functional.jacobian(
                lambda q: flux_mhd_aniso(q, net=None, use_cgl=True,
                                         gamma_par=gamma_par,
                                         gamma_perp=gamma_perp,
                                         B_floor=B_floor),
                Q_i,
            )

        eig_ml = torch.linalg.eigvals(A_ml)
        eig_cgl = torch.linalg.eigvals(A_cgl)

        # Smooth |Im(eig)| — use sqrt(x² + eps) for differentiability at 0
        eps = 1e-12
        gamma_ml = k * torch.sqrt(eig_ml.imag ** 2 + eps)
        gamma_cgl = k * torch.sqrt(eig_cgl.imag ** 2 + eps)

        # Soft-max via logsumexp
        g_max_ml = torch.logsumexp(temperature * gamma_ml, dim=0) / temperature
        g_max_cgl = (torch.logsumexp(temperature * gamma_cgl, dim=0)
                     / temperature).detach()
        total = total + (g_max_ml - g_max_cgl) ** 2

    return total / n


# ── Synthetic data generation ────────────────────────────────────────────


def _sample_unit_vectors(n_samples, rng):
    """Sample *n_samples* uniformly distributed unit vectors on S²."""
    vec = rng.normal(size=(n_samples, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec


def _augment_directions(B_dirs, v_dirs, rng,
                        axis_fraction, near_axis_fraction, near_axis_noise):
    """Overwrite a fraction of direction arrays with axis-aligned / near-axis vectors.

    Modifies *B_dirs* and *v_dirs* **in-place**.  A fraction
    *axis_fraction* of rows are set to exact coordinate-axis directions,
    and a further *near_axis_fraction* are perturbed slightly off-axis.
    This ensures the training data covers the axis-aligned states that
    commonly appear in 1-D simulations.
    """
    n = len(B_dirs)

    # Exact axis-aligned directions
    n_axis = int(axis_fraction * n)
    axis_idx = rng.choice(n, size=n_axis, replace=False)
    axis_dirs = np.eye(3)[rng.integers(0, 3, size=n_axis)]
    axis_signs = rng.choice([-1.0, 1.0], size=(n_axis, 1))
    B_dirs[axis_idx] = axis_dirs * axis_signs
    v_dirs[axis_idx] = axis_dirs * axis_signs

    # Near-axis directions (small perturbation off a coordinate axis)
    remaining_idx = np.setdiff1d(np.arange(n), axis_idx, assume_unique=False)
    n_near = int(near_axis_fraction * n)
    near_idx = rng.choice(remaining_idx, size=n_near, replace=False)
    pref_axis = np.eye(3)[rng.integers(0, 3, size=n_near)]
    noise = near_axis_noise * rng.normal(size=(n_near, 3))
    B_dirs[near_idx] = pref_axis + noise
    B_dirs[near_idx] /= np.linalg.norm(B_dirs[near_idx], axis=1, keepdims=True)
    v_dirs[near_idx] = pref_axis + noise
    v_dirs[near_idx] /= np.linalg.norm(v_dirs[near_idx], axis=1, keepdims=True)


def generate_synthetic_cgl_dataset(
    n_samples=30000,
    rho_range=(0.35, 2.35),
    Bmag_range=(0.35, 2.25),
    vmag_range=(0.0, 1.0),
    axis_fraction=0.25,
    near_axis_fraction=0.15,
    near_axis_noise=0.08,
    gamma_par=3.0,
    gamma_perp=2.0,
    seed=42,
):
    """Generate a broad synthetic CGL state-space dataset.

    Samples ``(rho, |B|, |v|)`` uniformly across the given ranges and
    assigns random 3-D directions (with axis-aligned / near-axis
    augmentation via :func:`_augment_directions`).  CGL pressures are
    computed analytically by :func:`cgl_pressures_np`.

    Returns
    -------
    dict
        ``X`` (n, 7), ``y`` (n, 2), ``rho``, ``Bmag``, ``vmag`` arrays.
    """
    rng = np.random.default_rng(seed)
    rho_s = rng.uniform(rho_range[0], rho_range[1], n_samples)
    Bmag_s = rng.uniform(Bmag_range[0], Bmag_range[1], n_samples)
    vmag_s = rng.uniform(vmag_range[0], vmag_range[1], n_samples)

    B_dirs = _sample_unit_vectors(n_samples, rng)
    v_dirs = _sample_unit_vectors(n_samples, rng)
    _augment_directions(B_dirs, v_dirs, rng,
                        axis_fraction, near_axis_fraction, near_axis_noise)

    B_samples = Bmag_s[:, None] * B_dirs
    V_samples = vmag_s[:, None] * v_dirs
    Bx, By, Bz = B_samples.T
    Vx, Vy, Vz = V_samples.T

    ppar, pperp, _, _ = cgl_pressures_np(rho_s, Bx, By, Bz,
                                         gamma_par=gamma_par, gamma_perp=gamma_perp)

    X = np.stack([rho_s, Vx, Vy, Vz, Bx, By, Bz], axis=-1)
    y = np.stack([ppar, pperp], axis=-1)
    return {"X": X, "y": y, "rho": rho_s, "Bmag": Bmag_s, "vmag": vmag_s}


def generate_lumped_campaign_dataset(
    campaign_centers,
    points_per_campaign=1200,
    sigma_B=0.05,
    sigma_rho=0.05,
    rho_clip=(0.35, 2.35),
    Bmag_clip=(0.35, 2.25),
    vmag_range=(0.0, 1.0),
    axis_fraction=0.25,
    near_axis_fraction=0.15,
    near_axis_noise=0.08,
    gamma_par=3.0,
    gamma_perp=2.0,
    seed=2026,
):
    """Generate a lumped dataset from localised simulation campaigns.

    Each campaign contributes *points_per_campaign* samples centred on a
    different ``(Bx0, rho0)`` equilibrium, with Gaussian scatter controlled
    by *sigma_B* and *sigma_rho*.  Direction augmentation and CGL
    pressure computation follow the same pattern as
    :func:`generate_synthetic_cgl_dataset`.

    Parameters
    ----------
    campaign_centers : (n_campaigns, 2) array
        Each row is ``(Bx0_center, rho0_center)``.

    Returns
    -------
    dict
        ``X`` (n, 7), ``y`` (n, 2), ``campaign_id`` (n,) arrays.
    """
    rng = np.random.default_rng(seed)
    n_campaigns = len(campaign_centers)
    B_list, rho_list, vmag_list, cid_list = [], [], [], []

    for cid, (B_c, rho_c) in enumerate(campaign_centers):
        B_loc = np.clip(rng.normal(B_c, sigma_B, size=points_per_campaign),
                        Bmag_clip[0], Bmag_clip[1])
        rho_loc = np.clip(rng.normal(rho_c, sigma_rho, size=points_per_campaign),
                          rho_clip[0], rho_clip[1])
        vmag_loc = rng.uniform(vmag_range[0], vmag_range[1], size=points_per_campaign)
        B_list.append(B_loc)
        rho_list.append(rho_loc)
        vmag_list.append(vmag_loc)
        cid_list.append(np.full(points_per_campaign, cid, dtype=int))

    Bmag_s = np.concatenate(B_list)
    rho_s = np.concatenate(rho_list)
    vmag_s = np.concatenate(vmag_list)
    campaign_id = np.concatenate(cid_list)

    n_total = len(Bmag_s)
    B_dirs = _sample_unit_vectors(n_total, rng)
    v_dirs = _sample_unit_vectors(n_total, rng)
    _augment_directions(B_dirs, v_dirs, rng,
                        axis_fraction, near_axis_fraction, near_axis_noise)

    B_samples = Bmag_s[:, None] * B_dirs
    V_samples = vmag_s[:, None] * v_dirs
    Bx, By, Bz = B_samples.T
    Vx, Vy, Vz = V_samples.T

    ppar, pperp, _, _ = cgl_pressures_np(rho_s, Bx, By, Bz,
                                         gamma_par=gamma_par, gamma_perp=gamma_perp)

    X = np.stack([rho_s, Vx, Vy, Vz, Bx, By, Bz], axis=-1)
    y = np.stack([ppar, pperp], axis=-1)
    return {"X": X, "y": y, "campaign_id": campaign_id}


# ── Data packaging ───────────────────────────────────────────────────────


def prepare_ml_data(X, y, train_fraction=0.8, target_log_eps=1e-12,
                    batch_size_train=256, batch_size_test=512,
                    undersample_fraction=None, undersample_seed=None):
    """Split, log-transform, normalize, and package into dataloaders.

    Parameters
    ----------
    undersample_fraction : float or None
        If given (0 < value <= 1), randomly retain this fraction of the
        *training* set after the train/test split.  The test set is
        unaffected.  Useful to reduce over-represented regions.
    undersample_seed : int or None
        Random seed for reproducible undersampling.

    Returns a dict with all arrays, normalization stats, and dataloaders.
    """
    n = len(X)
    n_train = max(1, min(int(train_fraction * n), n - 1))
    y_log = np.log(np.clip(y, target_log_eps, None))

    X_train_raw = X[:n_train].astype(np.float32)
    y_train_raw = y[:n_train].astype(np.float32)
    y_train_log = y_log[:n_train].astype(np.float32)
    X_test_raw  = X[n_train:].astype(np.float32)
    y_test_raw  = y[n_train:].astype(np.float32)
    y_test_log  = y_log[n_train:].astype(np.float32)

    if undersample_fraction is not None and undersample_fraction < 1.0:
        rng = np.random.default_rng(undersample_seed)
        keep = rng.choice(len(X_train_raw),
                          size=max(1, int(undersample_fraction * len(X_train_raw))),
                          replace=False)
        keep.sort()
        X_train_raw = X_train_raw[keep]
        y_train_raw = y_train_raw[keep]
        y_train_log = y_train_log[keep]

    eps = 1e-12
    x_mean = X_train_raw.mean(axis=0, keepdims=True)
    x_std  = X_train_raw.std(axis=0, keepdims=True) + eps
    y_mean = y_train_log.mean(axis=0, keepdims=True)
    y_std  = y_train_log.std(axis=0, keepdims=True) + eps

    X_train_norm = (X_train_raw - x_mean) / x_std
    y_train_norm = (y_train_log - y_mean) / y_std
    X_test_norm  = (X_test_raw - x_mean) / x_std
    y_test_norm  = (y_test_log - y_mean) / y_std

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_train_norm)),
        batch_size=batch_size_train, shuffle=True,
    )
    test_dl = DataLoader(
        TensorDataset(torch.tensor(X_test_norm), torch.tensor(y_test_norm)),
        batch_size=batch_size_test, shuffle=False,
    )
    return {
        "X_train_raw": X_train_raw, "y_train_raw": y_train_raw,
        "X_test_raw": X_test_raw, "y_test_raw": y_test_raw,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "train_dl": train_dl, "test_dl": test_dl,
        "n_train": len(X_train_raw),
    }


# ── Training ─────────────────────────────────────────────────────────────


def _run_epoch(model, dataloader, optimizer, scheduler, loss_fn, device,
               train=False, rotate_batches=False,
               physics_closure=None, lambda_phys=0.0):
    """Run a single training or evaluation epoch.

    Parameters
    ----------
    physics_closure : callable or None
        ``fn(model) -> scalar tensor`` that computes a physics-informed
        regularisation loss (e.g. tangent matching).  Called once per
        mini-batch during training and added with weight *lambda_phys*.
    lambda_phys : float
        Multiplicative weight for the physics loss.

    Returns
    -------
    (data_loss, phys_loss) : tuple[float, float]
        Mean per-sample data loss and mean per-sample physics loss.
    """
    model.train(train)
    # Determine model dtype so we can cast batches (guards against
    # torch.set_default_dtype(float64) producing double-precision DataLoaders
    # while model weights are float32).
    model_dtype = next(model.parameters()).dtype
    total_data_loss = 0.0
    total_phys_loss = 0.0
    if not train:
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device=device, dtype=model_dtype), yb.to(device=device, dtype=model_dtype)
                pred = model(xb)
                total_data_loss += loss_fn(pred, yb).item() * len(xb)
    else:
        for xb, yb in dataloader:
            xb, yb = xb.to(device=device, dtype=model_dtype), yb.to(device=device, dtype=model_dtype)
            if rotate_batches:
                xb = rotate_state_vectors_torch(xb)
            pred = model(xb)
            data_loss = loss_fn(pred, yb)

            combined = data_loss
            if physics_closure is not None and lambda_phys > 0:
                p_loss = physics_closure(model)
                combined = data_loss + lambda_phys * p_loss
                total_phys_loss += p_loss.item() * len(xb)

            optimizer.zero_grad()
            combined.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_data_loss += data_loss.item() * len(xb)
    n = len(dataloader.dataset)
    return total_data_loss / n, total_phys_loss / n


def train_surrogate(
    model,
    train_dl,
    test_dl,
    device="cpu",
    max_epochs=500,
    patience=50,
    base_lr=1e-3,
    weight_decay=1e-4,
    use_rotation_augmentation=True,
    min_delta=1e-5,
    print_every=10,
    norm_data=None,
    # ── Physics-informed regularisation ──
    physics_loss_fn=None,
    lambda_phys=0.0,
    lambda_warmup_epochs=0,
    n_colloc=16,
    colloc_rho_range=(0.5, 2.0),
    colloc_Bmag_range=(0.5, 2.0),
):
    """Train an MLP surrogate with AdamW + OneCycleLR + early stopping.

    Parameters
    ----------
    norm_data : dict or None
        If given (typically the dict returned by ``prepare_ml_data``), the
        normalization statistics (``x_mean``, ``x_std``, ``y_mean``,
        ``y_std``) are embedded into *model* via ``model.set_normalization``
        before the function returns.
    physics_loss_fn : callable or None
        A physics-informed loss with signature ``fn(net, Q_batch) ->
        scalar tensor``.  See :func:`closure_tangent_loss`,
        :func:`flux_jacobian_loss`, or :func:`growth_rate_loss`.
        Normalization buffers must already be set on the model **before**
        calling ``train_surrogate`` when this is used (so that
        ``model.predict_t`` works inside the loss).
    lambda_phys : float
        Final weight for the physics loss.  Linearly ramped from 0 over
        *lambda_warmup_epochs*.
    lambda_warmup_epochs : int
        Number of epochs over which *lambda_phys* is linearly ramped
        from 0 to its final value.
    n_colloc : int
        Number of collocation states sampled per mini-batch for the
        physics loss.
    colloc_rho_range, colloc_Bmag_range : tuple[float, float]
        Sampling ranges for collocation states.

    Returns (model, history_dict).
    """
    model = model.to(device)

    # If physics loss is requested, ensure normalization buffers are set
    # so that model.predict_t works from the very first epoch.
    if physics_loss_fn is not None and lambda_phys > 0 and norm_data is not None:
        if model.x_mean is None:
            model.set_normalization(
                norm_data['x_mean'], norm_data['x_std'],
                norm_data['y_mean'], norm_data['y_std'],
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=base_lr, epochs=max_epochs,
        steps_per_epoch=max(1, len(train_dl)),
        pct_start=0.2, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0,
    )

    train_losses, test_losses, phys_losses, lr_history = [], [], [], []
    best_test_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model_dtype = next(model.parameters()).dtype

    for epoch in range(1, max_epochs + 1):
        # ── Curriculum: ramp physics-loss weight ──
        if physics_loss_fn is not None and lambda_phys > 0:
            if lambda_warmup_epochs > 0:
                lam = lambda_phys * min(1.0, epoch / lambda_warmup_epochs)
            else:
                lam = lambda_phys

            def _phys_closure(net):
                Q = sample_collocation_Q(
                    n_colloc,
                    rho_range=colloc_rho_range,
                    Bmag_range=colloc_Bmag_range,
                    device=device, dtype=model_dtype,
                )
                return physics_loss_fn(net, Q)
        else:
            _phys_closure = None
            lam = 0.0

        tl, pl = _run_epoch(model, train_dl, optimizer, scheduler, loss_fn,
                            device, train=True,
                            rotate_batches=use_rotation_augmentation,
                            physics_closure=_phys_closure,
                            lambda_phys=lam)
        vl, _ = _run_epoch(model, test_dl, None, None, loss_fn, device,
                           train=False)
        train_losses.append(tl)
        test_losses.append(vl)
        phys_losses.append(pl)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))

        if vl < best_test_loss - min_delta:
            best_test_loss = vl
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epoch % print_every == 0 or epoch == 1:
            msg = (f"epoch {epoch:4d} | lr {lr_history[-1]:.2e} | "
                   f"train {tl:.4f} | test {vl:.4f} | "
                   f"best {best_test_loss:.4f} (ep {best_epoch})")
            if _phys_closure is not None:
                msg += f" | phys {pl:.4e} (λ={lam:.3e})"
            print(msg)

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    model.load_state_dict(best_state)
    if norm_data is not None:
        model.set_normalization(
            norm_data['x_mean'], norm_data['x_std'],
            norm_data['y_mean'], norm_data['y_std'],
        )
    return model, {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "phys_losses": phys_losses,
        "lr_history": lr_history,
        "best_test_loss": float(best_test_loss),
        "best_epoch": best_epoch,
    }


# ── Inference helpers ────────────────────────────────────────────────────


def ml_predict_pressures_np(X_raw_np, model, device=None):
    """Predict (p_par, p_perp) from raw state vectors (NumPy in/out).

    Normalization is handled internally by *model* (see ``MLP.predict_np``).
    The *device* argument is kept for call-site compatibility; when omitted
    the model's current device is used.
    """
    return model.predict_np(X_raw_np, device=device)


def r2_score_np(y_true, y_pred):
    """R-squared score (NumPy)."""
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 1e-14:
        return np.nan
    return 1.0 - np.sum((y_pred - y_true) ** 2) / denom




__all__ = [
    # CGL closure (NumPy + torch)
    "cgl_pressures_np",
    "cgl_pressures_torch",
    "cgl_total_energy_density_np",
    # Linear analysis / Dedalus
    "build_linear_operator",
    "build_seed_state_arrays",
    "FIELD_NAMES",
    "make_real_eigenmode_profile",
    "run_eigenmode_firehose_case",
    "run_firehose_case",
    # Spectral diagnostics
    "high_k_fraction",
    "nearest_snapshot",
    "spectral_mode_summary",
    "spectral_resolution_metrics",
    # ML surrogate
    "dispersion_from_jacobian",
    "flux_mhd_aniso",
    "generate_lumped_campaign_dataset",
    "generate_synthetic_cgl_dataset",
    "MLP",
    "ml_predict_pressures_np",
    "NN_INPUT_FEATURES",
    "prepare_ml_data",
    "pressures_nn_aniso",
    "r2_score_np",
    "random_rotation_matrices_np",
    "random_rotation_matrices_torch",
    "rotate_state_vectors_np",
    "rotate_state_vectors_torch",
    "train_surrogate",
    # Physics-informed losses
    "cgl_log_pressure_jacobian",
    "closure_tangent_loss",
    "flux_jacobian_loss",
    "growth_rate_loss",
    "sample_collocation_Q",
]
