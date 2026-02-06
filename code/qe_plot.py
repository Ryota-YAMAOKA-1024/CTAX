#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


HBAR2_OVER_2MN = 2.072  # meV * Å^2  (neutron: E[meV] = 2.072 * k^2[Å^-2])
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "code" else SCRIPT_DIR


def load_params_list_txt(path: str) -> dict:
    """
    Read key=value (or key: value) style config.
    - Lines starting with # are ignored
    - Inline comments after # are removed
    - Values are parsed as int or float when possible, otherwise kept as string
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parameter file not found: {p.resolve()}")

    params = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # remove inline comments
        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        if "=" in line:
            key, val = line.split("=", 1)
        elif ":" in line:
            key, val = line.split(":", 1)
        else:
            raise ValueError(f"Bad line (use key=value): {raw}")

        key = key.strip()
        val = val.strip()

        # preserve quoted strings verbatim (without quotes)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            params[key] = val[1:-1]
            continue

        # number parsing
        if re.fullmatch(r"[+-]?\d+", val):
            params[key] = int(val)
        else:
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val

    return params


def require(params: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in params]
    if missing:
        raise KeyError(f"Missing keys in list.txt: {missing}")


def q_to_Qmag(
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
    q110: float,
    q1m10: float,
    q001: float,
) -> np.ndarray:
    """
    Convert (H,K,L) coordinates into the orthogonal
    (1,1,0)/(1,-1,0)/(0,0,1) basis, then scale by the
    absolute magnitudes provided via list.txt.
    """
    h_plus = 0.5 * (h + k)
    h_minus = 0.5 * (h - k)
    comps = np.array(
        [
            h_plus * q110,
            h_minus * q1m10,
            l * q001,
        ]
    )
    return np.linalg.norm(comps, axis=0)


def gamma1(qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    return (
        np.cos(2.0 * np.pi * qx)
        + np.cos(2.0 * np.pi * qy)
        + np.cos(2.0 * np.pi * (qx + qy))
    )


def gamma2(qz: np.ndarray) -> np.ndarray:
    return np.cos(np.pi * qz)


def magnon_energy_meV(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, S: float, J1: float, J2: float
) -> np.ndarray:
    """
    E(q) = ħ ω_q in meV (assuming J1,J2 are in meV)
    """
    g1 = gamma1(qx, qy)
    g2 = gamma2(qz)

    A = J1 * (3.0 + 2.0 * g1) + 2.0 * J2 * (-1.0 + g2)
    B = J1 * (3.0 - 1.0 * g1) + 2.0 * J2 * (-1.0 + g2)

    omega2 = A * B
    # numerical safety: negative -> NaN (unphysical region for given parameters)
    omega2 = np.where(omega2 >= 0.0, omega2, np.nan)
    return S * np.sqrt(omega2)


def kinematics_Q(E: np.ndarray, Ef: float, two_theta_deg: float) -> np.ndarray:
    """
    For fixed-final-energy (Ef) kinematics:
      ki^2 = (E + Ef)/2.072, kf^2 = Ef/2.072  (E is energy transfer)
      Q^2 = ki^2 + kf^2 - 2 ki kf cos(2θ)
    => Q^2 = (E + 2Ef - 2 cos(2θ)*sqrt(Ef(E+Ef))) / 2.072
    """
    cos2t = np.cos(np.deg2rad(two_theta_deg))
    numerator = (E + 2.0 * Ef) - 2.0 * cos2t * np.sqrt(Ef * (E + Ef))
    Q2 = numerator / HBAR2_OVER_2MN
    Q2 = np.where(Q2 >= 0.0, Q2, np.nan)
    return np.sqrt(Q2)


def make_energy_grid(Emax: float, n: int = 800) -> np.ndarray:
    return np.linspace(0.0, float(Emax), int(n))


def plot_path(
    ax,
    x_path: np.ndarray,
    x_label: str,
    Q_path: np.ndarray,
    E_path: np.ndarray,
    E_grid: np.ndarray,
    Q_low: np.ndarray,
    Q_high: np.ndarray,
    title: str,
):
    """Plot dispersion vs. path parameter with Q-based kinematics on a twin axis."""
    (spin_line,) = ax.plot(x_path, E_path, label="spin-wave dispersion")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy transfer E (meV)")
    ax.grid(True, alpha=0.3)

    ax_q = ax.twiny()
    ax_q.patch.set_alpha(0.0)

    (low_line,) = ax_q.plot(
        Q_low,
        E_grid,
        label="kinematics lower (2θ=0°)",
        color="tab:orange",
    )
    (high_line,) = ax_q.plot(
        Q_high,
        E_grid,
        label="kinematics upper (2θ=2θ_max)",
        color="tab:green",
    )
    region = ax_q.fill_betweenx(
        E_grid,
        Q_low,
        Q_high,
        alpha=0.15,
        color="tab:green",
        label="measurable region",
    )

    finite_low = (
        np.nanmin(Q_low[np.isfinite(Q_low)]) if np.any(np.isfinite(Q_low)) else 0.0
    )
    finite_high = (
        np.nanmax(Q_high[np.isfinite(Q_high)]) if np.any(np.isfinite(Q_high)) else 1.0
    )
    ax_q.set_xlim(finite_low, finite_high)
    ax_q.set_xlabel(r"$Q\ (\AA^{-1})$")

    handles, labels = ax.get_legend_handles_labels()
    handles_q = [low_line, high_line, region]
    labels_q = [h.get_label() for h in handles_q]
    ax.legend(handles + handles_q, labels + labels_q, loc="best")


def main():
    if len(sys.argv) < 2:
        print("Usage: python qe_plot.py list.txt")
        sys.exit(1)

    params = load_params_list_txt(sys.argv[1])

    # Required physics params
    require(params, ["S", "J1", "J2", "Ef", "Ei_max", "two_theta_max_deg"])
    require(params, ["Q_110", "Q_1m10", "Q_001"])

    S = float(params["S"])
    J1 = float(params["J1"])
    J2 = float(params["J2"])

    Ef = float(params["Ef"])  # meV
    Ei_max = float(params["Ei_max"])  # meV
    two_theta_max = float(params["two_theta_max_deg"])  # deg
    Q_110 = float(params["Q_110"])
    Q_1m10 = float(params["Q_1m10"])
    Q_001 = float(params["Q_001"])

    # Sampling params (no hardcode of values; still require existence)
    require(
        params,
        ["H_min", "H_max", "n_H", "K_min", "K_max", "n_K", "L_min", "L_max", "n_L"],
    )

    H_min, H_max, n_H = (
        float(params["H_min"]),
        float(params["H_max"]),
        int(params["n_H"]),
    )
    K_min, K_max, n_K = (
        float(params["K_min"]),
        float(params["K_max"]),
        int(params["n_K"]),
    )
    L_min, L_max, n_L = (
        float(params["L_min"]),
        float(params["L_max"]),
        int(params["n_L"]),
    )

    # Energy grid for kinematics overlay (use Ei_max - Ef by default)
    E_transfer_max = max(0.0, Ei_max - Ef)
    E_grid_max = float(params.get("E_grid_max", E_transfer_max))
    E_grid_max = min(E_grid_max, E_transfer_max) if E_transfer_max > 0 else E_grid_max
    E_grid = make_energy_grid(E_grid_max, int(params.get("n_Egrid", 800)))

    Q_low = kinematics_Q(E_grid, Ef=Ef, two_theta_deg=0.0)
    Q_high = kinematics_Q(E_grid, Ef=Ef, two_theta_deg=two_theta_max)

    outdir_raw = str(params.get("outdir", ".")).strip()
    outdir_path = Path(outdir_raw).expanduser()
    if not outdir_path.is_absolute():
        outdir_path = (PROJECT_ROOT / outdir_path).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    # ---- (H,H,0) ----
    H = np.linspace(H_min, H_max, n_H)
    h1, k1, l1 = H, H, np.zeros_like(H)
    E1 = magnon_energy_meV(h1, k1, l1, S=S, J1=J1, J2=J2)
    Q1 = q_to_Qmag(h1, k1, l1, Q_110, Q_1m10, Q_001)

    fig, ax = plt.subplots()
    plot_path(
        ax,
        H,
        "(H,H,0)",
        Q1,
        E1,
        E_grid,
        Q_low,
        Q_high,
        title="Dispersion along (H,H,0)",
    )
    fig.tight_layout()
    fig.savefig(outdir_path / "QE_HH0.png", dpi=200)

    # ---- (1/3+K, 1/3-K, 0) ----
    K = np.linspace(K_min, K_max, n_K)
    h2 = (1.0 / 3.0) + K
    k2 = (1.0 / 3.0) - K
    l2 = np.zeros_like(K)
    E2 = magnon_energy_meV(h2, k2, l2, S=S, J1=J1, J2=J2)
    Q2 = q_to_Qmag(h2, k2, l2, Q_110, Q_1m10, Q_001)

    fig, ax = plt.subplots()
    plot_path(
        ax,
        K,
        "(1/3+K, 1/3−K, 0)",
        Q2,
        E2,
        E_grid,
        Q_low,
        Q_high,
        title="Dispersion along (1/3+K, 1/3−K, 0)",
    )
    fig.tight_layout()
    fig.savefig(outdir_path / "QE_Kpath.png", dpi=200)

    # ---- (1/3,1/3,L) ----
    L = np.linspace(L_min, L_max, n_L)
    h3 = np.full_like(L, 1.0 / 3.0)
    k3 = np.full_like(L, 1.0 / 3.0)
    l3 = L
    E3 = magnon_energy_meV(h3, k3, l3, S=S, J1=J1, J2=J2)
    Q3 = q_to_Qmag(h3, k3, l3, Q_110, Q_1m10, Q_001)

    fig, ax = plt.subplots()
    plot_path(
        ax,
        L,
        "(1/3, 1/3, L)",
        Q3,
        E3,
        E_grid,
        Q_low,
        Q_high,
        title="Dispersion along (1/3, 1/3, L)",
    )
    fig.tight_layout()
    fig.savefig(outdir_path / "QE_Lpath.png", dpi=200)

    # Optional: show on screen
    if int(params.get("show", 1)) == 1:
        plt.show()

    print(f"Saved figures to: {outdir_path}")


if __name__ == "__main__":
    main()
