"""
Generate a PDF with Boissier rotation curve fits for all SPARC galaxies.
Each page shows one galaxy with a status note indicating if it passed the cuts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from warnings import catch_warnings, simplefilter

# Paths
DATADIR = "../data/"
OUTDIR = "../outputs/rotcurves/"
ROTCURVE_DIR = os.path.join(DATADIR, "SPARC_rotcurves_mod/")

# Load SPARC catalog
sparc = pd.read_csv(os.path.join(DATADIR, 'SPARC_Lelli2016c.mrt.csv'))
sparc_idx = sparc.drop_duplicates('Name').set_index('Name')

# ============================================================================
# Model and fitting functions (from notebook)
# ============================================================================

def boissier(r, wc, rv):
    """Boissier rotation curve model."""
    return wc * (1 - np.exp(-r / rv))


def _clean_rc(df):
    """Return sorted, cleaned (r, v, ev) numpy arrays."""
    r = pd.to_numeric(df['Rad[kpc]'], errors='coerce').to_numpy()
    v = pd.to_numeric(df['Vobs[km/s]'], errors='coerce').to_numpy()
    ev = pd.to_numeric(df['errV[km/s]'], errors='coerce').to_numpy()

    # finite, non-negative radii; finite velocities and errors
    m = np.isfinite(r) & np.isfinite(v) & np.isfinite(ev) & (r >= 0)
    r, v, ev = r[m], v[m], ev[m]
    if r.size:
        order = np.argsort(r)
        r, v, ev = r[order], v[order], ev[order]
    # avoid zero/near-zero uncertainties
    ev = np.where(ev <= 0, np.nan, ev)
    return r, v, ev


def _initial_guesses(r, v):
    """Data-driven initial guesses."""
    if len(r) == 0 or len(v) == 0:
        return 150.0, 1.0
    vf0 = float(np.nanpercentile(v, 90)) if np.isfinite(np.nanpercentile(v, 90)) else 150.0
    R = float(np.nanmax(r)) if np.isfinite(np.nanmax(r)) and np.nanmax(r) > 0 else 1.0
    rv0 = max(0.2, 0.4 * R)
    return vf0, rv0


def _fit_model(model_fn, r, v, ev, p0, bounds):
    """Try weighted fit first; on failure retry unweighted."""
    use_weight = np.isfinite(ev).sum() >= max(3, len(ev) // 3)
    try_p0 = p0
    for attempt in range(3):
        try:
            with catch_warnings():
                simplefilter("ignore")
                if use_weight:
                    popt, pcov = curve_fit(model_fn, r, v, sigma=ev, p0=try_p0,
                                           bounds=bounds, absolute_sigma=True, maxfev=200000)
                else:
                    popt, pcov = curve_fit(model_fn, r, v, p0=try_p0,
                                           bounds=bounds, maxfev=200000)
            perr = np.sqrt(np.diag(pcov))
            if not np.all(np.isfinite(perr)):
                raise RuntimeError("non-finite covariance")
            return popt, perr
        except Exception:
            use_weight = False
            try_p0 = _initial_guesses(r, v)
    return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])


# ============================================================================
# Status determination logic
# ============================================================================

# Special cases: galaxies with Inc between 79-80 that look too edge-on
SPECIAL_EDGE_ON = {"ESO079-G014", "NGC3917"}

# Inclination bounds
INC_MIN = 40
INC_MAX = 80

# Fit quality thresholds
DELTA_V_THRESHOLD = 25  # km/s
RV_BOISSIER_THRESHOLD = 4  # kpc
VROT_THRESHOLD = 150  # km/s
VFLAT_HIGH_THRESHOLD = 200  # km/s
RV_HIGH_VFLAT_THRESHOLD = 2  # kpc


def determine_status(galaxy, inc, vflat_sparc, vrot_fit, rv_fit, r, v):
    """
    Determine the status of a galaxy and return (is_ok, reason_string).
    """
    reasons = []

    # Check if we have enough data
    if len(r) < 5 or len(v) < 5:
        return False, "NOT OK (insufficient data points < 5)"

    # Check inclination
    if not np.isfinite(inc):
        return False, "NOT OK (Inc = NaN)"

    if inc <= INC_MIN:
        return False, f"NOT OK (Inc = {inc:.1f}° <= {INC_MIN}°)"

    if inc >= INC_MAX:
        return False, f"NOT OK (Inc = {inc:.1f}° >= {INC_MAX}°)"
    
    # Check for special edge-on cases
    if galaxy in SPECIAL_EDGE_ON and inc <= INC_MAX:
        return False, f"NOT OK (Inc = {inc:.1f}° <= {INC_MAX}°, special reason: still edge-on)"

    # Check fit validity
    if not np.isfinite(rv_fit) or rv_fit <= 0:
        return False, f"NOT OK (Rv = {'inf' if np.isinf(rv_fit) else 'NaN' if np.isnan(rv_fit) else rv_fit:.2f})"

    if not np.isfinite(vrot_fit) or vrot_fit <= 0:
        return False, f"NOT OK (Vrot_fit = {'inf' if np.isinf(vrot_fit) else 'NaN'})"

    # Check SPARC Vflat validity for comparison
    if np.isfinite(vflat_sparc) and vflat_sparc > 0:
        delta_v = abs(vrot_fit - vflat_sparc)

        # Check fit residual
        if delta_v > DELTA_V_THRESHOLD:
            return False, f"NOT OK (|Vflat_fit - Vflat_SPARC| = {delta_v:.1f} km/s > {DELTA_V_THRESHOLD})"

        # Check high Rv + high vrot (Boissier specific)
        if rv_fit > RV_BOISSIER_THRESHOLD and vrot_fit > VROT_THRESHOLD:
            return False, f"NOT OK (Rv = {rv_fit:.2f} kpc > {RV_BOISSIER_THRESHOLD} & Vrot = {vrot_fit:.1f} > {VROT_THRESHOLD})"

        # Check high Vflat + high Rv
        if vflat_sparc > VFLAT_HIGH_THRESHOLD and rv_fit > RV_HIGH_VFLAT_THRESHOLD:
            return False, f"NOT OK (Vflat = {vflat_sparc:.1f} > {VFLAT_HIGH_THRESHOLD} & Rv = {rv_fit:.2f} > {RV_HIGH_VFLAT_THRESHOLD})"

    return True, "OK"


# ============================================================================
# Main PDF generation
# ============================================================================

def create_pdf():
    """Create the PDF with all galaxy fits."""

    # Get all rotation curve files
    rot_files = sorted([f for f in os.listdir(ROTCURVE_DIR) if f.endswith('.dat')])

    pdf_path = os.path.join(OUTDIR, "SPARC_boissier_fits_all_galaxies.pdf")

    with PdfPages(pdf_path) as pdf:
        for i, filename in enumerate(rot_files):
            galaxy = os.path.splitext(filename)[0].split('_')[0]
            filepath = os.path.join(ROTCURVE_DIR, filename)

            print(f"Processing {i+1}/{len(rot_files)}: {galaxy}")

            # Load rotation curve data
            try:
                df_rc = pd.read_table(filepath, sep="\t", usecols=["Rad[kpc]", "Vobs[km/s]", "errV[km/s]"])
                r, v, v_e = _clean_rc(df_rc)
            except Exception as e:
                print(f"  Error loading data: {e}")
                r, v, v_e = np.array([]), np.array([]), np.array([])

            # Get SPARC catalog info
            if galaxy in sparc_idx.index:
                inc = float(sparc_idx.at[galaxy, 'Inc'])
                vflat_sparc = float(sparc_idx.at[galaxy, 'Vflat'])
                e_vflat_sparc = float(sparc_idx.at[galaxy, 'e_Vflat'])
            else:
                inc = np.nan
                vflat_sparc = np.nan
                e_vflat_sparc = np.nan

            # Fit Boissier model
            if len(r) >= 5:
                p0 = _initial_guesses(r, v)
                bounds = ([1.0, 1e-3], [1000.0, 100.0])
                popt, perr = _fit_model(boissier, r, v, v_e, p0=p0, bounds=bounds)
                vrot_fit, rv_fit = popt
                vrot_fit_err, rv_fit_err = perr
            else:
                vrot_fit, rv_fit = np.nan, np.nan
                vrot_fit_err, rv_fit_err = np.nan, np.nan

            # Determine status
            is_ok, status_str = determine_status(galaxy, inc, vflat_sparc, vrot_fit, rv_fit, r, v)

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

            # Plot data
            if len(r) > 0:
                ax.scatter(r, v, s=50, c='black', label='Observations', zorder=3)
                # Plot error bars where valid
                valid_err = np.isfinite(v_e) & (v_e > 0)
                if np.any(valid_err):
                    ax.errorbar(r[valid_err], v[valid_err], yerr=v_e[valid_err],
                               fmt=' ', ecolor='grey', capsize=3, zorder=2, alpha=0.7)

            # Plot fit if valid
            if np.isfinite(rv_fit) and np.isfinite(vrot_fit) and len(r) > 0:
                r_model = np.linspace(0, r.max() * 1.1, 200)
                v_model = boissier(r_model, vrot_fit, rv_fit)
                ax.plot(r_model, v_model, 'b-', lw=2.5,
                       label=f'Boissier fit: $V_{{flat}}$={vrot_fit:.1f}±{vrot_fit_err:.1f} km/s, $R_v$={rv_fit:.2f}±{rv_fit_err:.2f} kpc',
                       zorder=4)

            # Plot SPARC Vflat reference line if valid
            if np.isfinite(vflat_sparc) and vflat_sparc > 0:
                ax.axhline(vflat_sparc, color='red', ls='--', lw=1.5, alpha=0.7,
                          label=f'SPARC $V_{{flat}}$ = {vflat_sparc:.1f}±{e_vflat_sparc:.1f} km/s')

            # Formatting
            ax.set_xlabel('Radius (kpc)', fontsize=14)
            ax.set_ylabel(r'$V_{rot}$ (km s$^{-1}$)', fontsize=14)
            ax.set_title(f'{galaxy} Rotation Curve', fontsize=16, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

            # Add status box
            status_color = 'green' if is_ok else 'red'
            props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=status_color, linewidth=2)

            # Add inclination info to the status text
            if np.isfinite(inc):
                info_text = f"Inc = {inc:.1f}°\n{status_str}"
            else:
                info_text = f"Inc = N/A\n{status_str}"

            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=props, fontweight='bold', color=status_color)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPDF saved to: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_pdf()
