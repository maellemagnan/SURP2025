import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve

from astropy.nddata import StdDevUncertainty
from ppxf.ppxf import ppxf
from ppxf.ppxf_util import log_rebin, emission_lines
import ppxf.sps_util as lib
from time import perf_counter as clock

def load_spectrum(csv_path):
    """Read in a CSV with columns 'waveem' and 'flux'."""
    return pd.read_csv(csv_path)

def process_spectrum(df, lam_min=4000, lam_max=7000):
    """Trim to [lam_min, lam_max], sort ascending, normalize to median."""
    m = df.waveem.between(lam_min, lam_max)
    df2 = df.loc[m].sort_values('waveem').reset_index(drop=True)
    df2['flux'] /= np.median(df2['flux'])
    return df2

def rebin_to_log(data, velscale=None):
    """
    Log-rebin spectrum.
    Returns rebinned flux, log_lam array, and computed velscale.
    """
    flux_rebinned, ln_wave, velscale = log_rebin(data['waveem'], data['flux'])
    lam = np.exp(ln_wave)
    data = pd.DataFrame({'lam' : lam, 'ln_wave' : ln_wave, 'flux' : flux_rebinned})
    return data, lam, flux_rebinned, ln_wave

def make_noise(data, gain=1.2, readnoise=5):
    e = data['flux'] * gain
    noise = np.sqrt(np.clip(e, 0, None) + readnoise**2) / gain
    data['noise'] = noise
    return noise


def calculate_velscale_fwhm(ln_wave, lam):
    """
    Calculate the velocity scale using the ln wavelengths from log rebinning.
    """
    c = 299792.458 # speed of light in km/s
    d_ln_lam = (ln_wave[-1] - ln_wave[0])/(ln_wave.size - 1) 
    velscale = c * d_ln_lam   
    
    R = 1092
    fwhm = lam / R
    return velscale, fwhm 

def plot_ppxf(data, pp):
    """
    Overlay data, total fit, stellar and gas fits, and individual lines.
    """
    galaxy = data['flux']

    # Plot setup
    plt.figure(figsize=(10, 5))
    plt.plot(lam, galaxy, 'k', label='Data')
    plt.plot(lam, pp.bestfit, 'r', label='Total Fit')
    plt.plot(lam, pp.bestfit - pp.gas_bestfit, 'orange', label='Stellar Fit')
    plt.plot(lam, pp.gas_bestfit, 'magenta', label='Gas Fit')

    # Zoom in
    plt.xlim(6500, 7000)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(True)
    plt.title("Zoom on H-alpha and Nearby Emission Lines")
    plt.show()
    
def run_ppxf(lam, fwhm, velscale, data, noise, redshift=0.041185744):
    """
    Run pPXF with improved noise, masking, and polynomial handling.
    """
    from ppxf.ppxf import ppxf
    from ppxf.ppxf_util import emission_lines
    import ppxf.sps_util as lib

    c = 299792.458
    lam_range_gal = np.array([lam[0], lam[-1]])/ (1 + redshift)

    # --- Build templates
    sps_name = "emiles"
    ppxf_dir = Path(lib.__file__).parent
    filename = ppxf_dir / "sps_models" / f"spectra_{sps_name}_9.0.npz"
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + filename.name
        urlretrieve(url, filename)
    
    
    fwhm_gal_dic = {"lam": lam, "fwhm": fwhm}
    sps = lib.sps_lib(filename, velscale, fwhm_gal_dic)

    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
    n_stellar = stars_templates.shape[1]

    gas_templates, gas_names, gas_wave = emission_lines(
        sps.ln_lam_temp, lam_range_gal, fwhm_gal_dic)
    # --- Gas templates (include all lines around Hα)
    mask = np.isin(
        gas_names,
        ["Halpha", "[NII]6548_d", "[NII]6583_d", "[SII]6716", "[SII]6731"]
    )
    gas_templates, gas_names = gas_templates[:, mask], gas_names[mask]
    templates = np.hstack([stars_templates, gas_templates])

    # --- Components: stars (0), gas (1)
    component = np.array([0]*n_stellar + [1]*len(gas_names))

    # Moments: stars = 2 (v, σ), gas = 2 (v, σ)
    moments = [2, 2]

    # --- Build better noise
    flux_arr = data["flux"].values   # numpy array
    noise = np.full_like(data['flux'], np.std(data['flux']))
    galaxy   = flux_arr

    # --- Initial guesses
    vel0 = c * np.log(1 + redshift)
    start = [[vel0, 120],   # stars
         [vel0, 50]] 


    # --- Run pPXF
    pp = ppxf(
        templates, galaxy, noise, velscale, start,
        moments=moments, degree=4, mdegree=10,   # smaller poly
        lam=lam, component=component,
        gas_component=component > 0, gas_names=gas_names,
        lam_temp=sps.lam_temp
    )

    return pp

    