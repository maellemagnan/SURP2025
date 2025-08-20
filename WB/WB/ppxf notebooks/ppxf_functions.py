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
    data['flux'] = data['flux'] / np.median(data['flux'])

    return data, lam, flux_rebinned, ln_wave

def make_noise(data):
    """
    Create a constant-noise array for pPXF.
    You can replace this by a wavelength-dependent noise if you like.
    """
    noise = np.full_like(data['flux'], np.std(data['flux']))
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


def run_ppxf(lam, fwhm, velscale, data, noise):
    """
    Download (if needed) and logarithmically rebin the stellar templates.
    Returns a 2D array (npix × n_templates).
    """
    lam_range_gal = np.array([np.min(lam), np.max(lam)])
    # ensure models are downloaded
    sps_name = 'emiles'
    ppxf_dir = Path(lib.__file__).parent

    basename = f"spectra_{sps_name}_9.0.npz"
    filename = ppxf_dir / 'sps_models' / basename
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)

    fwhm_gal_dic = {"lam": lam, "fwhm": fwhm}
    sps = lib.sps_lib(filename, velscale, fwhm_gal_dic)

    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
    n_stellar = stars_templates.shape[1]
    
    gas_templates, gas_names, gas_wave = \
    emission_lines(sps.ln_lam_temp, lam_range_gal, fwhm_gal_dic)
    mask = np.isin(gas_names, ["Halpha", "[NII]6583_d"])
    gas_templates = gas_templates[:, mask]
    gas_names = gas_names[mask]
    
    templates = np.hstack([stars_templates, gas_templates])
    
    galaxy = data['flux'] 
 
    vel0 = 0
    sol = [vel0, 200]
    
    component = [0]*n_stellar  # Single stellar kinematic component=0 for all templates
    component += [1]
    component += [2]
    component = np.array(component)
    
    n_comp = component.max()+1
    moments = [2]*n_comp
    
    start = [sol for j in range(len(moments))]
    
    degree=-1
    mdegree=10
    t = clock()
    pp = ppxf(templates, galaxy, noise, velscale, start, plot=False,
            moments=moments, degree=degree, mdegree=mdegree, 
            lam=lam, component=component, 
            gas_component=component > 0, gas_names=gas_names,
            lam_temp=sps.lam_temp)
    print(f"pPXF fit done in {clock()-t:.2f} s")
    
    return pp, gas_templates


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
    plt.xlim(6500, 6800)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(True)
    plt.title("Zoom on H-alpha and Nearby Emission Lines")
    plt.show()
