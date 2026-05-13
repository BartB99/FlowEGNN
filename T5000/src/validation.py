import numpy as np

try:
    from pycorr import TwoPointCorrelationFunction
except:
    print("2PCF cannot be loaded")

def get_tpcf(pos, boxsize, r_bins=None, mu_bins=None):
    
    if r_bins is None and mu_bins is None:
        r_bins = np.linspace(0.5, 150.0, 25) # Defines 25 linearly spaced bins from 0.5 to 150 for the radial separation r
        mu_bins = np.linspace(-1, 1, 201)

    return TwoPointCorrelationFunction(
                "smu",
                edges=(np.array(r_bins), np.array(mu_bins)),
                data_positions1=pos.T,
                engine="corrfunc",
                n_threads=2,
                boxsize=boxsize,
                los="z",
    )(ells=[0])[0]