from ast import parse
import numpy as np
import pandas as pd
import json

def Missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total,percent], axis=1, keys=['Total', 'Pourcentage'])
    print (missing_data[(percent>0)],'\n' )
    

    
def fft_denoiser(x, n_components, to_real=True):
    """Fast fourier transform denoiser.
    
    Denoises data using the fast fourier transform.
    
    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)
    """
    n = len(x)
    
    # compute the fft
    fft = np.fft.fft(x, n)
    PSD = fft * np.conj(fft) / n
    
    _mask = PSD > n_components
    fft = _mask * fft
    
    clean_data = np.fft.ifft(fft)
    
    if to_real:
        clean_data = clean_data.real
    
    return clean_data

def anomaly_to_pd(file):
    anom_pd = pd.read_csv(file, names=["timestamp","value"], header=None, parse_dates=["timestamp"])
    return anom_pd

def read_anomaly_file(file):
    d = {"timestamp": [], "value":[]}
    with open(file,'r') as f:
        for line in f:
            res = line.rstrip('\n')
            time,val = res.split(',')
            d["timestamp"].append(time)
            d["value"].append(val)
    return d

def init_data(data, date_limit, time_col):
    warmup = data[data[time_col] <= date_limit]
    learning = data[data[time_col] > date_limit]
    return warmup, learning