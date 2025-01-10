import os
import emcee
import corner
import warnings
import wfdb as wf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm, gamma, beta, invgamma
from scipy.signal import butter, filtfilt, resample, find_peaks

from multiprocessing import Pool, set_start_method
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

EPS_HBO2_660 = 319.6
EPS_HB_660 = 3226.56

EPS_HBO2_940 = 1214
EPS_HB_940 = 693.44

encounter_id = "c5dd95c1ac9fc618cab2e940096089c6a91be58206fa6fc6a1375c69c4922779"

f_spo2 = 2

start = 5 * 60

saturation = pd.read_csv(f'data/waveforms/{encounter_id[0]}/{encounter_id}_2hz.csv')
spo2 = saturation['dev59_SpO2'].to_numpy()[start * f_spo2:]
t_spo2 = np.arange(spo2.shape[0]) / (60 * f_spo2)

f_ppg = 86

start = (5 - 2.8) * 60

ppg, ppg_info = wf.rdsamp(f'data/waveforms/{encounter_id[0]}/{encounter_id}_ppg')
ppg = ppg[int(start * f_ppg):]

ir = ppg[:, 0]
red = ppg[:, 1]
t_ppg = np.arange(len(red)) / (60 * f_ppg)

def extract_beats(ppg, f_ppg, min_time_between=0.4):
    """
    Arguments:
    ----------
    ppg: np.ndarray
        A one dimensional time series of ppg data (red or ir).
    f_ppg: int
        The sampling frequency (Hz).
    min_time_between: float
        The minimal time between two heartbeats.

    Returns:
    --------
    - peaks: np.ndarray
        The indices of the heartbeats peaks in the ppg time series.
    """
    min_number_between = int(min_time_between * f_ppg)
    peaks, _ = find_peaks(ppg, distance=min_number_between)
    return peaks

#choosing a random cycle
red_peaks = extract_beats(red, f_ppg)
random_cycle = np.random.RandomState(seed=24).randint(0, len(red_peaks) - 1)
start_random_cycle = red_peaks[random_cycle]
end_random_cycle = red_peaks[random_cycle + 1] - 1 # - 1 in order to stop right before the next cycle

red_random_cycle = red[start_random_cycle:end_random_cycle + 1]
ir_random_cycle = ir[start_random_cycle:end_random_cycle + 1]
t_random_cycle = np.arange(len(red_random_cycle)) / (60 * f_ppg)

#computing distribution
#in order to remove the DC component we will mutiply each cycle by its mean
red_AC = red.copy()
ir_AC = ir.copy()
red_hist = []
ir_hist = []

for i in range(len(red_peaks) - 1):
    
    start = red_peaks[i]
    end = red_peaks[i + 1]
    
    red_DC_cycle = np.mean(red[start:end])
    ir_DC_cycle = np.mean(ir[start:end])

    for j in range(start, end):
        red_AC[j] = red_AC[j] - red_DC_cycle
        ir_AC[j] = ir_AC[j] - ir_DC_cycle
        


#excluding values that are not in any cycle
red_AC = red_AC[red_peaks[0]:red_peaks[len(red_peaks) - 1]]
ir_AC = ir_AC[red_peaks[0]:red_peaks[len(red_peaks)- 1]]

red_ACs = []
red_DCs = []
ir_ACs = []
ir_DCs = []

for i in range(len(red_peaks) - 1):
    start = red_peaks[i]
    end = red_peaks[i + 1]
    
    DC = np.mean(red[start:end])
    red_DCs.append(DC)
    red_ACs.append(max(red[start:end]) - min(red[start:end]))
    
    DC = np.mean(ir[start:end])
    ir_DCs.append(DC)
    ir_ACs.append(max(ir[start:end]) - min(ir[start:end]))

R = []
for i in range(len(red_peaks) - 1):
    R.append((red_ACs[i] / red_DCs[i]) / (ir_ACs[i] / ir_DCs[i]))

R = resample(R, len(spo2))

R_subsampled = R[:len(R):200]
spo2_subsampled = spo2[:len(spo2):200]


def sample_likelihood(SpO2_1T, sigma_squared, eps_hbo2_660, eps_hbo2_940, eps_hb_660, eps_hb_940, n=1):
    if(SpO2_1T[0] > 1):
        SpO2_1T /= 100
    
    R = np.zeros((len(SpO2_1T), n))
    
    for i in range(len(SpO2_1T)):
        tmp = SpO2_1T[i]
        mean = (tmp * eps_hbo2_660 + (1 - tmp) * eps_hb_660)  / (tmp * eps_hbo2_940 + (1 - tmp) * eps_hb_940)
        for j in range(n):
            R[i][j] = norm.rvs(loc=mean, scale=np.sqrt(sigma_squared))
        
    return R

def log_likelihood(R, SpO2_1T, sigma_squared, eps_hbo2_660, eps_hbo2_940, eps_hb_660, eps_hb_940):
    if(SpO2_1T[0] > 1):
        SpO2_1T /= 100
    mean = (SpO2_1T * eps_hbo2_660 + (1 - SpO2_1T) * eps_hb_660)  / (SpO2_1T * eps_hbo2_940 + (1 - SpO2_1T) * eps_hb_940)
    
    sigma_squared = np.maximum(sigma_squared, 1e-10)
    
    return norm.logpdf(R, loc=mean, scale=np.sqrt(sigma_squared)).sum()

def bounded_beta(a, b, size, lower_bound=0, upper_bound=1):
    samples = []
    while len(samples) < size:
        value = beta.rvs(a, b)
        if lower_bound <= value <= upper_bound:
            samples.append(value)
    return np.array(samples)

def sample_prior(size_SpO2, n=1):
    SpO2 = bounded_beta(12, 2, size_SpO2, lower_bound=0.7, upper_bound=1)
    
    sigma_squared = abs(norm.rvs(loc=0.02, scale=0.004, size=n))
    eps_hbo2_660 = norm.rvs(loc=EPS_HBO2_660, scale=100, size=n)
    eps_hbo2_940 = norm.rvs(loc=EPS_HB_940, scale=100, size=n)
    eps_hb_660 = norm.rvs(loc=EPS_HB_660, scale=100, size=n)
    eps_hb_940 = norm.rvs(loc=EPS_HB_940, scale=100, size=n)
    
    return SpO2, sigma_squared, eps_hbo2_660, eps_hbo2_940, eps_hb_660, eps_hb_940

def log_prior(SpO2, sigma_squared, eps_hb_660, eps_hbo2_660, eps_hb_940, eps_hbo2_940, n=1):
    if(SpO2[0] > 1):
        SpO2 /= 100
    
    return (beta.logpdf(SpO2, a=12, b=2).sum() 
            + norm.logpdf(sigma_squared, loc=0.02, scale=0.004).sum() 
            + norm.logpdf(eps_hbo2_660, loc=EPS_HBO2_660, scale=100).sum() 
            + norm.logpdf(eps_hbo2_940, loc=EPS_HB_940, scale=100).sum() 
            + norm.logpdf(eps_hb_660, loc=EPS_HB_660, scale=100).sum() 
            + norm.logpdf(eps_hb_940, loc=EPS_HB_940, scale=100).sum())

def sample_joint(size_SpO2, n=1):
    SpO2_prior, sigma_squared_prior, eps_hbo2_660_prior, eps_hbo2_940_prior, eps_hb_660_prior, eps_hb_940_prior = sample_prior(size_SpO2, n)
    mean = (SpO2_prior * eps_hbo2_660_prior + (1 - SpO2_prior) * eps_hb_660_prior)  / (SpO2_prior * eps_hbo2_940_prior + (1 - SpO2_prior) * eps_hb_940_prior)
    R = norm.rvs(loc=mean, scale=np.sqrt(sigma_squared_prior))
    return R * SpO2_prior * sigma_squared_prior * eps_hbo2_660_prior * eps_hbo2_940_prior * eps_hb_660_prior * eps_hb_940_prior

def log_posterior(params, R, SpO2_sampled):
    SpO2 = params[:len(SpO2_sampled)]
    if(SpO2[0] > 1):
        SpO2 /= 100
    sigma_squared, eps_hbo2_660, eps_hbo2_940, eps_hb_660, eps_hb_940 = params[len(SpO2_sampled):]
    
    log_ll = log_likelihood(R, SpO2, sigma_squared, eps_hbo2_660, eps_hbo2_940, eps_hb_660, eps_hb_940)
    log_p = log_prior(SpO2, sigma_squared, eps_hbo2_660, eps_hbo2_940, eps_hb_660, eps_hb_940)
    
    return log_ll + log_p

n = 1
size_prior = len(R_subsampled)
SpO2_prior, sigma_squared_prior, eps_hbo2_660_prior, eps_hbo2_940_prior, eps_hb_660_prior, eps_hb_940_prior = sample_prior(len(spo2_subsampled), n=n)
mean = (SpO2_prior * eps_hbo2_660_prior + (1 - SpO2_prior) * eps_hb_660_prior)  / (SpO2_prior * eps_hbo2_940_prior + (1 - SpO2_prior) * eps_hb_940_prior)

# print(SpO2_prior, sigma_squared_prior, eps_hbo2_660_prior, eps_hbo2_940_prior, eps_hb_660_prior, eps_hb_940_prior)
# print(mean, np.sqrt(sigma_squared_prior))

R_prior = np.zeros(size_prior)
for i in range(size_prior):
    R_prior[i] = norm.rvs(loc=mean[i], scale=np.sqrt(sigma_squared_prior))

t_prior = np.linspace(0, 1, size_prior)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    set_start_method('spawn', force=True)
    with Pool() as pool:
        nwalkers = 150
        ndim = 30 + 5
        nsteps = 200000

        pos = np.zeros((nwalkers, ndim))

        for i in range(len(spo2_subsampled)):
            pos[:, i] = np.random.uniform(0.7, 1, nwalkers)

        interval = 100
        pos[:, len(spo2_subsampled)] = np.abs(0.02 + np.random.uniform(-0.01, 0.01, nwalkers))
        pos[:, len(spo2_subsampled) + 1] = np.random.uniform(EPS_HBO2_660-interval, EPS_HBO2_660+interval, nwalkers)
        pos[:, len(spo2_subsampled) + 2] = np.random.uniform(EPS_HBO2_940-interval, EPS_HBO2_940+interval, nwalkers)
        pos[:, len(spo2_subsampled) + 3] = np.random.uniform(EPS_HB_660-interval, EPS_HB_660+interval, nwalkers)
        pos[:, len(spo2_subsampled) + 4] = np.random.uniform(EPS_HB_940-interval, EPS_HB_940+interval, nwalkers)
        

        SpO2_generated, sigma_squared_generated, eps_hbo2_660_generated, eps_hbo2_940_generated, eps_hb_660_generated, eps_hb_940_generated = sample_prior(len(spo2_subsampled), n=1)
        SpO2_generated *= 100

        parameters = (SpO2_generated, sigma_squared_generated, eps_hbo2_660_generated, eps_hbo2_940_generated, eps_hb_660_generated, eps_hb_940_generated)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(R_subsampled, spo2_subsampled), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
        
    with open("sampler", "wb") as file:
        pickle.dump(sampler, file)
        print("chain created !")
