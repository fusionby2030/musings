""" 
author: Adam Kit 
email: adam.kit@helsinki.fi 

Each function is headed by the libraries required for it
"""
from typing import Dict, List, Tuple
import numpy as np 

from scipy.fft import  fft, fftfreq
from scipy import signal as sig
def get_elm_timings_from_pulse_dict(pulse_dict: Dict[str, Dict[str, np.ndarray]], param='edg8/tbeo') -> Tuple[List[Tuple[float, float]], List[np.ndarray], Tuple[float, float]]: 
    """ 
    Calculates the elm timings from a pulse dict
    returns 
    1. window steady states 
    2. elm timings 
    3. the start and end of hmode 
    """
    t_window_steady_states, (t_start_hmode, t_end_hmode), (lh_domain, ratio_threshold_total_power) = find_steady_state_windows(pulse_dict)
    elm_param_signals = ['edg8/tbeo', 'edg8/tbei']
    elm_timings = []
    time, data = pulse_dict[param]['time'], pulse_dict[param]['data']
    for n, (t_beg, t_end) in enumerate(t_window_steady_states): 
        _window_bool = np.logical_and(time >= t_beg, time <= t_end)
        _time, _data = time[_window_bool], data[_window_bool]
        N = len(_time)
        T = np.diff(_time).mean() 
        _mean, _std = _data.mean(), _data.std()
        # To get rid of the close to 0 frequency in the fft, we subtract the mean and divide by std. 
        tbeo_dat_normed = (_data - _mean)/_std
        yf = fft(tbeo_dat_normed)
        xf = fftfreq(N, T)[:N//2]

        likely_freq = xf[np.argmax(2.0/N*np.abs(yf[0:N//2]))]
        likely_freq_time = 1.0/ likely_freq
        num_samples_patience = int(likely_freq_time / T)
        distance_in_indicie_space = num_samples_patience - (num_samples_patience // 3) # TODO: Free parameter here! 

        peaks, _ = sig.find_peaks(_data,  height=_mean+_std, distance=distance_in_indicie_space) 
        window_elm_timings = _time[peaks]
        elm_timings.append(window_elm_timings)
    return t_window_steady_states, elm_timings, (t_start_hmode, t_end_hmode)

from scipy.interpolate import interp1d 
def p_lh_threshold_martin_scaling_from_pulse_dict(pulse_dict: Dict[str, Dict[str, np.ndarray]]) -> List[np.ndarray]: 
    """ Using martin scaling we find the ratio of the PTOT/PLH 
    This is done by first interpolating all diagnostics used onto a common time domain 
    The common time domain is given by the lowest order sampled parameter used in the martin scaling 

    The power variables are then interpolated to the same axis 

    returns the time domain for the lh threshold, plh [MW], the ratio of ptot/plh
    """
    power_scaling = lambda n, r, a, b : (2.14)*(np.exp(0.107))*(np.power(n*1e-20,0.728))*(np.power(abs(b),0.772))*(np.power(a,0.975))*(np.power(r,0.999))
    lh_threshold_diagnostic_signal_names = ['kg1l/lad3', 'efit/Rgeo', 'efit/ahor', 'magn/BTF']
    # Finds the lowest sampled signal so we can interpolate onto that domain
    data_size_list = [(diag_sig, signal_dims['data'].shape) for diag_sig, signal_dims in pulse_dict.items() if diag_sig in lh_threshold_diagnostic_signal_names]
    domain_diagnostic, _ = min(data_size_list, key=lambda t: t[1]) # https://docs.python.org/3.8/library/functions.html#min
    diagnostics_to_interpolate = [d for d in lh_threshold_diagnostic_signal_names if d != domain_diagnostic]
    time_domain = pulse_dict[domain_diagnostic]['time'].copy()

    interpolated_dict = {domain_diagnostic: pulse_dict[domain_diagnostic]['data']}

    for key in diagnostics_to_interpolate: 
        _interp = interp1d(pulse_dict[key]['time'], pulse_dict[key]['data'])
        interp_data = _interp(time_domain)
        interpolated_dict[key] = interp_data

    plh_threshold = 1e6*power_scaling(*[interpolated_dict[key] for key in lh_threshold_diagnostic_signal_names])
    # Now to compare against the input power 
    power_diagnostic_signal_names = ['power/PNBI_TOT', 'power/P_OH', 'power/PICR_TOT', 'nbip/shi']
    total_power = np.zeros_like(time_domain)
    for key in power_diagnostic_signal_names: 
        try: 
            _interp = interp1d(pulse_dict[key]['time'], pulse_dict[key]['data'], bounds_error=False, fill_value=0.0)
            power_interped = _interp(time_domain)
        except ValueError: 
            power_interped = np.zeros_like(time_domain)
        except KeyError as e: 
            if e.args[0] == 'nbip/shi':
                power_interped = np.zeros_like(time_domain)
            else: 
                raise KeyError('pulse_dict does not contain the following necessary dignostics diagnostics: {}'.format(power_diagnostic_signal_names + lh_threshold_diagnostic_signal_names))
        if key in ['nbip/sh']: 
            total_power -= power_interped
        else:     
            total_power += power_interped
    ratio_threshold_total_power = total_power / plh_threshold
    return [time_domain, plh_threshold, ratio_threshold_total_power]

import ruptures as rpt 
def find_steady_state_windows(pulse_dict: Dict[str, Dict[str, np.ndarray]]) -> Tuple[List[Tuple[float, float]], Tuple[float, float], Tuple[np.ndarray, np.ndarray]]: 
    """ Uses ruptures to find steady state windows using the PLH threshold and the diamagnetic energy """
    if not 'ehtr/wdia' in pulse_dict.keys(): raise KeyError('must include ehtr/wdia in list of diagnostics to pull')
    # FREE PARAMETERS 
    MIN_TIME_WINDOW_SIZE: float = 0.5
    MAX_DEVIATION_OF_MEAN_FROM_SLOP = 0.15 # Percent deviation for a window to be marked as a steady state
    lh_domain, _, ratio_threshold_total_power = p_lh_threshold_martin_scaling_from_pulse_dict(pulse_dict)
    h_mode_window = np.logical_and(ratio_threshold_total_power > 1.0, np.logical_and(lh_domain > 44, lh_domain < 60)) # TODO: This is not scalable for AUG
    t_start_hmode, t_end_hmode = lh_domain[h_mode_window][0], lh_domain[h_mode_window][-1]

    time, data = pulse_dict['ehtr/wdia']['time'], pulse_dict['ehtr/wdia']['data']
    dt = (time[-1] - time[0])/len(time)
    dt = np.diff(time).mean()
    window_size = int(MIN_TIME_WINDOW_SIZE / dt)
    if window_size < 100: 
        print(f'window_size too small {window_size}, the estimated sampling frequency is {dt:.5}')
        window_size = 1000 
    algo = rpt.Pelt(model='l2', min_size=window_size).fit(data)
    pen = np.log(len(time)) # This is tehcnically also a free parameter 
    my_bkps = algo.predict(pen=pen)
    t_bkps = np.array([time[bkp] for bkp in my_bkps if bkp != len(time)])
    t_bkps_in_hmode = t_bkps[np.logical_and(t_bkps > t_start_hmode, t_bkps <= t_end_hmode)]

    t_window_steady_states = []
    for t_idx, t in enumerate(t_bkps_in_hmode): 
        if t_idx == 0: 
            t_beg, t_end = t_start_hmode, t
        elif t_idx == len(t_bkps_in_hmode): 
            t_beg, t_end = t, t_end_hmode
        else: 
            t_beg, t_end = t_bkps_in_hmode[t_idx-1], t
        _window_bool = np.logical_and(time >= t_beg, time <= t_end)
        slope = (data[_window_bool][-1] - data[_window_bool][0]) / (time[_window_bool][-1] - time[_window_bool][0])
        if abs(slope / data[_window_bool].mean()) < MAX_DEVIATION_OF_MEAN_FROM_SLOP: 
            t_window_steady_states.append((t_beg, t_end))
    return t_window_steady_states, (t_start_hmode, t_end_hmode), (lh_domain, ratio_threshold_total_power)