# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pycpx import CPlexModel

epochs_per_day = 48
packet_per_node = 1
Cs, Ce = 12e-9, 15e-9
begin, days = 60, 60
# We assume the battery is large enough to accumulate energy
# for cap days with this profile
cap = 5
battery_cap = 0
rho = 128 * 8
Rs = 62500
nodes = 20
bin_num = 10
D = 50e-3
# The number of values that b can take.
mod_levels = 5

def energy_trace(start_day, days):
    """
    The first column contains the seconds since the start of this year
     and the second column holds the current (in mA!) produced by our solar cell. 
    The measurement interval is 30s and the max. 
    current of the cell should be 35mA in the Maximum Power Point according to the data sheet. 
    It was sometimes a little larger.
    """
    voltage, milli_amper = 2.3, 1e-3
    # One sample every 30 seconds
    sam_per_day = (24 * 60 * 60) / 30
    sam_per_epoch = int(sam_per_day / epochs_per_day)
    ''' 
    days: The number of the days we are interested in the energy profile
    start_day: The first day of the range we are interested in since July 20
    -----------------------------------------------------------------------
    50 <= start_day <= 173, 50 <= days + start_day <= last_day
    -----------------------------------------------------------------------
    The first column contains the seconds since the start of this year
    '''
    f = ((np.loadtxt('tuhh_sun.dat')).T)
    #f = ((np.asarray([row.strip().split(' ') for row in file])).T)[1]
    
    # zero_day is the rank of the first sample in terms of the number in year
    zero_day = int(f[0,0] / (60 * 60 * 24))
    # The last day of the samples = 173
    # endDay = int(f[0, -1] / (60 * 60 * 24))
    f = f[1]
    first = (start_day - zero_day) * sam_per_day + 1
    last = first + days * sam_per_day
    rng = np.arange(first, last + 1 * sam_per_epoch, sam_per_epoch)
    average_energy = ((((np.add.reduceat(f, rng))[:-1]) *\
    (voltage * milli_amper)).reshape(days,epochs_per_day))
    print "avg_en is:"
    np.savetxt("hrv_profile", average_energy.mean(axis = 0))
    lower, upper = average_energy.min(), average_energy.max()
    energy_hist, bins, _ = plt.hist(average_energy, normed = 1,\
    range = (lower, upper))
    plt.legend()
    return energy_hist / np.sum(energy_hist[0]), bins[:-1]
    

# Input is an array of energy random variable, of type rv_dtype
def objective_function(rem_rv_en, rem_rv_prob):
    rv_array = np.array([stats.rv_discrete(\
    values = (e, p)).mean() for e, p in zip(rem_rv_en, rem_rv_prob)])
    return np.mean(rv_array)
    
# Input is a rv_dtype array of remaining energy at the end of epoch j in n nodes    
def en_neutral_const(rem_rv_en, rem_rv_prob, e_target):
    # rv.sf = 1 - CDF = The probability of being greater than e_target
    neutral_prob = np.array([stats.rv_discrete(\
    values = (e, p)).sf(e_target) for e, p in zip(rem_rv_en, rem_rv_prob)])
    return neutral_prob

# Input is a rv_dtype array of remaining energy at the end of epoch j in n nodes
def underflow_const(rem_rv_en):
    return [row.min() for row in rem_rv_en].min()
    
def next_battery_level(bat_rv_en, bat_rv_prob, hrv_rv_en, hrv_rv_prob):
    print bat_rv_en.shape, hrv_rv_en.shape
    print bat_rv_prob.shape, hrv_rv_prob.shape
    en = np.clip((np.add.outer(bat_rv_en, hrv_rv_en)).flatten(), -1, battery_cap)
    prob = (np.multiply.outer(bat_rv_prob, hrv_rv_prob)).flatten()
    print "----------------------en-------------------"
    print en.dtype
    print prob
    prob, en = np.histogram(en, bins = bin_num, weights = prob, density = True)
    en = np.lib.pad(en, (0, bin_num - en.size), 'constant', constant_values = (0))
    prob = np.lib.pad(prob[:-1], (0, bin_num - en.size), 'constant', constant_values = (0))
    return en, prob
    
def opt(e_init, e_target, hrv_hist, hrv_en):
    # verbosity is how much log is reported back from CPlex. 3 is the most verbose
    verbosity = 3
    m = CPlexModel(verbosity)
    b = m.new((epochs_per_day, nodes, mod_levels), vtype = int, lb = 0, ub = 1, name = 'b')
    l = m.new((epochs_per_day, nodes, bin_num), vtype = float, lb = -1, ub = battery_cap, name = 'l')
    fixed_prob = np.linespace(0, 1, num = bin_num, endpoint = True, dtype = float)    
    e_init_hist = np.zeros(e_init.shape, dtype = float)
    e_init_hist[:, 0] = 1
    hist_rv = np.zeros((epochs_per_day, nodes, bin_num), dtype = float)
    hist_rv[0] = e_init_hist
    # prepare the energy vector here
    en_rv = np.zeros((epochs_per_day, nodes, bin_num), dtype = float) 
    en_rv[0] = e_init
    for i in xrange(1, epochs_per_day):
        en_rv[i], hist_rv[i] = next_battery_level(en_rv[i - 1], hist_rv[i - 1],\
        hrv_en[i, :, :] - (np.vectorize(energy))(b[i, :]), hrv_hist[i, :, :])
        m.constrain(en_rv[i] >= 0)
        m.constrain(sum(np.vectorize(time)(b[i, :])) <= D)
    m.maximize(objective_function(en_rv[-1], hist_rv[-1]))
    return m
    
# non-vectorized
def energy(modulation_level):
    levels = 2 * modulation_level
    print type(rho / levels)
    print type(Cs * (2 * levels - 1) + Ce).shape
    # en = (rho / levels) * (Cs * (2 ** levels - 1) + Ce) ####Wroooong
    return  (Cs * (2 * levels - 1) + Ce)
    
# non-vectorized
def time(modulation_level):
    b = 2 * modulation_level
    return rho / (Rs * b)
    
def randomize_energy(hist, en):
    return hist, en
    
def main():
    hist, en = energy_trace(begin, days)
#    hrv_hist = np.zeros((epochs_per_day, nodes, hist.shape[-1]), dtype = float)
#    hrv_en = hrv_hist
#    for i in xrange(nodes):
#        hrv_hist[:, i, :], hrv_en[:, i, :] = randomize_energy(hist, en)
#    battery_cap = cap #* int(sum(hrv_hist.dot(hrv_en)))
#    e_init = np.zeros((nodes, bin_num), dtype = float)
#    e_init[:, 0] = battery_cap / 3
#    opt(e_init, e_init, hrv_hist, hrv_en)

if __name__ == '__main__':
    main()
