# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:50:46 2015

@author: maryam
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
# from scipy import stats

epochs_per_day = 48 
packet_per_node = 2 * 40
Cs, Ce = 12e-9, 15e-9
begin, days = 60, 60
# We assume the battery is large enough to accumulate energy
# for cap days with this profile
cap = 1
target = cap / 2.0
rho = 128 * 8 * packet_per_node
Rs = 62500
nodes = 8
bin_num = 10
D = 50e-3 * 38
init_bat = np.zeros(nodes)
# The number of values that b can take.
mod_levels = 5

# tmp1 = np.loadtxt("hrv_profile")
# hrv_profile = tmp1.reshape(nodes, epochs_per_day)
# rem_mean = np.zeros((epochs_per_day, nodes), dtype = float)

def plot_energy():
    mx = np.max(hrv_profile)
    for i in xrange(nodes):
        t = plt.plot(hrv_profile[i] / mx)
        plt.setp(t, linewidth = 2)
    plt.xlabel('Epoch Number')
    plt.ylabel('Harvested Energy')
    plt.title('Energy Harvesting Profile of the Nodes in the Cluster')
    plt.grid(True)
    plt.savefig('hrv_profile.pdf')
    plt.show()

def energy_trace(start_day, days):
    """
    The first column contains the seconds since the start of this year
     and the second column holds the current (in mA!) produced by our solar cell. 
    The measurement interval is 30s and the max. 
    current of the cell should be 35mA in the Maximum Power Point according to the data sheet. 
    It was sometimes a little larger.
    """
    voltage, milli_amper = 4.8, 40e-3
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
    f = ((np.loadtxt('../tuhh_sun.dat')).T)
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
    (voltage * milli_amper)).reshape(days, epochs_per_day))
    plt.plot(average_energy.mean(axis = 0))
    print "Summation of energy during the day is:" , average_energy.mean(axis = 0).sum()
    # np.savetxt("hrv_profile", average_energy.mean(axis = 0))
    np.savetxt("hrv_profile", average_energy[0 : nodes, ])
    # hrv_profile = average_energy[0 : nodes + 1, ]
   #  hrv_profile *= 0.2

def time_energy():
    tmp_t = np.array([ float(rho) / (Rs * mod) for mod in xrange(2, 1 + 2 * mod_levels, 2) ])
    tmp_en = np.array([ (float(rho) / mod) * (Cs * (2 ** mod - 1) + Ce)  for mod in xrange(2, 1 + 2 * mod_levels, 2 )])
    return (tmp_t, tmp_en)

t_time, en = time_energy()
    
def uniform():
    tmp = np.where(nodes * t_time <= D)[0]
    if tmp.size == 0:
        return 0
    com_e = np.full(nodes, en[tmp[0]])
    l = np.array([ x for x in init_bat ]) 
    for j in xrange(epochs_per_day):
        l = l + hrv_profile[:, j] - com_e
        l = np.minimum(cap, l)
        if np.min(l) < 0:
            return 0
    if np.min(l) < target:
        return 0
    return np.min(l)

def gradual0():
    com_e = np.full(nodes, en[0] )
    l = ma.array(np.array([ x for x in init_bat ]))
    for j in xrange(epochs_per_day):
        b_indx = np.full( nodes, np.int8(0) )
        l = l + hrv_profile[:, j] - com_e
        l = np.minimum(cap, l)
        l = ma.array(l)
        slack = D - nodes * t_time[0]
        k_coef = 0.6
        while (not l.mask.all()) and (slack < 0):
            max_indx = np.argmax(l * k_coef + (epochs_per_day - j) * (1 - k_coef) * rem_mean[j, ])
            oldb = b_indx[max_indx]
            slack += t_time[oldb] - t_time[oldb + 1]
            l[max_indx] += en[oldb] - en[oldb + 1]
            b_indx[max_indx] += 1
            if b_indx[max_indx] == mod_levels - 1:
                l[max_indx] = ma.masked
        l.mask = ma.nomask
        if np.min(l) < 0:
            return 0
    if np.min(l) < target:
        return 0
    return np.min(l)
    

def avarice():
    tmp = np.where(nodes * t_time <= D)[0]
    if tmp.size == 0:
        return 0
    tmp = min(mod_levels - 1, tmp[0] + 1)
    com_e = np.full( nodes, en[tmp] )
    l = ma.array(np.array([ x for x in init_bat ]))
    for j in xrange(epochs_per_day):
        b_indx = np.full( nodes, np.int8(tmp) )
        l = l + hrv_profile[:, j] - com_e
        l = np.minimum(cap, l)
        l = ma.masked_where(b_indx <= 0, l)
        slack = D - nodes * t_time[tmp]
        while not l.mask.all():
            min_indx = np.argmin(l)
            if slack + t_time[b_indx[min_indx]] - t_time[b_indx[min_indx] - 1] >= 0:
                slack += t_time[b_indx[min_indx]] - t_time[b_indx[min_indx] - 1]
                l[min_indx] += en[b_indx[min_indx]] - en[b_indx[min_indx] - 1]
                l[min_indx] = min(cap, l[min_indx])
                b_indx[min_indx] -= 1
                if b_indx[min_indx] == 0:
                    l[min_indx] = ma.masked
            else:
                l[min_indx] = ma.masked
        l.mask = ma.nomask
        if np.min(l) < 0:
            return 0
    if np.min(l) < target:
        return 0
    return np.min(l)
    
    
def avarice2():
    tmp = mod_levels - 1
    com_e = np.full( nodes, en[tmp] )
    l = ma.array(np.array([ x for x in init_bat ]))
    for j in xrange(epochs_per_day):
        b_indx = np.full( nodes, np.int8(tmp) )
        l = l + hrv_profile[:, j] - com_e
        l = np.minimum(cap, l)
        l = ma.masked_where(b_indx <= 0, l)
        slack = D - nodes * t_time[tmp]
        while not l.mask.all():
            min_indx = np.argmin(l)
            if slack + t_time[b_indx[min_indx]] - t_time[b_indx[min_indx] - 1] >= 0:
                slack += t_time[b_indx[min_indx]] - t_time[b_indx[min_indx] - 1]
                l[min_indx] += en[b_indx[min_indx]] - en[b_indx[min_indx] - 1]
                l[min_indx] = min(cap, l[min_indx])
                b_indx[min_indx] -= 1
                if b_indx[min_indx] == 0:
                    l[min_indx] = ma.masked
            else:
                l[min_indx] = ma.masked
        l.mask = ma.nomask
        if np.min(l) < 0:
            return 0
    if np.min(l) < target:
        return 0
    return np.min(l)

def gradual1():
    tmp = np.where(nodes * t_time <= D)[0]
    if tmp.size == 0:
        return 0
    tmp = max(0, tmp[0] - 1)
    com_e = np.full( nodes, en[tmp] )
    l = ma.array(np.array([ x for x in init_bat ]))
    for j in xrange(epochs_per_day):
        b_indx = np.full( nodes, np.int8(tmp) )
        l = l + hrv_profile[:, j] - com_e
        l = np.minimum(cap, l)
        l = ma.masked_where(b_indx >= mod_levels - 1, l)
        slack = D - nodes * t_time[tmp]
        while not l.mask.all() and slack < 0:
            max_indx = np.argmax(l)
            slack += t_time[b_indx[max_indx]] - t_time[b_indx[max_indx] + 1]
            l[max_indx] += en[b_indx[max_indx]] - en[b_indx[max_indx] + 1]
            l[max_indx] = min(cap, l[max_indx])
            b_indx[max_indx] += 1
            if b_indx[max_indx] == mod_levels - 1:
                l[max_indx] = ma.masked
        l.mask = ma.nomask
        if np.min(l) < 0:
            return 0
    if np.min(l) < target:
        return 0
    return np.min(l)    


def greedy():
    tmp = np.where(nodes * t_time <= D)[0]
    if tmp.size == 0:
        return 0
    com_e = np.full( nodes, en[tmp[0]] )
    l = ma.array(np.array([ x for x in init_bat ]))
    for j in xrange(epochs_per_day):
        b_indx = np.full( nodes, np.int8(tmp[0]) )
        l = l + hrv_profile[:, j] - com_e
        l = np.minimum(cap, l)
        l = ma.masked_where(b_indx <= 0, l)
        slack = D - nodes * t_time[tmp[0]]
        while not l.mask.all():
            min_indx = np.argmin(l)
            if slack + t_time[b_indx[min_indx]] - t_time[b_indx[min_indx] - 1] >= 0:
                slack += t_time[b_indx[min_indx]] - t_time[b_indx[min_indx] - 1]
                l[min_indx] += en[b_indx[min_indx]] - en[b_indx[min_indx] - 1]
                l[min_indx] = min(cap, l[min_indx])
                b_indx[min_indx] -= 1
                l[min_indx] = ma.masked
            else:
                l[min_indx] = ma.masked
        l.mask = ma.nomask
        if np.min(l) < 0:
            return 0
    if np.min(l) < target:
        return 0
    return np.min(l)


def visualization(f, lab, pattern):
    e = np.loadtxt(f)
    e_ma = ma.masked_equal(e, 0)
    # print np.count_nonzero(e[:, ])
    tmp = e_ma.mean(axis = 1)
    print f
    print tmp
    plt.plot(np.arange(23, 51, 3), tmp, linewidth = 2, marker = pattern, label = lab)
    return tmp


def main():

    # energy_trace(begin, days)
    # rs = np.random.random_sample( (1, 100 * nodes))
    # np.savetxt('../rand_values', rs)
    
    rs = np.loadtxt('../rand_values')
    global hrv_profile, init_bat, packet_per_node, rho, t_time, en
    hrv_profile *= 0.001
    simcount = 35
    uni_res = np.zeros((10, simcount))
    greedy_res = np.zeros((10, simcount))
    avarice_res = np.zeros((10, simcount))      
        
    for i in xrange(1, 11):
        rand_idx = 0
        packet_per_node = 2 * (20 + i * 3)
        rho = 128 * 8 * packet_per_node
        t_time, en = time_energy() 
        for j in xrange(simcount):
            for n in xrange(nodes):
                init_bat[n] = rs[rand_idx] * .5 * 5 / 10.0
                rand_idx += 1
            uni_res[i - 1, j] = uniform()
            greedy_res[i - 1, j] = greedy()
            avarice_res[i - 1, j] = avarice()
    np.savetxt('uniform6', uni_res)
    np.savetxt('avarice6', avarice_res)
    np.savetxt('greedy6', greedy_res) 
    
    s = "_" * 80 + "\n"
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(labelsize = 14)
    visualization('uniform6', 'Uniform', 'o')
    print s
    visualization('avarice6', 'Avarice', 'v')
    print s
    visualization('greedy6', 'Greedy', 'p')
    print s
    visualization('min_opt6', 'Optimal', 's')
    lgd = plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
           ncol = 2, mode = "expand", borderaxespad = 0., fontsize = 14)
    # plt.title('The effect of initial battery level, r$D = 50 ms$')
    plt.xlabel('Number of transmissions per epoch', size = 14)
    plt.ylabel('Normalized minimum remaining energy', size = 14)
    plt.grid(True)
    plt.savefig('exp26.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
if __name__ == '__main__':
    # main()
    energy_trace(begin, days)
   
