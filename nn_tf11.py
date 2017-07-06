#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:34:57 2016

@author: maryam
"""


import numpy as np
from scipy import stats
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
from random import shuffle
from random import seed

seed(313)

plt.ion()
n = 20
k = 4
period = 300
infcost = -1
normfact = float(n)

step = 500
h_max = 1.
hrv_step = h_max / 20
# hrv = [.1, .1, .1, .1]
hrv = np.arange(0., h_max, hrv_step)
B_max = 15
B_init = B_max / 2
sim_rounds = 1000
workload = .3
hrv_size = len(hrv)
simnum = 0
epsilon = 0.01
gamma = .001

hrv_pdf1 = np.zeros((hrv_size, hrv_size))
hrv_pdf2 = np.zeros((hrv_size, hrv_size))
for i in range(len(hrv)):
    hrv_pdf1[i][i] = .7
    hrv_pdf2[i][i] = .7

for i in range(1, len(hrv) - 1):
    hrv_pdf1[i][i + 1] = .2
    hrv_pdf1[i + 1][i] = .1
    hrv_pdf2[i][i + 1] = .2
    hrv_pdf2[i + 1][i] = .1
hrv_pdf1[len(hrv) - 1][len(hrv) - 2] = .3
hrv_pdf1[0][1] = .7
hrv_pdf2[len(hrv) - 1][len(hrv) - 2] = .3
hrv_pdf2[0][1] = .7
hrv_pdf1[0][0] = .3
hrv_pdf2[0][0] = .3
custm = [
    [stats.rv_discrete(values=(np.arange(len(hrv)), hrv_pdf1[i])) for \
     i in range(len(hrv))],
    [stats.rv_discrete(values=(np.arange(len(hrv)), hrv_pdf2[i])) for \
     i in range(len(hrv))]
    ]


def next_hrv_cont(i, hrv_index):
    rv = custm[int(2 * i / period)]
    hrv_index = (rv[hrv_index].rvs(size=1))[0]
    return hrv_index

def harvesting():
    h = np.zeros(period)
    hrv_det = np.zeros(period, dtype=int)
    hrv_index = 0
    for i in range(period):
        hrv_index = next_hrv_cont(i, hrv_index)
        hrv_det[i] = hrv_index
        h[i] = hrv[hrv_index]
    if  (sum(h) - (period) * workload < 0):
        return harvesting()
    # plt.plot(h)
    return hrv_det


hrv_det = harvesting()
hrv_change = False
'''hrv_i = [0, 0, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 1, 1, 1, 1, 0, 0, 0, 1]
'''

def next_hrv_cont2(epoch, hrv_index):
    return hrv_det[epoch]


def next_hrv_cont3(epoch, hrv_index):
    global hrv_det
    global hrv_change
    global simnum
    if hrv_change == True:
        hrv_change = False
        hrv_det = harvesting()
        simnum += 1
    return hrv_det[epoch]

def epsilon_greedy(eps, q1, q2):
    # type: (float, float, float) -> int
    # type: (float, float, float) -> int
    eg = np.random.uniform(0, 1)
    if eg < eps:
        # choose a random skip decision uniformly
        aprime = (np.random.choice(2, 1))[0]
    else:
        aprime = np.argmax([q1, q2])
    return aprime


def init_values(action=0):
    bat_index = int(B_init * step / B_max)
    battery = float(B_init)
    window = "0" * (n - 1)
    window += str(action)
    reward = 1 - action
    skips_in_window = action
    return (battery, bat_index, reward, skips_in_window, window)

train_round = 0

class World:
    def __init__(self, battery, bat_index, reward, skips_in_window, window, i):
        self.battery = battery
        self.bat_index = bat_index
        self.reward = reward
        self.skips_in_window = skips_in_window
        self.window = window
        self.i = i

    # skip ? action = 1 : action = 0
    def take_a_observe_r_sprime(self, action, hrv_index, i):

        battery = self.battery - workload * (1 - action) + hrv[hrv_index]
        battery = max(0, min(battery, B_max))
        bat_index = min(int(battery * step / B_max), step - 1)
        # observe r
        r = float(1 - action)
        skips_in_window = self.skips_in_window + action - int(self.window[0])
        window = self.window[1:] + str(action)
        # print (window)
        if skips_in_window > k or bat_index == 0:
            r = infcost
        elif i == (period - 1) and battery < B_init:
            # print (battery)
            r += (-infcost / B_init) * (battery - B_init)
        reward = self.reward + r
        return (World(battery, bat_index, reward, skips_in_window, window, i + 1), r)


def binary_encode(window):
    return np.array([int(c) for c in window])


X_SIZE = 5 #+ n


def phi_of_s_a(s, action, hrv_index):
    a = np.zeros(X_SIZE, dtype=float)
    a[0] = s.battery / B_max
    a[1] = hrv[hrv_index] / h_max
    a[2] = s.i / period
    a[3] = action
    a[4] = s.skips_in_window / k
    #a[5: 5 + n] = binary_encode(s.window)
    return a


class DType:
    def __init__(self, phi, r, phiprime, final):
        self.phi = phi
        self.r = r
        self.phiprime = phiprime
        self.final = final




def train(train_list, session, output, train_operation, X, Y, train_writer, merged, train_opt):
    global train_round
    train_round += 1
    # print (train_round)
    x0_ = np.asarray([t.phi for t in train_list])
    x1_ = np.asarray([t.phiprime for t in train_list])
    f_ = np.asarray([[t.final] for t in train_list])
    r_ = np.asarray([[t.r] for t in train_list])
    rmax = np.max(session.run(output, feed_dict={X: x1_}))
    y_ = r_ + gamma * f_ * rmax
    session.run(train_operation, feed_dict={X: x0_, Y: y_})
    summary, _ = session.run([merged, train_opt], feed_dict={X: x0_, Y: y_})
    train_writer.add_summary(summary, train_round)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        if act == '':
            activations = preactivate
        else:
            activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations

batch_size =  100# * period
term = np.log(batch_size) / np.log(10.0) - 1.0
lrate = float(0.002 * 2 ** term)
print (lrate)
log_dir = "log_simple_stats2"

def initialize_dqn():
    hls = [500, 500, 500, 500]
    session = tf.Session()
    # inputs:
    X = tf.placeholder(tf.float32, [None, X_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, 1], name="Y")


    hidden = [nn_layer(X, X_SIZE, hls[0], 'layer1')]
    for layer in range(1, len(hls)):
        hidden.append(nn_layer(hidden[layer - 1], hls[layer - 1], hls[layer], 'layer' + str(1 + layer)))

    output = nn_layer(hidden[-1], hls[-1], 1, 'output', act=tf.nn.tanh)

    loss = tf.reduce_mean(tf.square(output - Y), name='loss')
    tf.scalar_summary('loss', loss)
    train_operation = tf.train.GradientDescentOptimizer(lrate).minimize(loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(output, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(log_dir + '/train',
                                          session.graph)
    test_writer = tf.train.SummaryWriter(log_dir + '/test')
    session.run(tf.initialize_all_variables())
    return (session, output, train_operation, X, Y, train_writer, merged, train_operation)





def dqn():
    shutil.rmtree(log_dir, True)
    (session, output, train_operation, X, Y, train_writer, merged, train_operation) = initialize_dqn()
    D = np.empty(batch_size, dtype=object)
    d_indx = 0
    bad_count = 0
    repeat = False
    while True:
        (battery, bat_index, reward, skips_in_window, window) = init_values()
        w = World(battery, bat_index, reward, skips_in_window, window, 0)
        hrv_index = 0
        global hrv_change
        hrv_change = not repeat
        for i in range(period):
            repeat = False
            # Note: reward here, is an immediate reward, r in the textbook.
            # (World(battery, bat_index, reward, skips_in_window, window, i + 1), r)
            (w0, r0) = \
                w.take_a_observe_r_sprime(0, hrv_index, i)
            (w1, r1) = \
                w.take_a_observe_r_sprime(1, hrv_index, i)
            hrv_index_ = next_hrv_cont3(i, hrv_index)
            x = [
                [phi_of_s_a(w0, 0, hrv_index_), phi_of_s_a(w0, 1, hrv_index_)],
                [phi_of_s_a(w1, 0, hrv_index_), phi_of_s_a(w1, 0, hrv_index_)]
            ]
            r = [np.max(session.run(output, feed_dict={X: x[0]})),
                 np.max(session.run(output, feed_dict={X: x[1]}))]
            # Choose a from s using policy derived from Q. e.g., epsilon_greedy
            action = epsilon_greedy(epsilon, r0 + r[0], r1 + r[1])
            # s_prime_action
            spa = np.argmax(x[action])
            # Take action a, observe r, and s'
            (wprime, immediate_r) = [(w0, r0), (w1, r1)][action]

            final = 1 - ((i == (period - 1)) or (immediate_r == infcost))
            d = DType(phi_of_s_a(w, action, hrv_index), immediate_r, phi_of_s_a(wprime, spa, hrv_index_), final)
            D[d_indx] = d
            d_indx += 1
            if d_indx == batch_size:
                shuffle(D)
                train(D, session, output, train_operation, X, Y, train_writer, merged, train_operation)
                del D
                D = np.empty(batch_size, dtype=object)
                d_indx = 0
            w = wprime
            hrv_index = hrv_index_
            if immediate_r <= infcost:
                bad_count += 1
                repeat = True
                if bad_count % 49 == 0:
                    repeat = False
                print (simnum)
                break


dqn()


#np.save("qsa.npy",qsa)

def heuristic(i):
    if (i % n < k):
        skip = 1
    else:
        skip = 0
    return skip

def evaluate():
    for j in range(2000):
        w1 = World()
        w2 = World()
        hrv_index = 0
        for i in range(period):
            p = int(2 * i / period)
            # Choose a from s using policy derived from Q. e.g., epsilon_greedy
            a1 = np.argmax(qsa[w1.bat_index][hrv_index][p][w1.skips_in_window])
            a2 = heuristic(i)
            # Take action a, observe r, and s'
            w1.take_action(a1, hrv_index, i)
            w2.take_action(a2, hrv_index, i)
            # print (battery)
            hrv_index = next_hrv_cont(i, hrv_index)
        print (w1.reward, w2.reward)

# evaluate()


    
