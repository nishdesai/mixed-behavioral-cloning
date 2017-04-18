import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import keras as k
from keras.models import Sequential
from keras.layers import Dense

import pickle

# from behavior_cloning import *

import logging
logging.basicConfig(filename='./train_log.txt', filemode='w',level=logging.INFO)

BATCH_SIZE = 1000

def random_tensor(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def get_model(input_size, output_size, scope=None):

    NUM_HIDDEN_UNITS = 128

    with tf.variable_scope(scope):
        input_tensor = tf.placeholder(tf.float32, [None, input_size])

        #first hidden layer:
        W_1 = random_tensor([input_size, NUM_HIDDEN_UNITS])
        b_1 = random_tensor([1, NUM_HIDDEN_UNITS])
        fc_1 = tf.nn.relu(tf.matmul(input_tensor, W_1) + b_1)

        #second hidden layer:
        W_2 = random_tensor([NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS])
        b_2 = random_tensor([1, NUM_HIDDEN_UNITS])
        fc_2 = tf.nn.relu(tf.matmul(fc_1, W_2) + b_2)

        #output layer
        W_out = random_tensor([NUM_HIDDEN_UNITS, output_size])
        b_out = random_tensor([1, output_size])
        output_tensor = tf.matmul(fc_2, W_out) + b_out

    return input_tensor, output_tensor

def train_model(observations, actions, NUM_EPOCHS=45, sess=tf.get_default_session(), scope=''):
    assert observations.shape[0] == actions.shape[0], "Must have same number of observations and actions"
    N = observations.shape[0]
    input_size = observations.shape[1]
    output_size = actions.shape[1]

    input_tensor, output_tensor = get_model(input_size, output_size, scope=scope)
    labels_tensor = tf.placeholder(tf.float32, [None, output_size])

    loss = tf.reduce_sum(tf.square(labels_tensor - output_tensor))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)


    sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)))

    NUM_BATCHES = int(N / BATCH_SIZE)
    permutation = np.random.permutation(N)
    in_data, out_data = observations[permutation, :], actions[permutation, :]

    for epoch in range(NUM_EPOCHS):
        cumulative_loss = 0

        for batch in range(NUM_BATCHES):
            in_batch, out_batch = in_data[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :], out_data[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :]
            _, loss_value = sess.run([train_op, loss], feed_dict={input_tensor: in_batch, labels_tensor: out_batch})
            cumulative_loss += loss_value

        if (epoch+1) % 10 == 0:
            print("Epoch: %i/%i; Loss: %.3f"%(epoch+1, NUM_EPOCHS, cumulative_loss / N))

    # predictions = sess.run([output_tensor], feed_dict={input_tensor: observations})
    return input_tensor, output_tensor

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def pathlength(path):
    return len(path["reward"])

class LinearValueFunction:
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NnValueFunction:
    def __init__(self, ob_dim, n_epochs, stepsize, preproc=False):
        self.model = None
        self.ob_dim = ob_dim
        self.n_epochs = n_epochs
        self.learning_rate = stepsize
        self.should_preproc = preproc

        model = Sequential()
        if self.should_preproc:
            model.add(Dense(512, input_shape=(2*self.ob_dim,), activation='relu'))
        else:
            model.add(Dense(512, input_shape=(self.ob_dim,), activation='relu'))
        model.add(Dense(512, input_shape=(512,), activation='relu'))
        model.add(Dense(1, input_shape=(512,)))

        model.compile(optimizer='rmsprop', loss='mean_squared_error', lr=self.learning_rate)
        self.model = model

    def fit(self, X, y):
        if self.should_preproc:
            self.model.fit(self.preproc(X), y, nb_epoch=self.n_epochs)
        else:
            self.model.fit(X, y, nb_epoch=self.n_epochs)

    def predict(self, X):
        if not self.model:
            return np.zeros(X.shape[0])
        else:
            if self.should_preproc:
                return np.reshape(self.model.predict(self.preproc(X)), [-1,])
            else:
                return np.reshape(self.model.predict(X), [-1,])


    def preproc(self, X):
        return np.concatenate([X, np.square(X)/2.0], axis=1)


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def main(input_tensors, output_tensors, n_iter=500000, gamma=0.95, min_timesteps_per_batch=50, stepsize=1e-4, animate=False, logdir=None, seed=0, vf_type='linear', vf_params={}, desired_kl=1e-03):
    logfile = open('train_log.txt', 'w')
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("HalfCheetah-v1")
    ob_dim = env.observation_space.shape[0]
    num_actions = len(input_tensors)
    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(ob_dim=ob_dim, preproc=False, **vf_params)


    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_rew_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    sy_adv_n = sy_rew_n - vf.model(sy_ob_no)
    sy_h1 = lrelu(dense(sy_ob_no, 512, "h1", weight_init=normc_initializer(1.0))) # hidden layer
    sy_h2 = lrelu(dense(sy_h1, 512, "h2", weight_init=normc_initializer(1.0))) # hidden layer
    sy_logits_na = dense(sy_h2, num_actions, "final", weight_init=normc_initializer(0.05)) # "logits", describing probability distribution of final layer
    # we use a small initialization for the last layer, so the initial policy has maximal entropy
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    sy_logp_na = tf.nn.log_softmax(sy_logits_na) # logprobability of actions
    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0] # sampled actions, used for defining the policy (NOT computing the policy gradient)
    sy_n = tf.shape(sy_ob_no)[0]
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na) 
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = tf.reduce_mean(-sy_adv_n * sy_logprob_n +  10.0 * tf.square(sy_adv_n)) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.RMSPropOptimizer(sy_stepsize).minimize(sy_surr)

    # tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    # # use single thread. on such a small problem, multithreading gives you a slowdown
    # # this way, we can better use multiple cores for different experiments
    # sess.__enter__() # equivalent to `with sess:`
    to_initialize = set(tf.all_variables()) - ALREADY_INITIALIZED
    # import pdb; pdb.set_trace()
    sess.run(tf.initialize_variables(to_initialize))

    # for i in range(4):
    #     with tf.variable_scope('model{}'.format(i)):
    #         print test_model(input_tensors[i], output_tensors[i], 'HalfCheetah-v1', sess=sess)

    total_timesteps = 0

    ep_rew = 0.0
    ob = env.reset()
    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
       
        
        terminated = False
        obs, acs, rewards = [], [], []
        while timesteps_this_batch < min_timesteps_per_batch:
            obs.append(ob)
            ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
            acs.append(ac)
            # with tf.variable_scope("model{}".format(ac)) as scope:
            env_action = sess.run(output_tensors[ac], feed_dict={input_tensors[ac] : ob[None]}).squeeze()
            # import pdb; pdb.set_trace()
            ob, rew, done, _ = env.step(env_action)
            rewards.append(rew)
            ep_rew += rew
            if done:
                ob = env.reset()
                print "EP Rew:", ep_rew
                logfile.write("EP Rew: {}\n".format(ep_rew))
                logfile.flush()
                ep_rew = 0.0
            timesteps_this_batch += 1
        path = {"observation" : np.array(obs), "terminated" : terminated,
                "reward" : np.array(rewards), "action" : np.array(acs)}
        paths.append(path)

        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            # vpred_t = vf.predict(path["observation"])
            # adv_t = return_t - vpred_t
            # advs.append(adv_t)
            vtargs.append(return_t)
            # vpreds.append(vpred_t)

        # # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        # adv_n = np.concatenate(advs)
        # standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        # vpred_n = np.concatenate(vpreds)
        # # vf.fit(ob_no, vtarg_n)

        # # Policy update
        _, oldlogits_na = sess.run([update_op, sy_logits_na], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_rew_n:vtarg_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldlogits_na:oldlogits_na})

        if kl > desired_kl * 2 and i > 0: 
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2 and i > 0: 
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        # logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        # logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()


def main1(d):
    return main(**d)

if __name__ == "__main__":
    data = []
    with open('../data/half_cheetah_sub_expert_0.pkl', 'rb') as f:
        data.append(pickle.load(f))
    with open('../data/half_cheetah_sub_expert_1.pkl', 'rb') as f:
        data.append(pickle.load(f))
    with open('../data/half_cheetah_sub_expert_2.pkl', 'rb') as f:
        data.append(pickle.load(f))
    with open('../data/half_cheetah_sub_expert_3.pkl', 'rb') as f:
        data.append(pickle.load(f))

    env = gym.make("HalfCheetah-v1")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    with tf.Session() as sess:
        input_tensors, output_tensors = [], []
        for i in range(4):
            with tf.variable_scope("model{}".format(i)) as scope:
                data_i = data[i]
                observations = data_i['observations'].squeeze()
                actions = data_i['actions'].squeeze()

                input_tensor, output_tensor = train_model(observations, actions, NUM_EPOCHS=60, sess=sess, scope=scope.name)
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

                # print test_model(input_tensor, output_tensor, 'HalfCheetah-v1', sess=sess)


        # saver = tf.train.Saver()
        # saver.restore(sess, '../data/models/model.ckpt')

        # for i in range(4):
        #     with tf.variable_scope('model{}'.format(i)):
        #         print test_model(input_tensors[i], output_tensors[i], 'HalfCheetah-v1', sess=sess)


        # import pdb; pdb.set_trace()
        ALREADY_INITIALIZED = set(tf.all_variables())

        params = [
            dict(input_tensors=input_tensors, output_tensors=output_tensors, logdir=None, seed=0, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3)),
            # dict(logdir='/tmp/ref1/nnvf-kl2e-3-seed0', seed=0, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3)),
            dict(input_tensors=input_tensors, output_tensors=output_tensors, logdir=None, seed=1, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3)),
            # dict(logdir='/tmp/ref1/nnvf-kl2e-3-seed1', seed=1, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3)),
            dict(input_tensors=input_tensors, output_tensors=output_tensors, logdir=None, seed=2, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3)),
            # dict(logdir='/tmp/ref1/nnvf-kl2e-3-seed2', seed=2, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3)),
        ]
        # import multiprocessing
        # p = multiprocessing.Pool()
        # p.map(main1, params)
        for p in params:
            main1(p)

        