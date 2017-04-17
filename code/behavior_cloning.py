import pickle
import tensorflow as tf
import numpy as np
import data_collection.tf_util as tf_util
import gym
import data_collection.load_policy as load_policy

tf.logging.set_verbosity(tf.logging.ERROR)

env_names = ['Ant-v1', 'HalfCheetah-v1', 'Hopper-v1', 'Humanoid-v1', 'Reacher-v1', 'Walker2d-v1']
BATCH_SIZE = 1000
DAGGER_STARTING_N = 22000

def random_tensor(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def get_model(input_size, output_size):

    NUM_HIDDEN_UNITS = 128

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

def train_model(observations, actions, NUM_EPOCHS=45):
    assert observations.shape[0] == actions.shape[0], "Must have same number of observations and actions"
    N = observations.shape[0]
    input_size = observations.shape[1]
    output_size = actions.shape[1]

    input_tensor, output_tensor = get_model(input_size, output_size)
    labels_tensor = tf.placeholder(tf.float32, [None, output_size])

    loss = tf.reduce_sum(tf.square(labels_tensor - output_tensor))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

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

def generate_rollouts(input_tensor, output_tensor, env_name, num_rollouts=1):
    env = gym.make(env_name)
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    returns = []
    observations = []
    actions = []

    for i in range(num_rollouts):
        # if num_rollouts > 1 and i%10 == 0:
        #     print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run([output_tensor], feed_dict={input_tensor: obs[None,:]})[0]
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if steps >= max_steps:
                break
        returns.append(totalr)
    return np.array(observations), np.array(actions), returns

def test_model(input_tensor, output_tensor, env_name, num_rollouts=45):
    _, _, returns = generate_rollouts(input_tensor, output_tensor, env_name, num_rollouts)
    return np.mean(returns), np.std(returns)

def dagger(observations, actions, env_name, total_size, max_rollouts=60, logging=False):
    expert_policy = load_policy.load_policy('experts/%s.pkl'%env_name)
    rollouts = 0
    while observations.shape[0] < total_size and rollouts < max_rollouts:
        assert observations.shape[0] == actions.shape[0], "Must have same number of observations and actions"
        num_epochs = int(observations.shape[0]/1000)
        input_tensor, output_tensor = train_model(observations, actions, num_epochs)

        if logging and rollouts % 2 == 0:
            print("rollout", rollouts, "performance", test_model(input_tensor, output_tensor, env_name))
        
        new_observations, _, _ = generate_rollouts(input_tensor, output_tensor, env_name, 1)
        labeled_actions = expert_policy(new_observations)
        
        observations = np.vstack((observations, new_observations))
        actions = np.vstack((actions, labeled_actions))

        rollouts += 1

    return test_model(input_tensor, output_tensor, env_name)



if __name__ == '__main__':
    data = []
    with open('../data/half_cheetah_sub_expert_0.pkl', 'rb') as f:
        data.append(pickle.load(f))
    with open('../data/half_cheetah_sub_expert_1.pkl', 'rb') as f:
        data.append(pickle.load(f))
    with open('../data/half_cheetah_sub_expert_2.pkl', 'rb') as f:
        data.append(pickle.load(f))
    with open('../data/half_cheetah_sub_expert_3.pkl', 'rb') as f:
        data.append(pickle.load(f))

    with tf.Session() as sess:
        input_tensors, output_tensors = [], []
        for i in range(4):
            with tf.variable_scope("model{}".format(i)):
                data_i = data[i]
                observations = data_i['observations'].squeeze()
                actions = data_i['actions'].squeeze()
                input_tensor, output_tensor = train_model(observations, actions, NUM_EPOCHS=60)
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

        saver = tf.train.Saver()
        saver.save(sess, '../data/models/model.ckpt')



