import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

sess = tf.Session()
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
    labels_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, output_size])

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

        # if (epoch+1) % 10 == 0:
        #     print("Epoch: %i/%i; Loss: %.3f"%(epoch+1, NUM_EPOCHS, cumulative_loss / N))

    predictions = sess.run([output_tensor], feed_dict={input_tensor: observations})
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

def main_bc():
    results = ''
    for env_name in env_names:
        with open('data/'+env_name+'_data.pkl', 'rb') as f:
            train_data = pickle.load(f)

        observations = train_data['observations'].squeeze()
        actions = train_data['actions'].squeeze()

        print("Training model for %s"%env_name)

        input_tensor, output_tensor = train_model(observations, actions)

        print("Testing %s in environment"%env_name)
        mean, sd = test_model(input_tensor, output_tensor, env_name)
        expert_mean, expert_sd = np.mean(train_data['returns']), np.std(train_data['returns'])
        
        result_string = "%s\n\tExpert Mean:%.3f\n\tExpert SD:%.3f\n\tBC Mean:%.3f\n\tBC SD:%.3f"%(env_name, expert_mean, expert_sd, mean, sd)
        print(result_string)
        results += result_string + '\n'

    with open('bc_results.txt', 'w') as f:
        f.write(results)

def main_dagger():
    results = ''
    for env_name in env_names:
        with open('data/'+env_name+'_data.pkl', 'rb') as f:
            train_data = pickle.load(f)

        observations = train_data['observations'].squeeze()[:DAGGER_STARTING_N, :]
        actions = train_data['actions'].squeeze()[:DAGGER_STARTING_N, :]

        with sess.as_default():
            mean, sd = dagger(observations, actions, env_name, 45000)
            expert_mean, expert_sd = np.mean(train_data['returns']), np.std(train_data['returns'])
            result_string = "%s\n\tExpert Mean:%.3f\n\tExpert SD:%.3f\n\tDagger Mean:%.3f\n\tDagger SD:%.3f"%(env_name, expert_mean, expert_sd, mean, sd)
            print(result_string)
            results += result_string + '\n'

    with open('dagger_results.txt', 'w') as f:
        f.write(results)

def main_humanoid():
    with open('data/Humanoid-v1_data.pkl', 'rb') as f:
            train_data = pickle.load(f)

    observations = train_data['observations'].squeeze()
    actions = train_data['actions'].squeeze()

    means = []
    sds = []

    for i in range(3, 25):
        input_tensor, output_tensor = train_model(observations, actions, i*10)
        mean, sd = test_model(input_tensor, output_tensor, 'Humanoid-v1')
        print("%i Epochs: Mean=%.3f; SD=%.3f"%(i*10, mean, sd))
        means.append(mean)
        sds.append(sd)
    return means, sds

def main_humanoid_dagger():
    with open('data/Humanoid-v1_data.pkl', 'rb') as f:
            train_data = pickle.load(f)

    with sess.as_default():
        observations = train_data['observations'].squeeze()[:30000,:]
        actions = train_data['actions'].squeeze()[:30000,:]

        expert_policy = load_policy.load_policy('experts/Humanoid-v1.pkl')
        rollouts, max_rollouts = 0, 15
        while rollouts < max_rollouts:
            assert observations.shape[0] == actions.shape[0], "Must have same number of observations and actions"
            input_tensor, output_tensor = train_model(observations, actions, 200)
            new_observations, _, _ = generate_rollouts(input_tensor, output_tensor, 'Humanoid-v1', 1)
            labeled_actions = expert_policy(new_observations)
            
            observations = np.vstack((observations, new_observations))
            actions = np.vstack((actions, labeled_actions))

            print 
            rollouts += 1
            if rollouts > 0 and rollouts % 5 == 0:
                print("Dagger Iterations:", rollouts, test_model(input_tensor, output_tensor, 'Humanoid-v1'))

    return test_model(input_tensor, output_tensor, env_name)

def q2_2():
    envs = ['Ant-v1', 'Hopper-v1']
    for env_name in envs:
        with open('data/'+env_name+'_data.pkl', 'rb') as f:
            train_data = pickle.load(f)

        observations = train_data['observations'].squeeze()
        actions = train_data['actions'].squeeze()

        print("Training model for %s"%env_name)

        input_tensor, output_tensor = train_model(observations, actions)

        print("Testing %s in environment"%env_name)
        mean, sd = test_model(input_tensor, output_tensor, env_name)
        expert_mean, expert_sd = np.mean(train_data['returns']), np.std(train_data['returns'])
        
        result_string = "%s\n\tExpert Mean:%.3f\n\tExpert SD:%.3f\n\tBC Mean:%.3f\n\tBC SD:%.3f"%(env_name, expert_mean, expert_sd, mean, sd)
        print(result_string)        

def q2_3():
    main_humanoid()

def q3_2():
    env_name = 'Walker2d-v1'
    with open('data/'+env_name+'_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    observations = train_data['observations'].squeeze()[:DAGGER_STARTING_N, :]
    actions = train_data['actions'].squeeze()[:DAGGER_STARTING_N, :]

    with sess.as_default():
        mean, sd = dagger(observations, actions, env_name, 45000, logging=True)
        expert_mean, expert_sd = np.mean(train_data['returns']), np.std(train_data['returns'])
        result_string = "%s\n\tExpert Mean:%.3f\n\tExpert SD:%.3f\n\tDagger Mean:%.3f\n\tDagger SD:%.3f"%(env_name, expert_mean, expert_sd, mean, sd)
        print(result_string)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--q2_2', action='store_true')
    parser.add_argument('--q2_3', action='store_true')
    parser.add_argument('--q3_2', action='store_true')
    args = parser.parse_args()

    if args.q2_2:
        q2_2()
    if args.q2_3:
        q2_3()
    if args.q3_2:
        q3_2()




