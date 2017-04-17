import pickle
import numpy as np
from sklearn.cluster import KMeans


def generate_sub_experts(expert_path, num_sub_experts):
    with open(expert_path, 'rb') as f:
        expert = pickle.loads(f.read())
    observations = expert['observations']
    actions = expert['actions']
    clusterer = KMeans(num_sub_experts)
    cluster_labels = clusterer.fit_predict(observations)
    for i in range(num_sub_experts):
        with open('../data/half_cheetah_sub_expert_{0}.pkl'.format(i),
                  'wb') as f:
            pickle.dump(
                {'observations': observations[cluster_labels == i],
                 'actions': actions[cluster_labels == i]}, f)


if __name__ == '__main__':
    expert_path = '../data/half_cheetah_expert.pkl'
    generate_sub_experts(expert_path, 4)
