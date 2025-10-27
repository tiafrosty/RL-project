import numpy as np
from scipy.spatial import Delaunay
import random
from scipy.spatial import distance_matrix

from itertools import groupby
from itertools import groupby
import random
import math

import numpy as np
from scipy.spatial import Delaunay



class Env:
    def __init__(self):
        # ===== Parameters (same names/values as before) =====
        # buffer size
        #K = 100
        # users int
        self.user_intensity = 5000
        # action space
        self.A_vals = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0] #, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]
        # volume of action space
        self.M = len(self.A_vals)
        # 
        self.gamma = 0.9
        # SINR threshold
        self.T = 1.0
        ## maximal allowed intensity
        self.S_max = 20
        # sigma
        self.noise = 1  # 1e-3
        # input rate (let's try with p = q = 1 for now)
        #p_signal = 0.7
        # delay penalty 
        self.lambda_buffer = 0.3
        #lambda_losses = 10*lambda_buffer*K/p_signal
        # the default signal strength 
        self.mu_base = 1.0
        # pathloss exponent
        self.beta_pathloss = 4.0
        # simulation length
        self.n_steps_plot = 100

        # PPP domain and intensity
        # size of the network
        self.L = 1.0
        # BS intensity
        self.lambda_BS = 200 # BSs per unit area

        # ===== Random seed and PPP for BSs (same global RNG usage) =====
        np.random.seed(10)
        self.N_agents = np.random.poisson(self.lambda_BS * self.L * self.L)
        # PPP locations (FIXED)
        self.positions = np.random.uniform(0, self.L, (self.N_agents, 2))

        # ===== Compute Delaunay and neighbors (same loop/logic) =====
        self.delaunay = Delaunay(self.positions)
        self.neighbors = [[] for _ in range(self.N_agents)]
        for simplex in self.delaunay.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if b not in self.neighbors[a]:
                    self.neighbors[a].append(b)
                if a not in self.neighbors[b]:
                    self.neighbors[b].append(a)

        # ===== Users PPP and association  =====
        area = self.L ** 2
        self.num_points = np.random.poisson(self.user_intensity * area)
        self.user_locations = np.random.uniform(0, self.L, (self.num_points, 2))
        # create an association map between users and BSs
        self.users_and_bs = self.associate_users_to_bs(self.user_locations, self.positions)

        # ===== Delay tracker =====
        #self.delay_tracker = DelayTracker(self.N_agents)
        
        
    def compute_cost(self, SINR, buff_i):
        
        return  - math.log2(1 + SINR) + self.lambda_buffer * buff_i


    # Use shortest distance to associate users and BSs
    def associate_users_to_bs(self, user_locations, bs_locations):
        """
        Assign each user to the closest base station.

        Parameters:
            user_locations (np.ndarray)
            bs_locations (np.ndarray)

        Returns:
            dict: Keys are BS indices, values are lists of user indices.
        """
        dist_matrix = distance_matrix(user_locations, bs_locations)
        closest_bs = np.argmin(dist_matrix, axis=1)

        bs_user_map = {i: [] for i in range(len(bs_locations))}
        for user_idx, bs_idx in enumerate(closest_bs):
            bs_user_map[bs_idx].append(user_idx)

        return bs_user_map

    # a(x), x - BS location, y = user's location
    def attenuation(self, x, y, beta=None):
        if beta is None:
            beta = self.beta_pathloss
        if len(y) == len(x):
            return 1 / (np.linalg.norm(x - y) ** beta)
        else:
            x_array = np.atleast_2d(x)
            y_array = np.asarray(y)
            distances = np.linalg.norm(x_array - y_array, axis=1)
            return 1 / (distances ** beta)

    # MF implementation
    def all_equal(self, iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    def estimate_mean_fields(self, buffers, actions, neighbors, agent_id, K_buffer):
        neighbor_ids = neighbors[agent_id]
        # buffer mean-field distribution
        beta = {b: 0 for b in range(K_buffer + 1)}
        # action mean-field distribution
        pi = {mu: 0 for mu in self.A_vals}
        for j in neighbor_ids:
            beta[buffers[j]] += 1
            pi[actions[j]] += 1
        count = len(neighbor_ids) or 1
        beta = {b: v / count for b, v in beta.items()}
        pi = {mu: v / count for mu, v in pi.items()}
        # p active, i-dependent
        q_n = 1 - beta[0]
        return q_n, pi

    # the seeding function/random numbers generator
    def make_rng(self, seed=12345): 
        rng = np.random.default_rng(seed) 
        import random as _random
        _random.seed(seed) 
        return rng