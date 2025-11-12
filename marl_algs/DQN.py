import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
import random
from scipy.spatial import distance_matrix
import math
import pandas as pd
import matplotlib.animation as animation
from scipy.integrate import quad

import scipy.special as special

from tqdm import tqdm

from collections import deque

from .del_tracker import DelayTracker



###################33 DQN shit below ####################33


import random
from collections import deque, namedtuple
import torch

import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple('Transition', 'state action reward next_state done')

aa = 1

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        
        # take random batches from a buffer for training
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        # Stack to tensors
        # state
        s = torch.tensor(np.stack([b.state for b in batch], axis=0), dtype=torch.float32)
        # action
        a = torch.tensor([b.action for b in batch], dtype=torch.int64)
        # reward
        r = torch.tensor([b.reward for b in batch], dtype=torch.float32).unsqueeze(1)
        # next staet
        s2= torch.tensor(np.stack([b.next_state for b in batch], axis=0), dtype=torch.float32)
        # 
        d = torch.tensor([b.done for b in batch], dtype=torch.float32).unsqueeze(1)
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)
    

#######################3 Deep Q-Netowrk

def DQN_sim(env, count_after, p_signal, K, tracked_agent, name, n_steps,
            seed=12345,
            lr=1e-3, gamma=0.9, batch_size=256,
            replay_cap=50000,  # keep fairly recent window
            start_learning_after=1000,
            train_steps_per_slot=1,
            tau=0.005,         # target soft-update
            eps_start=1.0, eps_end=0.05):
    
    # make sure i didn't accidently fuck up the number of steps again
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError(f"n_steps is {n_steps!r} wtf are you doing??")
    
    # track training loss
    training_loss = []
    
    positions = env.positions
    neighbors = env.neighbors
    A_vals = env.A_vals
    noise = env.noise
    #lambda_buffer = env.lambda_buffer ## from environment
    users_and_bs = env.users_and_bs
    T = env.T
    S_max=  env.S_max
    
    ############################333
    
        # DQN
    # dimensions for my NN:
    # the input is a features vector; currently we have 5 features: state, next_state, q, attenuation and MF-attenuation
    # the output is the vector of actions that optimizes my Q-function
    class QNet(nn.Module):
        def __init__(self, in_dim, n_actions):
            super().__init__()
            # sequential NN
            self.net = nn.Sequential(
                # my features are 
                # relu activation 
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                # no activation in the final layer, just raw Q-values; 
                nn.Linear(128, n_actions)
            )
        def forward(self, x):
            return self.net(x)


    #########3
    ##  features: state, mean-field distribution of  q_n and policy pi, attenuation, mean attenuation. 
    def build_features(i, K, buffer_i, q_n, pi, user_id):

        user_pos = env.user_locations[user_id]
        a_x = env.attenuation(positions[i], user_pos)
        if len(neighbors[i]):
            a_bar = sum(env.attenuation(positions[j], user_pos) for j in neighbors[i]) / len(neighbors[i])
            pi_bar = sum(pi[mu] / mu for mu in A_vals)
        else:
            a_bar = 0.0
            pi_bar = 0.0

        # the vector of features   
        x = np.array([
            buffer_i / K,    # normalized buffer (porb not necessary if buffer is nor large)
            q_n,                        
            pi_bar,             
            a_x,
            a_bar
        ], dtype=np.float32)
        return x, a_x, a_bar, pi_bar
        

    ########################################3
    rng = env.make_rng(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 
    N_agents = env.N_agents
    
       # just for debugging: in addition to average capacity an cost, we will also track it separately by each user
    cost_every_agent = [[] for _ in range(N_agents)]
    cap_every_agent = [[] for _ in range(N_agents)]
    ### actions also
    action_every_agent = [[] for _ in range(N_agents)]

    ### and buffers
    buffer_every_agent = [[] for _ in range(N_agents)]

    tracker = DelayTracker(N_agents) #env.delay_tracker#(N_agents)
    losses = []
    shannon_cap_cum_all = []
    cumulative_cost_q_all = []
    coverage = []

    # NN setup
    # feature dim = 5 (buffer/K, q_n, pi_bar, a_x, a_bar)
    # number
    obs_dim = 5
    n_actions = len(A_vals)
    
    # We use a double DQN, so we create 2 identical NNs
    online = QNet(obs_dim, n_actions).to(device)
    target = QNet(obs_dim, n_actions).to(device)
    # inherit everything in online to target (like init) so they start identical
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    replay = ReplayBuffer(replay_cap)

    # Agent states
    buffers = [0 for _ in range(N_agents)]
    # initialize actions randomд
    actions_prev = list(rng.choice(A_vals, size=N_agents))

    # track one agents Q snapshots 
    q_table_history = [[] for _ in range(tracked_agent)]  

    global_step = 0

    for n in tqdm(range(1, n_steps + 1)):
        tracker.start_slot()
        sum_transmitted = 0
        shannon_cap_cum_current_step = 0.0
        cumulative_cost_q_current_step = 0.0

        # epsiloon schedule
        frac = n / max(1, n_steps)
        epsilon = max(eps_end, eps_start * (1 - frac))

        # generate arrivals
        signals = rng.binomial(1, p_signal, N_agents)

        # active users
        active_users = (signals == 1) | (np.array(buffers) > 0)

        # Update buffers with arrivals BUT only those that have a non-empty buffer. Otherwise loss
        inds_buffers_less_than_K = [jj for jj in range(len(buffers)) if buffers[jj] < K]
        losses_current_slot = 0
        for j in range(N_agents):
            if signals[j] == 1:
                if j in inds_buffers_less_than_K:
                    buffers[j] += 1
                    tracker.record_arrival(j, n)
                else:
                    losses_current_slot += 1
        losses.append(losses_current_slot / max(1, int(signals.sum())))

        # Freeze a snapshot of buffers to compute MF consistently in this slot
        buffers_freeze = buffers.copy()

        # Choose actions from NN Q(x)
        actions = []
        action_idx = []
        # randommly chose a user to be served
        chosen_users = []

        # Precompute per-agent frozen MF (q_n, pi) once for the slot
        #  use current 'actions' profile for MF.
        # But since 'actions' not yet chosen, fallback to previous profile actions_prev for MF snapshot.
        # After we pick actions, interference uses this frozen MF snapshot to avoid simultaneity.
        #
        # We compute per-agent MF after we know which user they target (a_x, a_bar depend on user).
        # So we’ll compute MF inside the loop but use buffers_freeze and actions_prev to freeze.
    
        mf_actions = actions_prev

        # First, pick a user per agent (random)
        for i in range(N_agents):
            if len(users_and_bs[i]) == 0:
                chosen_users.append(None)
                continue
            user_id = rng.choice(users_and_bs[i])
            chosen_users.append(user_id)

        # Now build state features and choose action with eps-greedy
        states_this_slot = [None] * N_agents  # store to reuse for transition logging
        
        for i in range(N_agents):
            if len(users_and_bs[i]) == 0:
                #actions.append(A_vals[0])  # idle or smallest power
                action_idx.append(0)
                states_this_slot[i] = None
                continue

            user_id = chosen_users[i]
            # Estimate MF for this agent (frozen)
            q_n_i, pi_i = env.estimate_mean_fields(buffers_freeze, mf_actions, neighbors, i, K_buffer=K)

            # Build features
            x_i, a_x, a_bar, pi_bar = build_features(i, K, buffers[i], q_n_i, pi_i, user_id)

            # ε-greedy
            if (rng.random() < epsilon) or (not np.isfinite(x_i).all()):
                a = rng.choice(A_vals)
            else:
                with torch.no_grad():
                    qvals = online(torch.tensor(x_i, dtype=torch.float32, device=device).unsqueeze(0))
                    a_idx = int(torch.argmax(qvals, dim=1).item())
                    a = A_vals[a_idx]

            actions.append(a)
            
            #action_every_agent[i].append(a)
            
            action_idx.append(A_vals.index(a))
            states_this_slot[i] = x_i

        # signal strengths given chosen actions
        strengths = rng.exponential(1 / np.array(actions)) * active_users

        # Interact with env, compute reward/cost, log transitions
        next_states_this_slot = [None] * N_agents

        for i in range(N_agents):
            if len(users_and_bs[i]) == 0:
                continue
            
            # tg passs:
            # 123wqeswGG3454hfg!!64rehg$$hh

            user_id = chosen_users[i]
            # Recompute MF (the same frozen snapshot!) do NOT let buffers update leak
            q_n_i, pi_i = env.estimate_mean_fields(buffers_freeze, mf_actions, neighbors, i, K_buffer=K)

            user_pos = env.user_locations[user_id]
            a_x = env.attenuation(positions[i], user_pos)
            if len(neighbors[i]):
                a_bar = sum(env.attenuation(positions[j], user_pos) for j in neighbors[i]) / len(neighbors[i])
                # with S ~ exp(mu):
                interference = len(neighbors[i]) * a_bar * q_n_i * sum(pi_i[mu] / mu for mu in A_vals)
                # with S = min (Y ~ exp(mu), S_max):
                #E_S = lambda mu: (1 - math.exp(-S_max*mu))/mu
                #interference = len(neighbors[i]) * a_bar * q_n_i * sum(pi_i[mu]*E_S(mu) for mu in A_vals)
            else:
                a_bar = 0.0
                interference = 0.0

            SINR = strengths[i] * a_x / (interference + noise)
            success = 1 if SINR > T else 0
            if success:
                tracker.record_service(i, n)
                sum_transmitted += 1

            # Update buffer (arrivals already added), remove one if served
            #prev_buf = buffers[i]
            buffers[i] = min(K, max(0, buffers[i] - success))

            # Cost and reward
            cost = env.compute_cost(SINR, buffers[i]) #- np.log2(1 + SINR) + lambda_buffer * buffers[i]
            reward = - float(cost)

            cumulative_cost_q_current_step += float(cost)
            shannon_cap_cum_current_step += float(math.log2(1 + SINR))
            
            
            cost_every_agent[i].append(cost)
            cap_every_agent[i].append(float(math.log2(1 + SINR)))
            buffer_every_agent[i].append(buffers[i])

            # Next state features (for x')
            x_next_i, _, _, _ = build_features(i, K, buffers[i], q_n_i, pi_i, user_id)
            next_states_this_slot[i] = x_next_i

            # Store transition in replay
            s = states_this_slot[i]
            a_idx = action_idx[i]
            s2 = x_next_i
            done = 0.0  # continuing task // actually don't need it
            if s is not None:
                replay.push(s, a_idx, reward, s2, done)

            global_step += 1

        # Metrics
        den = max(1, int(active_users.sum()))
        shannon_cap_cum_all.append(shannon_cap_cum_current_step / den)
        cumulative_cost_q_all.append(cumulative_cost_q_current_step / den)
        coverage.append(sum_transmitted / den)
        # track each agent separately
 
        # Learning...
        if len(replay) >= max(batch_size, start_learning_after):
            for _ in range(train_steps_per_slot):
                # now i have enough samples in replay so i start learning.
                # I randomly sample 256 tuples with features (or whatever batch size will be)
                s, a, r, s2, d = replay.sample(batch_size)
                s  = s.to(device); a = a.to(device); r = r.to(device); s2 = s2.to(device); d = d.to(device)

                #Now, when I've picked a sample, I also pull its Q-value (the one previoulsy stored in qvals)
                # Q(s, a)
                q_sa = online(s).gather(1, a.unsqueeze(1))  # [B,1]

                # Double DQN target
                with torch.no_grad():
                    # get the best action PER BATCH! (max Q value)
                    a_star = online(s2).argmax(dim=1, keepdim=True)       # [B,1]
                    # from a targer table, take the Q vals that corresponds to the best action
                    q_next = target(s2).gather(1, a_star)                  # [B,1]
                    # target formula
                    #y = r + gamma * (1.0 - d) * q_next
                    # I don't really need indicator so i do like this
                    y = r + gamma * q_next

                loss = F.mse_loss(q_sa, y)
                # keep track of losses
                training_loss.append(loss.item())
                
                # clear all previous gradients
                optimizer.zero_grad()
                # compute derivatives
                loss.backward()
                # normalize gradients
                torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()

                # Soft target update
                with torch.no_grad():
                    for p_t, p in zip(target.parameters(), online.parameters()):
                        p_t.data.mul_(1 - tau).add_(tau * p.data)

        # Stage commit for MF actions next slot (we used actions_prev to freeze MF here)
        actions_prev = actions

        #  snapshot a few agents' Q-vectors
        if n % env.n_steps_plot == 0:
            for i in range(min(tracked_agent, N_agents)):
                # record network output at current state (approx)
                if states_this_slot[i] is not None:
                    with torch.no_grad():
                        qvals = online(torch.tensor(states_this_slot[i], dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy().squeeze()
                    q_table_history[i].append(qvals.copy())
                    ######### update other metrics:
                    action_every_agent[i].append(actions[i])
                    #cost_every_agent[i].append

        tracker.end_slot() # for the delay tracker, at this point we compute the number of required transmission attempts

    print('NN-Q (DoubleDQN) avg cost:', float(np.mean(cumulative_cost_q_all)))
    print('NN-Q buffers:', float(np.sum(buffers) / n_steps / N_agents))

    delay = tracker.avg_delay_per_slot

    return cumulative_cost_q_all, shannon_cap_cum_all, q_table_history, delay, losses, coverage, float(np.mean(coverage[count_after:])), float(np.mean(shannon_cap_cum_all[count_after:])), training_loss, cost_every_agent, cap_every_agent, action_every_agent, buffer_every_agent

