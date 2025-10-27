import numpy as np
from tqdm import tqdm
import math
import random

from .del_tracker import DelayTracker

def Greedy(best_acts, env, p_signal, K, tracked_agent, name, n_steps):
    
    losses = []
    
    N_agents = env.N_agents
    T = env.T
    
    tracker = DelayTracker(N_agents)
    
    positions = env.positions
    neighbors = env.neighbors
    A_vals = env.A_vals
    noise = env.noise
    users_and_bs = env.users_and_bs

    
    # Agent states
    buffers = [0 for _ in range(N_agents)]
    # q_0
    q_n = p_signal

    # average reward
    shannon_cap_cum_all = []
    # average cost
    cumulative_cost_q_all = []
    buffer_every_agent = [[] for _ in range(N_agents)]
    
    def instant_cost_for_action(i, mu, s_i, q_n_i, pi_i, user_pos):
        # path losses towards the chosen user and average neighbor attenuation
        a_x = env.attenuation(positions[i], user_pos)
        a_bar = sum(env.attenuation(positions[j], user_pos) for j in neighbors[i]) / (len(neighbors[i]) or 1)

        # interference term built from neighbors' policy distribution (from prev actions)
        # same expression as in your simulation loop
        interf = a_bar * q_n_i * sum(pi_i[m] / m for m in A_vals) if len(neighbors[i]) else 0.0

        np.random.seed(n*i)
        scale =  np.random.exponential(1/mu)
        sinr = (scale * a_x) / (interf + noise)

        # instantaneous cost: -log2(1 + SINR) + lambda_buffer * buffer
        # remove buffer FOR NOW
        cost =  env.compute_cost(sinr, s_i)  #-math.log2(1.0 + sinr) + lambda_buffer * s_i
        return cost, math.log2(1 + sinr), sinr
    
    ##################################
    ###### MAIN LEARNING LOOP ########
    ##################################
    coverage = []
    
    ########3 just for debugging, remove later:
    ####3 thes are the best action values returned by DQN. I wanna see if i really can achieve the performance it did with them     
    #best_acts = best_acts
    
    
    for n in tqdm(range(1, n_steps + 1)):
        
        # compute the number of successfully transmitted singals to find coverage
        sum_transmitted =  0
        
        tracker.start_slot()
        
        shannon_cap_cum_current_step = 0
        cumulative_cost_q_current_step = 0

        actions = best_acts #[]
        
        np.random.seed(n)
        signals = np.random.binomial(1, p_signal, N_agents)

        # check which users are active
        active_users = np.array([
            (signal == 1) or (buf > 0)
            for signal, buf in zip(signals, buffers)
        ])
        
        # generate strengths only for active users
        np.random.seed(n)
        strengths = np.random.exponential(1/np.array(actions))*active_users
        
        inds_buffers_less_than_K = [jj for jj in range(len(buffers)) if buffers[jj] < K]
        
        losses_current_slot = 0
        # update buffers directly, send every obtained signal to the buffer
        for j in range(N_agents):
            # if user j obtained a new signal and
            # was like this but i have no clue why
            # if active_users[j] and (signals[j] == 1):
            # should be like this duh
            if signals[j] == 1:
                if j in inds_buffers_less_than_K:
                    buffers[j] += 1
                    tracker.record_arrival(j, n)
                else:
                    losses_current_slot += 1

        losses.append(losses_current_slot / signals.sum())
        # iterate agents
        for i in range(N_agents):
            
            # if we have a BS with no users assigned, skip
            if len(users_and_bs[i]) == 0:
                continue
                
            # randomly pick a user from the users set
            random.seed(i*n)
            user_id = random.choice(users_and_bs[i])
            # estimate the mean-field across the neighbors set
            q_n, pi = env.estimate_mean_fields(buffers, actions, neighbors, i,  K_buffer = K)
            user_pos = env.user_locations[user_id]
            a_x = env.attenuation(positions[i], user_pos)
            a_bar = sum(env.attenuation(positions[j], user_pos) for j in neighbors[i])/len(neighbors[i])
            interference =  a_bar * q_n * sum(pi[mu]/mu for mu in A_vals)
            SINR = strengths[i] * a_x / (interference + noise)
            success = 1 if SINR > T else 0
            
            if success:
                tracker.record_service(i, n)
            
            # buffers were already incremented by arrivals; now remove served packet if success
            buffers[i] = min(K, buffers[i] - success)
            
            
            buffer_every_agent[i].append(buffers[i])
            
            if active_users[i] == 0:
                best_cost =0
                best_cap = math.log2(1 + SINR) # 0 anyway but keep for sanity check  
                best_success = 0
            else:              
                best_cost = float('inf')
                best_cap = math.log2(1 + SINR)
                # don't need this loop
                mu_best =  best_acts[0]
                c_mu, cur_cap, sinr_mu = instant_cost_for_action(i, mu_best, buffers[i], q_n, pi, user_pos)
                best_cost = c_mu
                best_mu = mu_best
                best_cap = cur_cap
                best_success = 1 if sinr_mu > T else 0
                
                #actions[i] = best_mu
            sum_transmitted += best_success
            
            cost = best_cost #- np.log2(1 + SINR) + lambda_buffer * buffers[i] #+ #lambda_losses*losses_q[i]
            cumulative_cost_q_current_step += cost

            shannon_cap_cum_current_step += best_cap #np.log2(1 + SINR)
            
        coverage.append(sum_transmitted/len(active_users))
            
        avg_delay_this_slot = tracker.end_slot()
            
        # step summaries
        shannon_cap_cum_all.append(shannon_cap_cum_current_step/len(active_users))
        cumulative_cost_q_all.append(cumulative_cost_q_current_step/len(active_users))
        
        
    delay = tracker.avg_delay_per_slot
            
    print('Greedy cost:',  np.mean(cumulative_cost_q_all))
    print('Greedy buffers', np.sum(buffers)/n_steps/N_agents)

    return cumulative_cost_q_all, shannon_cap_cum_all, delay, losses, coverage, np.mean(coverage), np.mean(shannon_cap_cum_all), buffer_every_agent
    #return cumulative_cost_q_all
    