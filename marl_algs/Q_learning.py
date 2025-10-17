import numpy as np
from tqdm import tqdm
import random
import math

from .del_tracker import DelayTracker

###########################3 Q-learning

def Q_sim(env, p_signal, K, tracked_agent, name, n_steps):
    
    
    
    N_agents = env.N_agents
    T = env.T
    M = env.M
    
    positions = env.positions
    neighbors = env.neighbors
    A_vals = env.A_vals
    noise = env.noise
    #lambda_buffer = env.lambda_buffer
    users_and_bs = env.users_and_bs
    gamma = env.gamma
    
    tracker = DelayTracker(N_agents) #env.delay_tracker
    # Agent cumulative losses (Q-learning)
    losses = []
    #delay = np.array([0 for _ in range(N_agents)])
    
    Q_tables = [np.zeros((K + 1, M)) for _ in range(N_agents)]
    
    # Agent states
    buffers = [0 for _ in range(N_agents)]
    # q_0
    q_n = p_signal

    # try this: choose one agent and track its entire Q history
    q_table_history = [[] for _ in range(tracked_agent)]

    # average reward
    shannon_cap_cum_all = []
    # average cost
    cumulative_cost_q_all = []
    
    ##################################
    ###### MAIN LEARNING LOOP ########
    ##################################
    coverage = []

    for n in tqdm(range(1, n_steps + 1)):
        
        # at each slot, keep number of successfully tramsmitted signals 
        sum_transmitted = 0

        
        #################################
        # start recording arrivals
        tracker.start_slot()
        
        shannon_cap_cum_current_step = 0
        cumulative_cost_q_current_step = 0

        
        
        
        # was like that
        #epsilon = max(0.05, 1.0 - n / n_steps)  #1 / (1 + 0.001 * n)
        #but i wil do the same as DQN
          # ε schedule
        ###################################################3
        eps_start=1.0
        eps_end=0.05
        frac = n / max(1, n_steps)
        epsilon = max(eps_end, eps_start * (1 - frac))
        ################################################33
        
        alpha =  max(0.05, 0.5 / (1 + 0.001 * n)) #1 / (1 + 0.002 * n)
        actions = []

        for k in range(N_agents):
            # current state
            s = buffers[k]
            random.seed(k*n)
            rand_action = np.random.choice(A_vals) 
            # epsilon-greedy over Q (for exploration)
            if random.random() < epsilon:
                actions.append(rand_action)
            else:
                 actions.append(A_vals[np.argmin(Q_tables[k][s])])
                #mf = mf_bin(q_n)   # q_n already computed per agent
                #a_idx = np.argmin(Q_tables[k][s, :, mf])
                #actions.append(A_vals[a_idx])
            if n == 100:
                aa = 1

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
        
        # which users have non-full buffer
        inds_buffers_less_than_K = [jj for jj in range(len(buffers)) if buffers[jj] < K]
        # update buffers directly, send every obtained signal to the buffer
        losses_current_slot = 0
    
        for j in range(N_agents):
            # if user j obtained a new signal and 
            # was like this but i have no clue why
            #if active_users[j] and (signals[j] == 1):
            # should be like this duh 
            if  signals[j] == 1:
                if j in inds_buffers_less_than_K:
                    buffers[j] += 1
                    tracker.record_arrival(j, n)
                else:
                    losses_current_slot+=1
                    
        #  loss rate
        losses.append(losses_current_slot/signals.sum())
                
                
        #inds_buffers_K = list(np.delete(range(len(buffers)), inds_buffers_less_than_K))
        #mask = (signals == 1) & np.isin(np.arange(len(losses_q)), inds_buffers_K)
        #losses_q[mask] += 1
                    # freeze buffers to use the same buffer state for all agents 
        buffers_freeze = buffers.copy()


        # iterate agents
        for i in range(N_agents):
            
            # if we have a BS with no users assigned, skip
            if len(users_and_bs[i]) == 0:
                continue
                
            # randomly pick a user from the users set
            
            random.seed(i*n)
            user_id = random.choice(users_and_bs[i])

            # estimate the mean-field across the neighbors set
            #mf_prev = mf_bin(q_n)        # bin for previous step (later used in Q-learning)
            q_n, pi = env.estimate_mean_fields(buffers_freeze, actions, neighbors, i, K_buffer = K)
            user_pos = env.user_locations[user_id]
            a_x = env.attenuation(positions[i], user_pos)
            a_bar = sum(env.attenuation(positions[j], user_pos) for j in neighbors[i])/len(neighbors[i])
            interference =  a_bar * q_n * sum(pi[mu]/mu for mu in A_vals)
            SINR = strengths[i] * a_x / (interference + noise)
            success = 1 if SINR > T else 0
            
            if success:
                  tracker.record_service(i, n)
                  sum_transmitted += 1
            
            prev_state = buffers.copy()[i]
            # buffers were already incremented by arrivals; now remove served packet if success
            
            buffers[i] = min(K, buffers[i] - success)
            # remover buffer FOR NOW
            cost = env.compute_cost(SINR, buffers[i])  #- np.log2(1 + SINR) + lambda_buffer * buffers[i] #+ #lambda_losses*losses_q[i]
            cumulative_cost_q_current_step += cost
            Q = Q_tables[i]
            
            # now try to get back to the 2-d Q-table
            Q[prev_state][A_vals.index(actions[i])] += alpha * (
            cost + gamma * np.min(Q[buffers[i]]) - Q[prev_state][A_vals.index(actions[i])]
            )
            
            # was like this with bins:
            # mf_curr = mf_bin(q_n)        # or recompute with new state if q_n changes
            # a_idx = A_vals.index(actions[i])
            
            # Q[prev_state, a_idx, mf_prev] += alpha * (
            #    cost + gamma * np.min(Q[buffers[i], :, mf_curr])
            #    - Q[prev_state, a_idx, mf_prev]
            # )

            shannon_cap_cum_current_step += math.log2(1 + SINR)
            
        # end of slot: compute delay per slot
        avg_delay_this_slot = tracker.end_slot()
            
        # step summaries
        shannon_cap_cum_all.append(shannon_cap_cum_current_step/len(active_users)) # I replaced len(active_users) by N_agents 
        cumulative_cost_q_all.append(cumulative_cost_q_current_step/len(active_users))
        
        coverage.append(sum_transmitted/len(active_users))
        
        
        if n % env.n_steps_plot == 0:
            for i in range(tracked_agent):
                q_table_history[i].append(Q_tables[i].copy())
            
    print('Q-learning cost:',  np.mean(cumulative_cost_q_all))
    print('Q-learning buffers', np.sum(buffers)/n_steps/N_agents)
    
    delay = tracker.avg_delay_per_slot
    
        
    return cumulative_cost_q_all, shannon_cap_cum_all, q_table_history, delay, losses,  coverage, np.mean(coverage), np.mean(shannon_cap_cum_all)
    #return cumulative_cost_q_all


################################ Q-learning ends
