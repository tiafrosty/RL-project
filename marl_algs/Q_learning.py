import numpy as np
from tqdm import tqdm
#import random
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
    
    S_max=  env.S_max
    
    tracker = DelayTracker(N_agents) #env.delay_tracker
    # Agent cumulative losses (Q-learning)
    losses = []
    #delay = np.array([0 for _ in range(N_agents)])
    
    Q_tables = [np.zeros((K + 1, M, M)) for _ in range(N_agents)]
    
        ### and buffers
    buffer_every_agent = [[] for _ in range(N_agents)]
    
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
    # keep all chosen actions
    action_every_agent = [[] for _ in range(N_agents)]
    
    ##################################
    ###### MAIN LEARNING LOOP ########
    ##################################
    coverage = []


    mean_actions =[0 for _ in range(N_agents)]
    
    seed=12345
    rng = env.make_rng(seed)

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
            m_idx = mean_actions[k]
            #random.seed(k*n)
            #actions_prev = list(rng.choice(A_vals, size=N_agents))
            rand_action = list(rng.choice(A_vals, size=N_agents))[k] #np.random.choice(A_vals) 
            # epsilon-greedy over Q (for exploration)
            if rng.random() < epsilon:
                actions.append(rand_action)
            else:
                 # 2d was like this
                 #actions.append(A_vals[np.argmin(Q_tables[k][s])])
                 # 3d should be like this?
                a_idx = np.argmin(Q_tables[k][s][:, m_idx])  # fix current mfg
                actions.append(A_vals[a_idx])
                 
                #mf = mf_bin(q_n)   # q_n already computed per agent
                #a_idx = np.argmin(Q_tables[k][s, :, mf])
                #actions.append(A_vals[a_idx])
            if n == 100:
                aa = 1

        #np.random.seed(n)
        signals = rng.binomial(1, p_signal, N_agents)
        #signals = np.random.binomial(1, p_signal, N_agents)

        # check which users are active
        active_users = np.array([
            (signal == 1) or (buf > 0)
            for signal, buf in zip(signals, buffers)
        ])
        
        # generate strengths only for active users
        #np.random.seed(n)
        strengths = rng.exponential(1/np.array(actions))*active_users
        
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
            
            #random.seed(i*n)
            user_id = rng.choice(users_and_bs[i]) #random.choice(users_and_bs[i])

            # estimate the mean-field across the neighbors set
            #mf_prev = mf_bin(q_n)        # bin for previous step (later used in Q-learning)
            q_n, pi = env.estimate_mean_fields(buffers_freeze, actions, neighbors, i, K_buffer = K)
            user_pos = env.user_locations[user_id]
            a_x = env.attenuation(positions[i], user_pos)
            a_bar = sum(env.attenuation(positions[j], user_pos) for j in neighbors[i])/len(neighbors[i])
            
            interference =  len(neighbors[i]) * a_bar * q_n * sum(pi[mu]/mu for mu in A_vals)
            #E_S = lambda mu: (1 - math.exp(-S_max*mu))/mu
            #interference =  len(neighbors[i]) * a_bar * q_n * sum(pi[mu]*E_S(mu) for mu in A_vals)
            
            SINR = strengths[i] * a_x / (interference + noise)
            success = 1 if SINR > T else 0
            
            if success:
                  tracker.record_service(i, n)
                  sum_transmitted += 1
            
            prev_state = buffers.copy()[i]
            # buffers were already incremented by arrivals; now remove served packet if success
            
            buffers[i] = min(K, buffers[i] - success)
            
                ### and buffers
            buffer_every_agent[i].append(buffers[i])
            # remover buffer FOR NOW
            cost = env.compute_cost(SINR, buffers[i])  #- np.log2(1 + SINR) + lambda_buffer * buffers[i] #+ #lambda_losses*losses_q[i]
            cumulative_cost_q_current_step += cost
            Q = Q_tables[i]
            
            # check out my genious discretization idea 
            # i compute the mean actions of all agents like this
            #a_bar_discr = A_vals[np.argmin(abs(np.mean(actions) - np.array(A_vals)))]
            # no wait... probably like this:
            pi_bar = sum(pi[mu]/mu for mu in A_vals)
            a_bar_discr = int(A_vals[np.argmin(abs(pi_bar - np.array(A_vals)))])
            
            mean_actions[i] = a_bar_discr
            
            # 3-d Q-table
            
            a_idx = A_vals.index(actions[i])
            s_prev = prev_state
            s_next = buffers[i]

            Q[s_prev, a_idx, a_bar_discr] += alpha * (
                cost + gamma * np.min(Q[s_next]) - Q[s_prev, a_idx, a_bar_discr]
)
            # was like this
            #Q[prev_state][A_vals.index(actions[i])][a_bar_discr] += alpha * (
            #cost + gamma * np.min(Q[buffers[i]]) - Q[prev_state][A_vals.index(actions[i])]
            #)
            
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
        den = max(1, int(active_users.sum()))
        shannon_cap_cum_all.append(shannon_cap_cum_current_step/den) # I replaced len(active_users) by N_agents 
        cumulative_cost_q_all.append(cumulative_cost_q_current_step/den)
        
        coverage.append(sum_transmitted/den)
        
        
        if n % env.n_steps_plot == 0:
            for i in range(tracked_agent):
                q_table_history[i].append(Q_tables[i].copy())
                
                action_every_agent[i].append(actions[i])
            
    print('Q-learning cost:',  np.mean(cumulative_cost_q_all))
    print('Q-learning buffers', np.sum(buffers)/n_steps/N_agents)
    
    delay = tracker.avg_delay_per_slot
    
        
    return cumulative_cost_q_all, shannon_cap_cum_all, q_table_history, delay, losses,  coverage, np.mean(coverage), np.mean(shannon_cap_cum_all), action_every_agent, buffer_every_agent
    #return cumulative_cost_q_all


################################ Q-learning ends
