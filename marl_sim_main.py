from marl_setting import Env
from marl_algs import DQN, Q_learning, Myopic, Myopic_test
import matplotlib.pyplot as plt
from plotting import make_plots_3_vals, plot_norm_new, plot_norm_one_agent, write_data, plot_all_p, plot_bounded_voronoi, plot_voronoi_cells
import numpy as np
from scipy.spatial import Voronoi, Delaunay

env = Env()

####3 plot Voronoi
#plot_bounded_voronoi(Voronoi(env.positions), env.positions,  env.user_locations, 'delaunay.png')
#plot_voronoi_cells(env.positions, env.L, 'bs-users-voronoi.png')

aa = 1

DQN_sim = DQN.DQN_sim
Q_sim = Q_learning.Q_sim
Q_greedy_try2 = Myopic.Greedy
# this is just a simulation with pre-defined vectors of mu. I want to see how good can agents actually perform with these actions
sim_test = Myopic_test.Greedy


Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, Q_cov_mean, Q_cap_mean, q_actions, q_buffs = Q_sim(env, 0.2, 50, 212, 'Q-diff-0.png', 10000)
       

# k = 30, function of p
def compute_for_all_p(K_buff, n, count_after):
    all_p = np.linspace(0.1, 0.9, 9)
    cov_p_Q = []
    cap_p_Q = []
    loss_p_Q = []
    buff_p_Q = []

    cov_p_greedy = []
    cap_p_greedy = []
    loss_p_greedy = []
    buff_p_greedy = []
    
    cov_p_NN = []
    cap_p_NN = []
    loss_p_NN = []
    buff_p_NN = []
    #K_buff = 20
    for cur_p in all_p:
        
        # Q-learning
        Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, Q_cov_mean, Q_cap_mean, q_actions, q_buffs = Q_sim(env, cur_p, K_buff, 212, 'Q-diff-0.png', 10000)
        # greedy
        greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, greedy_cov_mean, greedy_cap_mean, greedy_actions, greedy_buffs = Q_greedy_try2(env, cur_p, K_buff, 212, 'greedy-diff-0.png', 5000)
        # DQN
        NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, NN_cov_mean, NN_cap_mean, nn_train_loss, cost_nn_all_agents, cap_nn_all_agents, act_all_agents, buff_all_agents  = DQN_sim(env, count_after, cur_p, K_buff, 212, 'Q-diff-0.png', n)

        cov_p_Q.append(Q_cov_mean)
        cap_p_Q.append(Q_cap_mean)
        loss_p_Q.append(np.array(Q_loss)[5000:].mean())
        buff_p_Q.append(np.array(q_buffs).mean(axis=0)[5000:].mean())
        
        cov_p_greedy.append(greedy_cov_mean)
        cap_p_greedy.append(greedy_cap_mean)
        loss_p_greedy.append(np.array(greedy_loss)[3000:].mean())
        buff_p_greedy.append(np.array(greedy_buffs).mean(axis=0)[3000:].mean())
        
        cov_p_NN.append(NN_cov_mean)
        cap_p_NN.append(NN_cap_mean)
        loss_p_NN.append(np.array(NN_loss)[count_after:].mean())
        buff_p_NN.append(np.array(buff_all_agents).mean(axis=0)[count_after:].mean())
        
    return cov_p_Q, cap_p_Q, loss_p_Q, buff_p_Q, cov_p_greedy, cap_p_greedy, loss_p_greedy, buff_p_greedy, cov_p_NN, cap_p_NN, loss_p_NN, buff_p_NN
    #return cov_p_greedy, cap_p_greedy, loss_p_greedy, buff_p_greedy
    

#cov_p_greedy, cap_p_greedy, loss_p_greedy, buff_p_greedy = compute_for_all_p(20, 20000, 10000)


#### K = 20, all p
# cov_p_Q, cap_p_Q, loss_p_Q, buff_p_Q, cov_p_greedy, cap_p_greedy, loss_p_greedy, buff_p_greedy, cov_p_NN, cap_p_NN, loss_p_NN, buff_p_NN = compute_for_all_p(20, 40000, 30000)
# np.savez("for_all_p_K20.npz", cov_p_Q=cov_p_Q, cap_p_Q=cap_p_Q, loss_p_Q=loss_p_Q, buff_p_Q=buff_p_Q, 
#          cov_p_greedy=cov_p_greedy, cap_p_greedy=cap_p_greedy, loss_p_greedy=loss_p_greedy, buff_p_greedy=buff_p_greedy,
#          cov_p_NN=cov_p_NN, cap_p_NN=cap_p_NN, loss_p_NN =loss_p_NN, buff_p_NN=buff_p_NN)

# data_all_p = np.load("for_all_p_K20.npz", allow_pickle=False)
# cov_p_Q_K20 =data_all_p['cov_p_Q']
# cap_p_Q_K20=data_all_p['cap_p_Q'] 
# loss_p_Q_K20 = data_all_p['loss_p_Q']
# buff_p_Q_K20 = data_all_p['buff_p_Q']

# cov_p_greedy_K20=data_all_p['cov_p_greedy']
# cap_p_greedy_K20=data_all_p['cap_p_greedy']
# loss_p_greedy_K20 = data_all_p['loss_p_greedy']
# buff_p_greedy_K20 = data_all_p['buff_p_greedy']

# cov_p_NN_K20=data_all_p['cov_p_NN']
# cap_p_NN_K20=data_all_p['cap_p_NN']
# loss_p_NN_K20 = data_all_p['loss_p_NN']
# buff_p_NN_K20 = data_all_p['buff_p_NN']


# plot_all_p('cov-cap-K20', 'loss-buff-K20',
#                #####3 coverage and capacity
#                cov_p_Q_K20, cov_p_greedy_K20, cov_p_NN_K20,
#                cap_p_Q_K20, cap_p_greedy_K20, cap_p_NN_K20,
#                ######### loss and buffer
#                loss_p_Q_K20, loss_p_greedy_K20, loss_p_NN_K20,
#                buff_p_Q_K20, buff_p_greedy_K20, buff_p_NN_K20
#                )


### K = 100, all p
cov_p_Q, cap_p_Q, loss_p_Q, buff_p_Q, cov_p_greedy, cap_p_greedy, loss_p_greedy, buff_p_greedy, cov_p_NN, cap_p_NN, loss_p_NN, buff_p_NN = compute_for_all_p(100, 40000, 30000)
np.savez("for_all_p_K100.npz", cov_p_Q=cov_p_Q, cap_p_Q=cap_p_Q, loss_p_Q=loss_p_Q, buff_p_Q=buff_p_Q, 
         cov_p_greedy=cov_p_greedy, cap_p_greedy=cap_p_greedy, loss_p_greedy=loss_p_greedy, buff_p_greedy=buff_p_greedy,

         cov_p_NN=cov_p_NN, cap_p_NN=cap_p_NN, loss_p_NN =loss_p_NN, buff_p_NN=buff_p_NN)

data_all_p = np.load("for_all_p_K100.npz", allow_pickle=False)
cov_p_Q_K100 =data_all_p['cov_p_Q']
cap_p_Q_K100=data_all_p['cap_p_Q'] 
loss_p_Q_K100 = data_all_p['loss_p_Q']
buff_p_Q_K100 = data_all_p['buff_p_Q']

cov_p_greedy_K100=data_all_p['cov_p_greedy']
cap_p_greedy_K100=data_all_p['cap_p_greedy']
loss_p_greedy_K100 = data_all_p['loss_p_greedy']
buff_p_greedy_K100 = data_all_p['buff_p_greedy']

cov_p_NN_K100=data_all_p['cov_p_NN']
cap_p_NN_K100=data_all_p['cap_p_NN']
loss_p_NN_K100 = data_all_p['loss_p_NN']
buff_p_NN_K100 = data_all_p['buff_p_NN']


plot_all_p('cov-cap-K100', 'loss-buff-K100',
           #####3 coverage and capacity
           cov_p_Q_K100, cov_p_greedy_K100, cov_p_NN_K100,
           cap_p_Q_K100, cap_p_greedy_K100, cap_p_NN_K100,
           ######### loss and buffer
           loss_p_Q_K100, loss_p_greedy_K100, loss_p_NN_K100,
           buff_p_Q_K100, buff_p_greedy_K100, buff_p_NN_K100
)

aa = 1

# large buffer, heavy traffic
### I changed buffer penalty to 0.3 maybe it will react better
############3 p = 0.5, K = 5
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _, q_actions, _ = Q_sim(env, 0.6, 100, 212, 'Q-diff-0.png', 40000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions, _ = Q_greedy_try2(env, 0.6, 100, 212, 'greedy-diff-0.png', 40000)
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _, nn_train_loss, cost_nn_all_agents, cap_nn_all_agents, act_all_agents, buff_all_agents  = DQN_sim(env, 1, 0.6, 100, 212, 'Q-diff-0.png', 40000)

# write data NN
write_data(NN_hist_all, act_all_agents,"experiment-p06-K100.npz",NN_cost=NN_cost, NN_cap=NN_cap, NN_delay=NN_delay, NN_loss=NN_loss, NN_coverage=NN_coverage, nn_train_loss=nn_train_loss, cost_nn_all_agents=cost_nn_all_agents, cap_nn_all_agents=cap_nn_all_agents, buff_all_agents = buff_all_agents)
# Write dataQ-learning
#write_data(greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions)
#
#np.savez("actions_p08_K5.npz", q_actions=np.array(q_actions)[:, env.N_agents-1], greedy_actions=np.array(greedy_actions)[:, env.N_agents-1], NN_actions = np.array(act_all_agents)[:, env.N_agents-1])


make_plots_3_vals('loss-delay-p06-K100.png', 'cost-shannon-p06-K100.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)




######3 low traffic, small buffer size
# p = 0.2, K = 20
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _, q_actions, _ = Q_sim(env, 0.2, 20, 212, 'Q-diff-0.png', 40000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions, _ = Q_greedy_try2(env, 0.2, 20, 212, 'greedy-diff-0.png', 40000)
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _, nn_train_loss, cost_nn_all_agents, cap_nn_all_agents, act_all_agents, buff_all_agents  = DQN_sim(env, 1, 0.2, 20, 212, 'Q-diff-0.png', 40000)

# write data NN
write_data(NN_hist_all, act_all_agents,"experiment-p02-K20.npz",NN_cost=NN_cost, NN_cap=NN_cap, NN_delay=NN_delay, NN_loss=NN_loss, NN_coverage=NN_coverage, nn_train_loss=nn_train_loss, cost_nn_all_agents=cost_nn_all_agents, cap_nn_all_agents=cap_nn_all_agents, buff_all_agents = buff_all_agents)
# Write dataQ-learning
#write_data(greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions)
#
#np.savez("actions_p08_K5.npz", q_actions=np.array(q_actions)[:, env.N_agents-1], greedy_actions=np.array(greedy_actions)[:, env.N_agents-1], NN_actions = np.array(act_all_agents)[:, env.N_agents-1])


make_plots_3_vals('loss-delay-p02-20.png', 'cost-shannon-p02-K20.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)


aa = 1

# just a test, p  =0.7. K = 20
#greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, all_buffs = sim_test(env, 0.7, 20, 212, 'greedy-diff-0.png', 10000)

        # DQN
#NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, NN_cov_mean, NN_cap_mean, nn_train_loss, cost_nn_all_agents, cap_nn_all_agents, act_all_agents, buff_all_agents  = DQN_sim(env, 1000, 0.7, 100, 212, 'Q-diff-0.png', 16000)




############3 p = 0.3, K = 30

# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _, q_actions, _ = Q_sim(env, 0.7, 100, 212, 'Q-diff-0.png', 30000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions, _ = Q_greedy_try2(env, 0.7, 100, 212, 'greedy-diff-0.png', 30000)
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _, nn_train_loss, cost_nn_all_agents, cap_nn_all_agents, act_all_agents, buff_all_agents  = DQN_sim(env, 1, 0.7, 100, 212, 'Q-diff-0.png', 30000)

# write data NN
write_data(NN_hist_all, act_all_agents,"experiment-p06-K100.npz",NN_cost=NN_cost, NN_cap=NN_cap, NN_delay=NN_delay, NN_loss=NN_loss, NN_coverage=NN_coverage, nn_train_loss=nn_train_loss, cost_nn_all_agents=cost_nn_all_agents, cap_nn_all_agents=cap_nn_all_agents, buff_all_agents = buff_all_agents)
# Write dataQ-learning
#write_data(greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions)

np.savez("actions_p08_K5.npz", q_actions=np.array(q_actions)[:, env.N_agents-1], greedy_actions=np.array(greedy_actions)[:, env.N_agents-1], NN_actions = np.array(act_all_agents)[:, env.N_agents-1])


make_plots_3_vals('loss-delay-p03-K30.png', 'cost-shannon-p03-K30.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)




plt.plot(cap_p_Q_K20, label = 'Q-learning')
plt.plot(cap_p_greedy_K20, label = 'greedy')
plt.plot(cap_p_NN_K20, label = 'DQN')
plt.legend(loc = 'best')
plt.show()
############3 p = 0.5, K = 5

# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _, q_actions = Q_sim(env, 0.8, 5, 212, 'Q-diff-0.png', 30000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions = Q_greedy_try2(env, 0.8, 5, 212, 'greedy-diff-0.png', 30000)

# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _, nn_train_loss, cost_nn_all_agents, cap_nn_all_agents, act_all_agents, buff_all_agents  = DQN_sim(env, 0.8, 5, 212, 'Q-diff-0.png', 30000)

# write data NN
write_data(NN_hist_all, act_all_agents,"experiment-p08-K5.npz",NN_cost=NN_cost, NN_cap=NN_cap, NN_delay=NN_delay, NN_loss=NN_loss, NN_coverage=NN_coverage, nn_train_loss=nn_train_loss, cost_nn_all_agents=cost_nn_all_agents, cap_nn_all_agents=cap_nn_all_agents, buff_all_agents = buff_all_agents)
# Write dataQ-learning
#write_data(greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions)

np.savez("actions_p08_K5.npz", q_actions=np.array(q_actions)[:, env.N_agents-1], greedy_actions=np.array(greedy_actions)[:, env.N_agents-1], NN_actions = np.array(act_all_agents)[:, env.N_agents-1])
data_actions = np.load("actions_p08_K5.npz", allow_pickle=False)


greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, all_buffs = sim_test(data_actions['greedy_actions'], env, 0.8, 5, 212, 'greedy-diff-0.png', 10000)
q_cost, q, q_delay, q_loss, q_coverage, _, _, all_buffs = sim_test(data_actions['q_actions'], env, 0.8, 5, 212, 'q-diff-0.png', 10000)
NN_cost, NN, NN_delay, NN_loss, NN_coverage, _, _, all_buffs = sim_test(data_actions['NN_actions'], env, 0.8, 5, 212, 'NN-diff-0.png', 10000)


make_plots_3_vals('loss-delay-p08-K5.png', 'cost-shannon-p06-K5.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)

############3 p = 0.3, K = 5

# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _, q_actions = Q_sim(env, 0.3, 5, 212, 'Q-diff-0.png', 30000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _, greedy_actions = Q_greedy_try2(env, 0.3, 5, 212, 'greedy-diff-0.png', 30000)

# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _, nn_train_loss, cost_nn_all_agents, cap_nn_all_agents, act_all_agents, buff_all_agents  = DQN_sim(env, 0.3, 5, 212, 'Q-diff-0.png', 30000)

write_data(NN_hist_all, act_all_agents,"experiment-p08-K5.npz",NN_cost=NN_cost, NN_cap=NN_cap, NN_delay=NN_delay, NN_loss=NN_loss, NN_coverage=NN_coverage, nn_train_loss=nn_train_loss, cost_nn_all_agents=cost_nn_all_agents, cap_nn_all_agents=cap_nn_all_agents, buff_all_agents = buff_all_agents)


make_plots_3_vals('loss-delay-p03-K5.png', 'cost-shannon-p03-K5.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)



############3 p = 1, K = 30
# DQN

   
# read from file:
dat = np.load("experiment-p07-K10.npz", allow_pickle=False)

NN_hist_all = dat["Q_function_all_agents"]
act_all_agents = dat["actions_all_agents"]

NN_cost             = dat["NN_cost"]
NN_cap              = dat["NN_cap"]
NN_delay            = dat["NN_delay"]
NN_loss             = dat["NN_loss"]
NN_coverage         = dat["NN_coverage"]
nn_train_loss       = dat["nn_train_loss"]
cost_nn_all_agents  = dat["cost_nn_all_agents"]
cap_nn_all_agents   = dat["cap_nn_all_agents"]
buff_all_agents     = dat["buff_all_agents"]


avg_across_agents = buff_all_agents.mean(axis=0)

NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _, nn_train_loss, cost_nn_all_agents, cap_nn_all_agents, act_all_agents, buff_all_agents = DQN_sim(env, 0.7, 20, 212, 'Q-diff-0.png', 30000)

write_data(
    NN_hist_all,
    act_all_agents,
    "experiment-p07-K20.npz",
    NN_cost=NN_cost,
    NN_cap=NN_cap,
    NN_delay=NN_delay,
    NN_loss=NN_loss,
    NN_coverage=NN_coverage,
    nn_train_loss=nn_train_loss,
    cost_nn_all_agents=cost_nn_all_agents,
    cap_nn_all_agents=cap_nn_all_agents,
    buff_all_agents = buff_all_agents
)


plt.plot(NN_cost)
#write_data(NN_hist_all, act_all_agents, 'experiment-p1-k30.npz')



plot_norm_one_agent(NN_hist_all[0], 100)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 1, 30, 51, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 1, 30, 51, 'greedy-diff-0.png', 20000)

 
    
# plot cost each agent
def plot_shit_all_agents(cost_nn_all_agents):
    import numpy as np
    for cost_cur_agent in cost_nn_all_agents:
        cost_cur_agent =  np.array(cost_cur_agent)
        n = len(cost_cur_agent)
        ks = np.arange(0, n - 1, 100)           # starting indices: 0, 100, 200, ... where k+1 < n
        diffs = np.abs(cost_cur_agent[ks + 1] - cost_cur_agent[ks])
        plt.plot(ks + 1, diffs)  

    plt.xlabel("time slot (second index in each pair)")
    plt.ylabel("Δ cost (slot k+1 − slot k)")
    plt.show()


#plot cap each agent
make_plots_3_vals('loss-delay-p1-K30.png', 'cost-shannon-p1-K30.png',
              Q_cap, Q_cost, Q_delay, Q_loss,
              greedy_cost, greedy, greedy_delay, greedy_loss,
              NN_cap, NN_cost, NN_delay, NN_loss)

# plot difference of norms to check convergence
plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p01-K30.png')

aa = 1

############3 p = 0.7, K = 100
# DQN
#NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.7, 100, 51, 'Q-diff-0.png', 20000)
# Q-learning
#Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.7, 100, 51, 'Q-diff-0.png', 20000)
# greedy
#greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.7, 100, 15, 'greedy-diff-0.png', 20000)

#make_plots_3_vals('loss-delay-p07-K100.png', 'cost-shannon-p07-K100.png',
#               Q_cap, Q_cost, Q_delay, Q_loss,
#               greedy_cost, greedy, greedy_delay, greedy_loss,
#               NN_cap, NN_cost, NN_delay, NN_loss)


# plot difference of norms to check convergence
#plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p07-K100.png')















############3 p = 0.5, K = 50
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.5, 50, 15, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.5, 50, 15, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.5, 50, 15, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p05-K50.png', 'cost-shannon-p05-K50.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)

plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p05-K50.png')


############3 p = 0.7, K = 50
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.7, 50, 15, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.7, 50, 15, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.7, 50, 15, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p07-K50.png', 'cost-shannon-p07-K50.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)


plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p07-K50.png')
############3 p = 0.9, K = 50 Begin here !!!!!!!!!!!
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.9, 50, 15, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.9, 50, 15, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.9, 50, 15, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p09-K50.png', 'cost-shannon-p09-K50.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)


plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p09-K50.png')

# Done! Change buffer size now, keep p = 0.6
############3 p = 0.6, K = 10
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.6, 10, 15, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.6, 10, 15, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.6, 10, 15, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p06-K10.png', 'cost-shannon-p06-K10.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)

plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p06-K10.png')

############3 p = 0.6, K = 30
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.6, 30, 15, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.6, 30, 15, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.6, 30, 15, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p06-K30.png', 'cost-shannon-p06-K30.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)


plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p06-K30.png')
############3 p = 0.6, K = 100
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.6, 100, 15, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.6, 100, 15, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.6, 100, 15, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p06-K100.png', 'cost-shannon-p06-K100.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)


plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p06-K100.png')
############3 p = 0.6, K = 200
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.6, 200, 15, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.6, 200, 15, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.6, 200, 15, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p06-K200.png', 'cost-shannon-p06-K200.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)

plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p06-K200.png')



aa = 1
