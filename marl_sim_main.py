from marl_setting import Env
from marl_algs import DQN, Q_learning, Myopic
import matplotlib.pyplot as plt
from plotting import make_plots_3_vals, plot_norm_new


env = Env()


DQN_sim = DQN.DQN_sim
Q_sim = Q_learning.Q_sim
Q_greedy_try2 = Myopic.Greedy

# just a test
############3 p = 0.6, K = 100
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _, train_loss_nn  = DQN_sim(env, 0.6, 100, 51, 'Q-diff-0.png', 10000)



############3 p = 0.2, K = 50
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.2, 50, 51, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.2, 50, 51, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.2, 50, 51, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p02-K50.png', 'cost-shannon-p02-K50.png',
              Q_cap, Q_cost, Q_delay, Q_loss,
              greedy_cost, greedy, greedy_delay, greedy_loss,
              NN_cap, NN_cost, NN_delay, NN_loss)

# plot difference of norms to check convergence
plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p02-K50.png')

############3 p = 0.7, K = 100
# DQN
NN_cost, NN_cap, NN_hist_all, NN_delay, NN_loss, NN_coverage, _, _ = DQN_sim(env, 0.7, 100, 51, 'Q-diff-0.png', 20000)
# Q-learning
Q_cost, Q_cap, q_hist_all, Q_delay, Q_loss, Q_coverage, _, _ = Q_sim(env, 0.7, 100, 51, 'Q-diff-0.png', 20000)
# greedy
greedy_cost, greedy, greedy_delay, greedy_loss, greedy_coverage, _, _ = Q_greedy_try2(env, 0.7, 100, 15, 'greedy-diff-0.png', 20000)

make_plots_3_vals('loss-delay-p07-K100.png', 'cost-shannon-p07-K100.png',
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss)


# plot difference of norms to check convergence
plot_norm_new(q_hist_all, NN_hist_all, 100, 'convergence-tab-nn-p07-K100.png')















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



aa = 1
