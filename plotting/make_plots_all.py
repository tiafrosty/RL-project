import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay

from scipy.integrate import quad



def smooth(x, w=25):
    x = np.asarray(x, float)
    if w <= 1: 
        return x
    k = np.ones(w)
    y = np.convolve(x, k, mode='same')                # sum in window
    n = np.convolve(np.ones_like(x), k, mode='same')  # number of real points
    return y / n              



def make_plots_3_vals(figname_loss_delay, figname_cost_cap,
               Q_cap, Q_cost, Q_delay, Q_loss,
               greedy_cost, greedy, greedy_delay, greedy_loss,
               NN_cap, NN_cost, NN_delay, NN_loss):
    
    # myopic, Q-learn, QDN
    cols = {'myopic':'blue', 'q-learn':'black', 'NN':'purple'}
  
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), sharex=True, constrained_layout=True
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), sharex=True, constrained_layout=True
    )

    # -------- subplot 1: loss rate --------
    for ax, y_raw, y_smooth, title, ylab in [
        (ax1, Q_loss, smooth(Q_loss, 21),
        "Per-slot loss probability", "Loss rate"),
        (ax2, Q_delay, smooth(Q_delay, 21),
        "Per-slot average delay", "Average delay"),
    ]:
        # Raw 
        ax.plot(y_raw, linewidth=1.2, alpha=0.35, color = cols['q-learn'])
        ax.plot(greedy_loss if ax is ax1 else greedy_delay,
                linewidth=1.2, alpha=0.35, color = cols["myopic"])
        ax.plot(NN_loss if ax is ax1 else NN_delay,
                linewidth=1.2, alpha=0.35, color = cols['NN'])
        # Smoothed (bold)
        ax.plot(y_smooth, linewidth=2.2, label="Q-learning", color = cols["q-learn"])
        base_sm = smooth(greedy_loss, 21) if ax is ax1 else smooth(greedy_delay, 21)
        ax.plot(base_sm, linewidth=2.2, label="Myopic", color = cols["myopic"])
        # DQN
        base_nn_sm = smooth(NN_loss, 21) if ax is ax1 else smooth(NN_delay, 21)
        ax.plot(base_nn_sm, linewidth=2.2, label="DQN", color = cols["NN"])

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time n", fontsize=12)
        ax.set_ylabel(ylab, fontsize=12)
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=False))
        ax.minorticks_on()

    # y-lims tight to data
    ax1.set_ylim(0, max(np.max(Q_loss), np.max(greedy_loss), np.max(NN_loss)) * 1.05)
    ax2.set_ylim(0, max(np.max(Q_delay), np.max(greedy_delay), np.max(NN_delay)) * 1.05)

    for ax in (ax1, ax2):
        ax.legend(loc = 'best', fontsize = 13)
        ax.tick_params(axis='both', labelsize=11)

    # Optional: save high-res for papers/slides
    plt.savefig(figname_loss_delay, dpi=300, bbox_inches="tight")
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), sharex=True, constrained_layout=True
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), sharex=True, constrained_layout=True
    )

    # -------- subplot 1: loss rate --------
    for ax, y_raw, y_smooth, title, ylab in [
        (ax1, Q_cost, smooth(Q_cost, 21),
        "Per-slot cost", "Cost"),
        (ax2, Q_cap, smooth(Q_cap, 21),
        "Per-slot Shannon capacity", "EC(p)"),
    ]:
        # Raw (faint) – no markers
        ax.plot(y_raw, linewidth=1.2, alpha=0.35, color = cols["q-learn"])
        ax.plot(greedy_cost if ax is ax1 else greedy,
                linewidth=1.2, alpha=0.35, color = cols["myopic"])
        # DQN raw
        ax.plot(NN_cost if ax is ax1 else NN_cap,
                linewidth=1.2, alpha=0.35, color = cols["NN"])

        # Smoothed (bold)
        ax.plot(y_smooth, linewidth=2.2, label="Q-learning", color = cols["q-learn"])
        base_sm = smooth(greedy_cost, 21) if ax is ax1 else smooth(greedy, 21)
        ax.plot(base_sm, linewidth=2.2, label="Myopic", color = cols["myopic"])
        # DQN smoothed
        base_nn_sm = smooth(NN_cost, 21) if ax is ax1 else smooth(NN_cap, 21)
        ax.plot(base_nn_sm, linewidth=2.2, label="DQN", color = cols["NN"])

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time n", fontsize=12)
        ax.set_ylabel(ylab, fontsize=12)
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=False))
        ax.minorticks_on()

    # y-lims tight to data
    ymin = min(np.min(Q_cost), np.min(greedy_cost), np.min(NN_cost)) - 1
    ymax = max(np.max(Q_cost), np.max(greedy_cost), np.max(NN_cost)) + 1
    ax1.set_ylim(ymin, ymax)
    #ax1.set_ylim(min(np.max(greedy_cost), np.max(Q_cost))*1.05, max(np.max(greedy_cost), np.max(Q_cost)) * 1.05)
    ax2.set_ylim(0, max(np.max(Q_cap), np.max(greedy), np.max(NN_cap)) * 1.05)

    # Legends outside to avoid covering data
    for ax in (ax1, ax2):
        ax.legend(loc = 'best', fontsize = 13)
        ax.tick_params(axis='both', labelsize=11)

    # Optional: save high-res for papers/slides
    plt.savefig(figname_cost_cap, dpi=300, bbox_inches="tight")
    


def write_data(
        NN_hist_all,
        act_all_agents,
        filename,
        NN_cost=None,
        NN_cap=None,
        NN_delay=None,
        NN_loss=None,
        NN_coverage=None,
        nn_train_loss=None,
        cost_nn_all_agents=None,
        cap_nn_all_agents=None,
        buff_all_agents = None,
    ):
    payload = {
            "Q_function_all_agents": np.asarray(NN_hist_all),
            "actions_all_agents":    np.asarray(act_all_agents),
        }

    extras = {
            "NN_cost": NN_cost,
            "NN_cap": NN_cap,
            "NN_delay": NN_delay,
            "NN_loss": NN_loss,
            "NN_coverage": NN_coverage,
            "nn_train_loss": nn_train_loss,
            "cost_nn_all_agents": cost_nn_all_agents,
            "cap_nn_all_agents": cap_nn_all_agents,
            'buff_all_agents': buff_all_agents,
        }

        # add only provided metrics
    for k, v in extras.items():
        if v is not None:
                payload[k] = np.asarray(v)

    np.savez(filename, **payload)
    
def compute_diff_norm_one_agent(q_tables_dqn):
    Q_norm_diffs = []
    q_table_history = q_tables_dqn
    for t in range(1, len(q_table_history)):
        Q_prev = q_table_history[t - 1]
        Q_curr = q_table_history[t]
        diff = np.linalg.norm(Q_curr - Q_prev)  # Frobenius norm
        Q_norm_diffs.append(diff)
    return Q_norm_diffs
    
    
def plot_norm_one_agent(q_tables_dqn, n_steps_plot):
    # visualize Q-table evolution via norm diffs for two methods side-by-side
    _, ax2 = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    Q_norm_diffs = []
    q_table_history = q_tables_dqn
    for t in range(1, len(q_table_history)):
        Q_prev = q_table_history[t - 1]
        Q_curr = q_table_history[t]
        diff = np.linalg.norm(Q_curr - Q_prev)  # Frobenius norm
        Q_norm_diffs.append(diff)

    if len(Q_norm_diffs) > 0:
        ax2.plot(np.arange(1, len(Q_norm_diffs) + 1) * n_steps_plot,
                Q_norm_diffs, '-', linewidth=2.0)

    ax2.set_title("DQN", fontsize=16)
    ax2.set_xlabel("Time n", fontsize=12)
    ax2.set_ylabel(r"$||Q_n - Q_{n-1}||$", fontsize=12)
    ax2.grid(True, which="major", alpha=0.25)
    ax2.grid(True, which="minor", alpha=0.12)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=False))
    ax2.minorticks_on()
    ax2.tick_params(axis='both', labelsize=11)

    
    
def plot_norm_new(q_tables_tab, q_tables_dqn, n_steps_plot, figname):
    # visualize Q-table evolution via norm diffs for two methods side-by-side
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), sharex=True, constrained_layout=True
    )

    # ---- Subplot 1: Tabular Q-learning ----
    for i in range(len(q_tables_tab)):
        Q_norm_diffs = []
        q_table_history = q_tables_tab[i]
        for t in range(1, len(q_table_history)):
            Q_prev = q_table_history[t - 1]
            Q_curr = q_table_history[t]
            diff = np.linalg.norm(Q_curr - Q_prev)  # Frobenius norm
            Q_norm_diffs.append(diff)

        if len(Q_norm_diffs) > 0:
            ax1.plot(np.arange(1, len(Q_norm_diffs) + 1) * n_steps_plot,
                     Q_norm_diffs, '-', linewidth=2.0)

    ax1.set_title("Tabular Q-learning", fontsize=16)
    ax1.set_xlabel("Time n", fontsize=12)
    ax1.set_ylabel(r"$||Q_n - Q_{n-1}||$", fontsize=12)
    ax1.grid(True, which="major", alpha=0.25)
    ax1.grid(True, which="minor", alpha=0.12)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=False))
    ax1.minorticks_on()
    ax1.tick_params(axis='both', labelsize=11)

    # ---- Subplot 2: DQN ----
    for i in range(len(q_tables_dqn)):
        Q_norm_diffs = []
        q_table_history = q_tables_dqn[i]
        for t in range(1, len(q_table_history)):
            Q_prev = q_table_history[t - 1]
            Q_curr = q_table_history[t]
            diff = np.linalg.norm(Q_curr - Q_prev)  # Frobenius norm
            Q_norm_diffs.append(diff)

        if len(Q_norm_diffs) > 0:
            ax2.plot(np.arange(1, len(Q_norm_diffs) + 1) * n_steps_plot,
                     Q_norm_diffs, '-', linewidth=2.0)

    ax2.set_title("DQN", fontsize=16)
    ax2.set_xlabel("Time n", fontsize=12)
    ax2.set_ylabel(r"$||Q_n - Q_{n-1}||$", fontsize=12)
    ax2.grid(True, which="major", alpha=0.25)
    ax2.grid(True, which="minor", alpha=0.12)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=False))
    ax2.minorticks_on()
    ax2.tick_params(axis='both', labelsize=11)

    plt.savefig(figname, dpi=300, bbox_inches="tight")


def plot_norm(q_tables_tab, q_tables_dqn, n_steps_plot, figname):
    # visualize Q-table evolution via norm diffs for two methods side-by-side
    plt.figure(figsize=(10, 5))

    # ---- Subplot 1: Tabular Q-learning ----
    plt.subplot(1, 2, 1)
    for i in range(len(q_tables_tab)):
        Q_norm_diffs = []
        q_table_history = q_tables_tab[i]
        for t in range(1, len(q_table_history)):
            Q_prev = q_table_history[t - 1]
            Q_curr = q_table_history[t]
            diff = np.linalg.norm(Q_curr - Q_prev)  # Frobenius norm by default
            Q_norm_diffs.append(diff)

        if len(Q_norm_diffs) > 0:
            plt.plot(np.arange(1, len(Q_norm_diffs) + 1) * n_steps_plot, Q_norm_diffs, '-', linewidth=2.0)

    plt.xlabel('Time step', fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel(r'$||Q_n - Q_{n-1}||$', fontsize=12)
    plt.title(r'Norm of the difference (Q_n - Q_{n-1})', fontsize=16)
    plt.grid(True)

    # ---- Subplot 2: DQN ----
    plt.subplot(1, 2, 2)
    for i in range(len(q_tables_dqn)):
        Q_norm_diffs = []
        q_table_history = q_tables_dqn[i]
        for t in range(1, len(q_table_history)):
            Q_prev = q_table_history[t - 1]
            Q_curr = q_table_history[t]
            diff = np.linalg.norm(Q_curr - Q_prev)  # Frobenius norm by default
            Q_norm_diffs.append(diff)

        if len(Q_norm_diffs) > 0:
            plt.plot(np.arange(1, len(Q_norm_diffs) + 1) * n_steps_plot, Q_norm_diffs, '-', linewidth=2.0)

    plt.xlabel('Time step', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel(r'$||Q_n - Q_{n-1}||$', fontsize=12)
    plt.title(r'Norm of the difference (Q_n - Q_{n-1})', fontsize=16)
    plt.grid(True)

    # Keep the same layout/save behavior
    #plt.tight_layout()
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    
    
    
def run_q_and_greedy(filename, file_mode, n,
                     p_signal, K_buffer):

    # The code is calling a function `Q_sim` with the arguments `p_signal`, `K_buffer`, 2,
    # 'Q-diff-0.png', `n`, and `estimate_mean_fields`. The function is being called with these
    # arguments and the return values are being assigned to variables `Q_cost` and `Q_cap`.
    
    Q_cost, Q_cap = Q_sim(p_signal, K_buffer, 2, 'Q-diff-0.png', n)
    greedy_cost, greedy_cap = Q_greedy_try2(p_signal, K_buffer, 2, '', n)
    
    with open(filename, file_mode) as f:
        f.write(f"\n# New experiment... \n # ...Done! \n")
        f.write(f"\n# params: p = {p_signal}, K = {K_buffer}, A = {A_vals}, n = {n}\n")
        f.write(f"\n# Q-learning results\n")
        f.write(f"\n# cost, reward, shannon\n")
        f.write(f"q_learning_cost = {Q_cost}\n")
        f.write(f"q_learning_reward = {list(np.array(Q_cost)*(-1))}\n")
        f.write(f"q_learning_shannon_cap = {Q_cap}\n")
        ### buffer, delay, loss
        f.write(f"# Greedy\n")
        # greedy
        f.write(f"greedy_cost = {greedy_cost}\n")
        f.write(f"greedy_cap = {greedy_cap}\n")
        
        
    return Q_cost, Q_cap, greedy_cost, greedy_cap



## Plotting Voronoi diagram (optional)
#vor = Voronoi(positions)
#fig = plt.figure(figsize=(6, 6))
## plot Delaunay connections
#plt.triplot(positions[:, 0], positions[:, 1], delaunay.simplices, color='gray')
## plot BS locations
#plt.plot(positions[:, 0], positions[:, 1], 'o', color='blue', markersize =  10)
## plot users locations
#plt.plot(user_locations[:, 0], user_locations[:, 1], 'o', color='red')
#plt.title("Voronoi-based BSs with Delaunay Neighbors", fontsize = 20)
#plt.xlim(0, L)
#plt.ylim(0, L)
#plt.gca().set_aspect('equal')
#plt.grid(True)
#plt.show()



# Plot Voronoi diagram with bounded cells (clipped to [0,1]^2 box)
def plot_bounded_voronoi(vor, points, users, box=[0, 1, 0, 1]):
    from shapely.geometry import Polygon, LineString
    from shapely.ops import clip_by_rect
    import matplotlib.patches as patches
    import matplotlib.collections as mcoll

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect('equal')
    ax.set_title("Voronoi Diagram of BSs with Users", fontsize=18)

    # Draw clipped Voronoi regions
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if not -1 in region and region != []:
            polygon = [vor.vertices[i] for i in region]
            poly = Polygon(polygon)
            clipped_poly = poly.intersection(Polygon([
                [box[0], box[2]], [box[1], box[2]],
                [box[1], box[3]], [box[0], box[3]]
            ]))
            if not clipped_poly.is_empty:
                patch = patches.Polygon(list(clipped_poly.exterior.coords), facecolor='lightgray', edgecolor='black', alpha=0.4)
                ax.add_patch(patch)

    # Plot Voronoi edges manually (just internal ones)
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            line = [vor.vertices[i] for i in simplex]
            ax.plot(*zip(*line), color='black', lw=1)

    # Plot BS positions
    ax.plot(points[:, 0], points[:, 1], 'o', color='blue', markersize=8, label='Base Stations')

    # Plot user positions
    ax.plot(users[:, 0], users[:, 1], '.', color='red', markersize=4, label='Users')

    ax.legend(loc='best', fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


