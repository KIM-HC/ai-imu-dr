import os
from termcolor import cprint
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import torch
import pickle

def results_filter():
    sets = ["git_model", "learned_1",
            "not_learned_1", "not_learned_2"]
    path_sets = "../result_backup"
    path_results = "../results"
    path_data_save = "../data"

    datasets = []
    for data_name in os.listdir(path_data_save):
        datasets += [data_name[:-2]]  # take just name, remove the ".p"

    for i in range(0, len(datasets)):
        p_align = []
        p_og = []
        skip_this_data = False
        plt.close('all')
        dataset_name = datasets[i]
        file_name_gt = os.path.join(path_data_save, datasets[i]+".p")
        with open(file_name_gt, "rb") as file_pi_gt:
            pickle_dict_gt = pickle.load(file_pi_gt)
        t = pickle_dict_gt['t']
        t = (t - t[0]).numpy()
        p_gt = pickle_dict_gt['p_gt']
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        print("Total sequence time: {:.2f} s".format(t[-1]))

        ## this part for each sets
        for set_name in sets:
            file_name = os.path.join(path_sets, set_name, dataset_name + "_filter.p")
            if not os.path.exists(file_name):
                print('No result for ' + dataset_name)
                skip_this_data = True
                break

            with open(file_name, "rb") as file_pi:
                mondict = pickle.load(file_pi)
            p = mondict['p']
            Rot_align, t_align, _ = umeyama_alignment(p_gt[:, :3].T, p[:, :3].T)
            p_og.append(p)
            p_align.append((Rot_align.T.dot(p[:, :3].T)).T - Rot_align.T.dot(t_align))

        if skip_this_data:
            continue

        # position in plan
        name_list = ["ground-truth trajectory"] + sets
        fig3, ax3 = plt.subplots(figsize=(20, 10))
        ax3.plot(p_gt[:, 0], p_gt[:, 1])
        for sn in range(len(sets)):
            ax3.plot(p_og[sn][:, 0], p_og[sn][:, 1])
        ax3.axis('equal')
        ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")
        ax3.grid()
        ax3.legend(name_list)
        figs = [fig3]

        plot_aligned = False
        if plot_aligned:
            # position in plan after alignment
            fig4, ax4 = plt.subplots(figsize=(20, 10))
            ax4.plot(p_gt[:, 0], p_gt[:, 1])
            for sn in range(len(sets)):
                ax4.plot(p_align[sn][:, 0], p_align[sn][:, 1])
            ax4.axis('equal')
            ax4.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Aligned position on $xy$")
            ax4.grid()
            ax4.legend(name_list)
            figs += [fig4]

        figs_name = ["position_xy", "position_xy_aligned"]
        for l, fig in enumerate(figs):
            fig_name = figs_name[l]
            fig.savefig(os.path.join(path_results, fig_name + ".png"))

        plt.show(block=True)

if __name__ == '__main__':
    results_filter()
    


