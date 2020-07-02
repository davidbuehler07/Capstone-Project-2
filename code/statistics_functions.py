import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
from scipy.stats import ttest_ind

def change_pitch_type(value, target):
    if value == target:
        return 1
    else:
        return 0
    
def bootstrap_reps(data, func):
    bs_sample = np.random.choice(data, size=len(data))
    return func(bs_sample)

def bootstrap_test(target_df, target):
    '''
    This function performs a bootstrap test on a target pitch (FB, BB, OS, OT) to determine the average score
    differential and see if that is statistically significant.
    '''
    
    df = target_df.copy()
    df['pitch_type'] = df.apply(lambda row:change_pitch_type(row['pitch_type'], target), axis=1)
    
    df_is = df[df.pitch_type == 1]
    df_not = df[df.pitch_type == 0]
    
    np.random.seed(47)
    N_rep = 1000

    is_bs_reps = np.empty(N_rep)
    not_bs_reps = np.empty(N_rep)
    
    for i in range(N_rep):
        is_bs_reps[i] = bootstrap_reps(df_is['score_diff'], np.mean)

    for i in range(N_rep):
        not_bs_reps[i] = bootstrap_reps(df_not['score_diff'], np.mean)

    print(f'95% percentile for pitches that are {target}: {np.percentile(is_bs_reps, [2.5, 97.5])}')
    print(f'95% percentile for pitches that are not {target}: {np.percentile(not_bs_reps, [2.5, 97.5])}')
    
    plt.hist(is_bs_reps, histtype='step', color='b')
    plt.title('Average score differential for pitches that are {}'.format(target))
    plt.xlabel('Average score differential')
    plt.ylabel('Number of occurences')
    plt.axvline(np.percentile(is_bs_reps, 2.5), linewidth=1, color='b', linestyle='--')
    plt.axvline(np.percentile(is_bs_reps, 97.5), linewidth=1, color='b', linestyle='--')
    plt.show()
    
    plt.hist(not_bs_reps, histtype='step', color='r')
    plt.title('Average score differential for pitches that are not {}'.format(target))
    plt.xlabel('Average score differential')
    plt.ylabel('Number of Occurences')
    plt.axvline(np.percentile(not_bs_reps, 2.5), linewidth=1, color='r', linestyle='--')
    plt.axvline(np.percentile(not_bs_reps, 97.5), linewidth=1, color='r', linestyle='--')
    plt.show()
    
    print(ttest_ind(df_not['b_score'], df_is['b_score'], equal_var=False))