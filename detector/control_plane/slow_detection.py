import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Rbeast as rb
from statsmodels.tsa import stattools
from scipy.signal import find_peaks
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf


def remove_outliers(iter_durations, iter_start):
    iter_durations = pd.DataFrame({
        "data": iter_durations,
        "iter_start": iter_start
    })
    # Calculate the outlier boundary using z_score
    # Zscore = (data_point - mean) / std. deviation
    threshold_z = 3.0
    z = np.abs(stats.zscore(iter_durations['data']))
    outlier_indices = np.where(z > threshold_z)[0]
    iter_durations.drop(outlier_indices, inplace=True)
    return iter_durations['data'].to_numpy(), iter_durations['iter_start'].to_numpy()


def find_performance_drop_naive(call_id, call_time, period, start):
    # By dt, not work...
    df = pd.DataFrame({"call_id": call_id, "call_time": call_time})
    for i, d in df.groupby("call_id"):
        ts = d['call_time'].to_numpy()
        dt = ts[1:] - ts[:-1]
        plt.scatter(ts[:-1], dt, label=str(i))
    plt.tight_layout()
    plt.show()


def validate_performance_drop(
        iter_durations, iter_start, change_point_ids, change_points,
        length_thresh=10, degradation_thresh=0.1):
    # @param length_thresh: we consider this performance change point is
    # valid only if the interval between two change points >= length_thresh
    # @param degradation_thresh: we report this event as a performance change
    # only if the performace varaition >= degradation_thresh
    assert len(iter_durations) == len(iter_start)
    change_point_df = pd.DataFrame({
        "ids": change_point_ids,
        "values": change_points
    })
    if len(change_point_df) == 0:
        # No change points
        return change_point_df

    # Sort it by occurance order
    change_point_df.sort_values(by='ids', inplace=True, ignore_index=True)
    # Calculate the difference between each number and its predecessor
    change_point_df['diff'] = change_point_df['ids'].diff()
    if change_point_df['ids'][0] < length_thresh:
        change_point_df.loc[0, 'diff'] = 0
    # Filter rows: keep the first row and any row where the difference is greater than 1
    change_point_df = change_point_df[(change_point_df['diff'] > length_thresh) | change_point_df['diff'].isnull()]
    change_point_df.drop(columns=['diff'], inplace=True)
    change_point_df.reset_index(drop=True, inplace=True)
    left_cp = lambda idx: change_point_df['ids'][idx - 1]\
        if idx > 0\
        else 0
    right_cp = lambda idx: change_point_df['ids'][idx + 1]\
        if idx < len(change_point_df) - 1\
        else len(iter_durations) - 1
    curr_cp = lambda idx: change_point_df['ids'][idx]

    rows_to_remove = []
    for i, row in change_point_df.iterrows():
        left_values = iter_durations[left_cp(i): curr_cp(i)]
        right_values = iter_durations[curr_cp(i): right_cp(i)]
        mean_l, mean_r = np.mean(left_values), np.mean(right_values)
        degradation = np.abs(mean_l - mean_r) / np.max((mean_l, mean_r))
        if degradation <= degradation_thresh:
            logging.info(
                f"Change point {curr_cp(i)} is filtered out due to less degradation: {degradation} < {degradation_thresh}")
            rows_to_remove.append(i)
        else:
            logging.info(
                f"Find valid change point {curr_cp(i)}, at time: {iter_start[curr_cp(i)]}, degradation={degradation}")
    change_point_df.drop(rows_to_remove, inplace=True)
    return change_point_df


def find_performance_drop(call_id, call_time, period, start, thresh_prob=0.8, plot=False, plot_args=None):
    # Find performance gap by repeat patterns
    ts, iter_start = [], []
    for i in range(start, len(call_id), period):
        if i + period >= len(call_time):
            continue
        ts.append(call_time[i + period] - call_time[i])
        iter_start.append(call_time[i])
    ts, iter_start = remove_outliers(ts, iter_start)
    last_10_avg_ts = np.mean(ts[-2:])
    try:
        while True:
            result = rb.beast(ts, season='none', print_options=False, print_progress=False, quiet=True, hasOutlier=True)
            if hasattr(result, 'trend'):
                break
    except:
        pass
    # rb.print(result)
    num_change_points = int(result.trend.ncp_mode[0])
    change_point_pos = np.array(result.trend.cp[:num_change_points], dtype=np.int32)
    change_point_prob = list(result.trend.cpPr)
    ymax = max(ts)
    real_change_points = []
    real_change_point_ids = []
    for i in range(num_change_points):
        # Breaks on invalid values (NaNs)
        if change_point_pos[i] == np.iinfo(np.int32).max or change_point_pos[i] == np.iinfo(np.int32).min:
            break
        real_pos = iter_start[change_point_pos[i]]
        if change_point_prob[i] <= thresh_prob:
            continue
        logging.info(f"Find suspective change point at t={real_pos}, prob={change_point_prob[i]}")
        real_change_points.append(real_pos)
        real_change_point_ids.append(change_point_pos[i])
        if plot:
            ax = plot_args['ax']
            ax.plot([real_pos, real_pos], [0, ymax], '-.', c='black', label='Suspective Change Point')
    real_change_point_df = validate_performance_drop(
        ts, iter_start, real_change_point_ids, real_change_points
    )
    if plot:
        ax = plot_args['ax']
        label = plot_args.get('label', 'Record')
        color = plot_args.get('color', 'blue')
        ax.scatter(iter_start, ts, label=label, color=color)
        ax.set_xlabel(plot_args.get('xlabel', 'X'))
        ax.set_ylabel(plot_args.get('ylabel', 'Y'))
        for cpt in real_change_point_df['values']:
            ax.plot([cpt, cpt], [0, ymax], c='red', label='Validated Change Point')
    real_change_point_df.reset_index(drop=True, inplace=True)
    return real_change_point_df, last_10_avg_ts


def find_period(seq, nlags=200, significance_level=0.7):
    def dist(seq1, seq2):
        # Count the number of different elements in two sequence
        # We assume the NCCL call pattern is *exactly the same*
        # within each training iteration
        assert len(seq1) == len(seq2)
        return len(seq1) - np.sum(seq1 == seq2)
    acf_values = stattools.acf(seq, nlags=nlags)
    # Find peaks in the ACF that are above the significance level
    peaks, _ = find_peaks(acf_values, height=significance_level)
    if len(peaks) >= 1:
        # Estimate period as the lag of the first peak
        estimated_period = peaks[0]
        for i in range(len(seq)):
            if dist(seq[i: i + estimated_period],
                    seq[i + estimated_period: i + 2 * estimated_period]) == 0:
                break
        logging.info(f"Repeat pattern starts from {i}, period = {estimated_period}, pattern = {seq[i: i + estimated_period]}")
        return i, estimated_period
    else:
        warnings.warn("No peaks found in ACF, no patterns are found in NCCL logs")
        return -1, None 
