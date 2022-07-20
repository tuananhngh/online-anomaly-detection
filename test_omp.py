import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from onlinempmodel import OnlineMatrixProfile
from utils import init_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, help="path of the input data folder")
    parser.add_argument("-o", "--output_dir",type=str, help="path of the output directory.")
    parser.add_argument("-sdt", "--start_date",type=str, help="limit warmup date: [YYYY-MM-DD]")
    parser.add_argument("-m", "--window_size",type=int, help="window size for matrix profile")

    args = parser.parse_args()

    data_path = args.input_dir 
    file_to_save = args.output_dir
    limit_warmup = args.start_date
    m = args.window_size

    time_col, val_col = "timestamp", "value"
    full_data = pd.read_csv(data_path,header=0, parse_dates=[time_col])
    warmup_data, learning_data = init_data(full_data, limit_warmup, time_col)

    ts_warmup, ts_learning = warmup_data[val_col].values, learning_data[val_col].values
    time_warmup, time_learning = list(warmup_data[time_col].astype(str)), list(learning_data[time_col].astype(str))

    omp = OnlineMatrixProfile(ts_warmup, time_warmup, m=m)
    #-------Learning--------#
    for time,val in zip(time_learning, ts_learning):
        omp.learn_one(time,val,file_to_save)

