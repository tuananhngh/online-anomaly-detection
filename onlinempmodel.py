import stumpy
import numpy as np
from river import stats

class OnlineMatrixProfile:

    def __init__(self, ts, timestamps, m, egress=False, normalize=False, p_norm=2.):
        """
        Initialization for the online matrix profile computation.
        Args:
            ts (array-like): initial time-series for warming up the matrix profile
            timestamps (array-like): timestamp corresponding to the initial time-series 
            m (int): window length 
            egress (bool, optional): Remove the oldest data from matrix profile. Defaults to False.
            normalize (bool, optional): Z-normalization. Defaults to False.
            p_norm (_type_, optional): Can be 1-norm or 2-norm. Defaults to 2..
        """
        self.m = m
        self.times = timestamps
        self.values = ts
        self.p_max = -1
        self.profile_discord = []
        self.time_discords = []
        self.idx_discords = []
        
        self._rolling_avg = stats.RollingMean(window_size=12)

        self.model = stumpy.stumpi(
            self.values, self.m, egress=egress, normalize=normalize, p=p_norm)
        self.len_ts = len(self.values)

    def nfill_mp(self, profile):
        """
        Fill the matrix profile missing values with 0.
        The value will be added at the end of the mp.
        Args:
            profile (array-like): matrix profile

        Returns:
            array-like: matrix profile with the same length as the original series
        """
        n_fill = self.m-1
        filler = np.empty(n_fill)
        filler.fill(0)
        fill_mp = np.concatenate((profile.astype(float), filler))
        return fill_mp
    
    def _rolling_avg(self, x):
        return self._roll.update(x).get()

    def learn_one(self, time, value, file_to_save):
        """
        Update matrix profile with one data point.
        Args:
            time (timestamp): time of the data point
            value (float): value of the data point
        """
        self.model.update(value)
        self.times.append(time)
        self.len_ts += 1
        left_profile = self.model.left_P_.copy()
        left_profile[left_profile == np.inf] = -1
        k_largest = np.argsort(left_profile)[-1]
        if left_profile[k_largest] > self.p_max:
            print(f"-----Problem when adding {time}, {value:.2f}-----")
            self.p_max = left_profile[k_largest]
            print(f"Found anomaly at {self.times[k_largest]}")
            tmp_discords = k_largest 
            with open(file_to_save, 'a') as f:
                f.writelines(f"{self.times[k_largest]},{left_profile[k_largest]}\n")
            self.idx_discords.append(tmp_discords)
            self.time_discords.append(self.times[k_largest])
            self.profile_discord.append(left_profile[k_largest])
