import numpy as np
from river import anomaly
from river import compose
from river import preprocessing
from river import metrics 
from river import optim
from river import stats
import json


class OnlineOneClassSVM:

    def __init__(self, nu=0.1, lr=0.5, q=0.98, window_roll=12, protect_detector=True):
        """
        Initialization online OneClass SVM.
        Args:
            nu (float):
            lr (float):
            q (float):
            protect_detector (bool) :  Default to True
        """
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            anomaly.QuantileFilter(anomaly.OneClassSVM(nu=nu, optimizer=optim.SGD(lr)),
                                   q=q, protect_anomaly_detector=protect_detector))
        #self.anom_value = []
        #self.anom_time = []
        self._roll = stats.RollingMean(window_roll)
        self.anom = {"timestamp":[], "value":[]}
        #self.anomaly_filepath = anomaly_filepath
        
    def _rolling_avg(self, x):
        return self._roll.update(x).get()

    def learn_one(self, time, value, file_to_write, warmup=False):
        """
        Args:
            time: datetime object
            value: value
        Returns:
            None
        """
        feature = {"value": self._rolling_avg(value)}
        score = self.model.score_one(feature)
        is_anomaly = self.model["QuantileFilter"].classify(score)
        self.model = self.model.learn_one(feature)
        if is_anomaly:
            if warmup:
                pass
            else:
                print(f"Found anomaly at {time}, value {value}")
                #d_anom = {"timestamp":time, "value":value}
                self.anom["timestamp"].append(time)
                self.anom["value"].append(value)
                with open(file_to_write,'a') as f:
                    f.writelines(f"{time},{value}\n")
                #   json.dump(self.anom, f, indent=4)


