import os
import joblib

import pandas as pd

from typing import Dict
from tensorflow.keras.models import save_model


def save_dataset(dataset, data_path: str, file_name: str):
    """Save Graph dataset"""
    path = os.path.join(data_path, file_name + ".gz")
    try:
        joblib.dump(dataset, path, compress=("gzip", 6), protocol=5)
    except Exception as err:
        raise IOError("Unable to save dataset: {}".format(str(err)))


def save_history(history, out_path: str):
    """Save training history"""
    try:
        dataset = pd.DataFrame.from_dict(history.history)
        dataset.to_csv(out_path, index=False)
    except OSError as err:
        raise OSError("Unable to save training history: {}".format(str(err)))


def save_gcnn(model, out_path: str):
    """Save GCNN model"""
    try:
        save_model(model,
                   out_path,
                   overwrite=True,
                   include_optimizer=True,
                   save_traces=True)
    except Exception as err:
        raise IOError("Unable to save model: {}".format(str(err)))
    

def save_metrics(metrics: Dict, out_path: str):
    """Save model performance"""
    tmp = "MAE {:2f}\nMSE {:2f}\nPearson {:2f}\n%%lefts {:2f}\n%%right {:2f}"
    try:
        with open(out_path, 'w+') as file:
            file.write(tmp.format(*metrics.values()))
    except Exception as err:
        raise IOError("Unable to save metrics: {}".format(str(err)))
