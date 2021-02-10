import tarfile
import tempfile
import pandas as pd


def save_history(history, output):
    """Save training history"""
    try:
        dataset = pd.DataFrame.from_dict(history.history)
        dataset.to_csv(output, index=False)

    except IOError as err:
        raise IOError("Wrong file path {}, err {}".format(output, str(err)))


def save_model(model, output):
    """Save GCNN model"""
    try:
        with tempfile.NamedTemporaryFile() as temp, tarfile.open(output, "w:gz") as tar:
            model.save(temp.name, overwrite=True, include_optimizer=True)
            tar.add(temp.name, arcname="gcnn.h5")

    except IOError as err:
        raise IOError("Unable to save model, err {}".format(str(err)))
