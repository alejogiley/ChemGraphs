import pandas as pd

def save_history(history, output):
	"""Save training history"""
	try:
		dataset = pd.DataFrame.from_dict(history.history)
		dataset.to_csv(output, index=False)
	except OSError as err:
		raise OSError("Wrong file path {}, err {}".format(output, str(err)))


def save_model(model, output):
    """Save GCNN model"""
    try:
        model.save(output, overwrite=True, include_optimizer=True)
    except Exception as err:
        raise ValueError("Unexpected error: {}".format(str(err)))
      
  