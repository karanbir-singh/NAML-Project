import numpy as np

def min_max(data, data_min=None, data_max=None):
  """
    Applies the min-max scaling method for normalizing data.

    Parameters:
      data: ndarray
        dataset to scale with samples on rows
      data_min: ndarray, optional
        array containing the min value for each feature
      data_max: ndarray, optional
        array containing the max value for each feature

    Returns:
      data_normalized: ndarray
        original dataset but normalized
      data_min: ndarray
        array containing the min value for each feature
      data_max: ndarray
        array containing the max value for each feature
  """

  if data_min is None or data_max is None:
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

  data_normalized = (data - data_min[None,:]) / (data_max[None,:] - data_min[None,:])
  return data_normalized, data_min, data_max


def data_split(data_input, train_size=0.8, valid_size=0.0):
  """
    Splits dataset in train and test dataset

    Parameters:
    data_input: ndarray
      dataset to split with samples on rows
    train_size: float, optional
      percentage of data to put in the train dataset
    valid_size: float, optional
      percentage of data to put in the validation dataset

    Returns:
    x_train: ndarray
      train dataset with samples
    y_train: ndarray
      train dataset with labels
    x_valid: ndarray
      validation dataset with samples
    y_valid: ndarray
      validation dataset with labels
    x_test: ndarray
      test dataset with samples
    y_test: ndarray
      test dataset with labels
  """

  # Get a copy of the data
  data = data_input.copy()

  np.random.seed(0)  # For reproducibility
  np.random.shuffle(data)  # Shuffle along rows

  # Take the number of samples to put in train dataset
  num_train = int(data.shape[0] * train_size)
  num_valid = num_train + int(data.shape[0] * valid_size)

  # Train dataset
  x_train = data[:num_train, :-1]  # Don't take last column which represents the class
  y_train = data[:num_train, -1:]

  # Validation dataset
  x_valid = data[num_train:num_valid, :-1]
  y_valid = data[num_train:num_valid, -1:]

  # Test dataset
  x_test = data[num_valid:, :-1]
  y_test = data[num_valid:, -1:]

  return x_train, y_train, x_valid, y_valid, x_test, y_test


