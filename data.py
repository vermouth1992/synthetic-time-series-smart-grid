"""
A set of utility functions to load and process data
"""
from __future__ import print_function

import datetime

import numpy as np

"""
Preprocessing functions
"""

start_date_str = '2013-01-01 00:00:00'
end_date_str = '2016-12-31 23:45:00'
special_date_str = '2016-02-29'
date_format = '%Y-%m-%d %H:%M:%S'
date_format_day = '%Y-%m-%d'
special_date = datetime.datetime.strptime(special_date_str, date_format_day)
start_date = datetime.datetime.strptime(start_date_str, date_format)
end_date = datetime.datetime.strptime(end_date_str, date_format)


def index_to_date(index):
    """ translate from index to date from 2013-01-01

    Args:
        index: the date from start-date (2013-01-01)

    Returns: string date from 2013-01-01
    """
    return (start_date + datetime.timedelta(index)).strftime(date_format_day)


def load_pecan_dataset(filepath='dataset/dataset.npz', include_special_day=True,
                       include_labels=False):
    """ Load the pecan dataset from previous saved numpy array

    Args:
        filepath: path to the file
        include_special_day: include 2/29. If not, the data is more easy to process
        include_labels: include month_label and weekday/weekend labels

    Returns:
        user_id_pv: user id with PV. Shape: (num_user_pv,), where num_user_pv=25
        data_matrix_pv: Shape: (num_user_pv, 1461, 96, 3), where 1461 is the number of days,
                        96 is number of sampling points within one day. The data is (use, gen, grid)
        user_id_no_pv: user id without pv. Shape: (num_user_without_pv,), where num_user_without_pv=6
        data_matrix_no_pv: Shape: (num_user_without_pv, 1461, 96, 3).
        month_label: numpy array of shape (1461,), the label are numbers from 0 - 11
        weekday/weekend: numpy array of shape (1461,), the label are numbers from 0 - 1

    """
    data = np.load(filepath)
    user_id_pv, data_matrix_pv = data['user_id_pv'], data['data_matrix_pv']
    user_id_no_pv, data_matrix_no_pv = data['user_id_no_pv'], data['data_matrix_no_pv']
    month_label = np.zeros(shape=(data_matrix_pv.shape[1]))
    day_label = np.zeros(shape=(data_matrix_pv.shape[1]))
    for i in range(data_matrix_pv.shape[1]):
        current_date = start_date + datetime.timedelta(i)
        month_label[i] = current_date.month - 1
        day_label[i] = current_date.weekday()

    if not include_special_day:
        special_day_index = (special_date - start_date).days
        print('Warning. Special day (2.29) at index {} is not included.'.format(special_day_index))
        data_matrix_pv = np.concatenate(
            (data_matrix_pv[:, :special_day_index, :, :], data_matrix_pv[:, special_day_index + 1:, :, :]), axis=1)
        data_matrix_no_pv = np.concatenate(
            (data_matrix_no_pv[:, :special_day_index, :, :], data_matrix_no_pv[:, special_day_index + 1:, :, :]),
            axis=1)
        month_label = np.concatenate((month_label[:special_day_index], month_label[special_day_index + 1:]))
        day_label = np.concatenate((day_label[:special_day_index], day_label[special_day_index + 1:]))
    if not include_labels:
        return user_id_pv, data_matrix_pv, user_id_no_pv, data_matrix_no_pv
    else:
        return user_id_pv, data_matrix_pv, user_id_no_pv, data_matrix_no_pv, month_label, day_label


def preprocess_pecan_dataset(user_id, threshold=(-10, 10)):
    """ Preprocess pecan dataset to have stationary patterns.

    The process is:
    1. Minus the mean.
    2. Divide the standard deviation
    3. Apply clip to reduce noise
    4. MinMaxScaler normalize to (0, 1)

    Args:
        user_id: user id
        threshold: (min, max)

    Returns: a tuple of (use, gen). Each with shape (1440, 96)
             a tuple of function (func_use, func_gen) to perform recover to original scale.

    """
    user_id_pv, data_matrix_pv, user_id_no_pv, data_matrix_no_pv, month_label, day_label = load_pecan_dataset(
        'dataset/dataset.npz', include_labels=True)
    low, high = threshold
    if user_id in user_id_pv:
        user_index = user_id_pv.tolist().index(user_id)
        # calculate mean
        daily_average_user_pv_usage = np.mean(data_matrix_pv[user_index, :, :, 0], axis=-1, keepdims=True)
        daily_average_user_pv_gen = np.mean(data_matrix_pv[user_index, :, :, 1], axis=-1, keepdims=True)
        # minus mean
        one_user_pv_usage_minus_average = data_matrix_pv[user_index, :, :, 0] - daily_average_user_pv_usage
        one_user_pv_gen_minus_average = data_matrix_pv[user_index, :, :, 1] - daily_average_user_pv_gen
        # calculate std
        daily_usage_std = np.std(one_user_pv_usage_minus_average, axis=-1, keepdims=True)
        daily_gen_std = np.std(one_user_pv_gen_minus_average, axis=-1, keepdims=True)
        # divide std
        one_user_pv_usage_minus_average_std = one_user_pv_usage_minus_average / daily_usage_std
        one_user_pv_gen_minus_average_std = np.divide(one_user_pv_gen_minus_average, daily_gen_std,
                                                      where=daily_gen_std != 0)
        # MinMaxScaler
        daily_usage_minimum = np.maximum(np.min(one_user_pv_usage_minus_average_std, axis=-1, keepdims=True), low)
        daily_usage_maximum = np.minimum(np.max(one_user_pv_usage_minus_average_std, axis=-1, keepdims=True), high)
        daily_gen_minimum = np.min(one_user_pv_gen_minus_average_std, axis=-1, keepdims=True)
        daily_gen_maximum = np.max(one_user_pv_gen_minus_average_std, axis=-1, keepdims=True)
        one_user_pv_usage_minus_average_normalize = (one_user_pv_usage_minus_average_std - daily_usage_minimum) \
                                                    / (daily_usage_maximum - daily_usage_minimum)
        one_user_pv_usage_minus_average_normalize = np.clip(one_user_pv_usage_minus_average_normalize, 0, 1)
        difference = daily_gen_maximum - daily_gen_minimum
        one_user_pv_gen_minus_average_normalize = np.divide(one_user_pv_gen_minus_average_std - daily_gen_minimum,
                                                            difference, where=difference != 0)

        def usage_recover(usage):
            """ usage is shape (1440, 96) """
            return (usage * (daily_usage_maximum - daily_usage_minimum) + daily_usage_minimum) * daily_usage_std + \
                   daily_average_user_pv_usage

        def gen_recover(gen):
            """ gen is shape (1440, 96) """
            return (gen * (daily_gen_maximum - daily_gen_minimum) + daily_gen_minimum) * daily_gen_std + \
                   daily_average_user_pv_gen

        return (one_user_pv_usage_minus_average_normalize, one_user_pv_gen_minus_average_normalize), \
               (usage_recover, gen_recover), (month_label, day_label)

    elif user_id in user_id_no_pv:
        user_index = user_id_pv.tolist().index(user_id)
        # calculate mean
        daily_average_user_usage = np.mean(data_matrix_no_pv[user_index, :, :, 0], axis=-1, keepdims=True)
        # minus mean
        one_user_usage_minus_average = data_matrix_no_pv[user_index, :, :, 0] - daily_average_user_usage
        # calculate std
        daily_usage_std = np.std(one_user_usage_minus_average, axis=-1, keepdims=True)
        # divide std
        one_user_usage_minus_average_std = one_user_usage_minus_average / daily_usage_std
        # MinMaxScaler
        daily_usage_minimum = np.maximum(np.min(one_user_usage_minus_average_std, axis=-1, keepdims=True), low)
        daily_usage_maximum = np.minimum(np.max(one_user_usage_minus_average_std, axis=-1, keepdims=True), high)
        one_user_usage_minus_average_normalize = (one_user_usage_minus_average_std - daily_usage_minimum) \
                                                 / (daily_usage_maximum - daily_usage_minimum)
        one_user_usage_minus_average_normalize = np.clip(one_user_usage_minus_average_normalize, 0, 1)

        def usage_recover(usage):
            """ usage is shape (1440, 96) """
            return (usage * (daily_usage_maximum - daily_usage_minimum) + daily_usage_minimum) * daily_usage_std + \
                   daily_average_user_usage

        gen_recover = lambda x: x

        return (one_user_usage_minus_average_normalize, data_matrix_no_pv[user_index, :, :, 1]), \
               (usage_recover, gen_recover), (month_label, day_label)

    else:
        raise ValueError('Invalid user id')
