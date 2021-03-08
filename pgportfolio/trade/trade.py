from __future__ import division,absolute_import,print_function
import numpy as np
from pgportfolio.nnagent.datamatrices import DataMatrices


def get_test_data(config):
    """
    :return : a 2d numpy array with shape(coin_number, periods),
     each element the relative price
    """
    config["input"]["feature_number"] = 1
    config["input"]["global_period"] = config["input"]["global_period"]
    price_matrix = DataMatrices.create_from_config(config)
    test_set = price_matrix.get_test_set()["y"][:, 0, :].T
    test_set = np.concatenate((np.ones((1, test_set.shape[1])), test_set), axis=0)
    return test_set


def asset_vector_to_dict(coin_list, vector, with_BTC=True):
    vector = np.squeeze(vector)
    dict_coin = {}
    if with_BTC:
        dict_coin['BTC'] = vector[0]
    for i, name in enumerate(coin_list):
        if vector[i+1] > 0:
            dict_coin[name] = vector[i + 1]
    return dict_coin


def save_test_data(config, file_name="test_data", output_format="csv"):
    if output_format == "csv":
        matrix = get_test_data(config)
        with open(file_name+"."+output_format, 'wb') as f:
            np.savetxt(f, matrix.T, delimiter=",")

