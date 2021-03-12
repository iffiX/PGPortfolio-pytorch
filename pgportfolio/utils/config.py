import json


def fill_default(config):
    set_missing(config, "random_seed", 0)
    set_missing(config, "agent_type", "NNAgent")
    fill_layers_default(config["layers"])
    fill_input_default(config["input"])
    fill_train_config(config["training"])


def fill_train_config(train_config):
    set_missing(train_config, "fast_train", True)
    set_missing(train_config, "decay_rate", 1.0)


def fill_input_default(input_config):
    set_missing(input_config, "save_memory_mode", False)
    set_missing(input_config, "portion_reversed", False)
    set_missing(input_config, "market", "poloniex")
    set_missing(input_config, "norm_method", "absolute")
    set_missing(input_config, "fake_ratio", 1)


def fill_layers_default(layers):
    for layer in layers:
        if layer["type"] == "ConvLayer":
            set_missing(layer, "padding", "valid")
            set_missing(layer, "strides", [1, 1])
            set_missing(layer, "activation_function", "ReLU")
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "EIIE_Dense":
            set_missing(layer, "activation_function", "ReLU")
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "DenseLayer":
            set_missing(layer, "activation_function", "ReLU")
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "EIIE_LSTM" or layer["type"] == "EIIE_RNN":
            set_missing(layer, "dropouts", None)
        elif layer["type"] == "EIIE_Output" or\
                layer["type"] == "Output_WithW" or\
                layer["type"] == "EIIE_Output_WithW":
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "DropOut":
            pass
        else:
            raise ValueError("layer name {} not supported"
                             .format(layer["type"]))


def set_missing(config, name, value):
    if name not in config:
        config[name] = value


def load_config(path):
    with open(path) as file:
        config = json.load(file)
    fill_default(config)
    return config


def save_config(config, path):
    with open(path, "w") as file:
        json.dump(config, file, sort_keys=True, indent=4)


# def check_config_backtest_same(config1, config2):
#     input1 = config1["input"]
#     input2 = config2["input"]
#     if input1["start_date"] != input2["start_date"]:
#         return False
#     elif input1["end_date"] != input2["end_date"]:
#         return False
#     elif input1["test_portion"] != input2["test_portion"]:
#         return False
#     else:
#         return True
