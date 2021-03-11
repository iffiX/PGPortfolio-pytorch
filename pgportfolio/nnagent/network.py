import numpy as np
import torch as t
import torch.nn as nn


def get_conv_output_size(shape, kernel_size, stride, padding):
    # shape is [N, C, H, W] format.
    # See https://stackoverflow.com/a/37674568
    if padding == "same":
        out_height = int(np.ceil(float(shape[2]) / float(stride[0])))
        out_width = int(np.ceil(float(shape[3]) / float(stride[1])))
    elif padding == "valid":
        out_height = int(np.ceil(
            float(shape[2] - kernel_size[0] + 1) / float(stride[0])
        ))
        out_width = int(np.ceil(
            float(shape[3] - kernel_size[1] + 1) / float(stride[1])
        ))
    else:
        raise ValueError("Unknown padding: %s" % padding)
    return [shape[0], shape[1], out_height, out_width]


def get_conv_same_pad_size(kernel_size, stride):
    # see 
    # https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/4
    # The total padding applied along the height and width is computed as:
    if kernel_size[0] % stride[1] == 0:
        pad_along_height = max(kernel_size[0] - stride[0], 0)
    else:
        pad_along_height = max(kernel_size[0] - (kernel_size[0] % stride[0]), 0)
    if kernel_size[1] % stride[2] == 0:
        pad_along_width = max(kernel_size[1] - stride[1], 0)
    else:
        pad_along_width = max(kernel_size[1] - (kernel_size[1] % stride[1]), 0)

    print(pad_along_height, pad_along_width)

    # Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def dense(shape, layer_conf):
    assert len(shape) == 2, \
        "Dense layer input shape incorrect: {}".format(shape)
    layer = nn.Sequential(
        nn.Linear(in_features=shape[1],
                  out_features=layer_conf["neuron_number"]),
        getattr(nn.modules.activation,
                layer_conf["activation_function"])()
    )
    shape = [shape[0], layer_conf["neuron_number"]]
    return shape, layer


def dropout(shape, layer_conf):
    return shape, nn.Dropout(p=layer_conf["keep_probability"])


def EIIE_dense(shape, layer_conf):
    # shape: [batch, feature, coin, window]
    assert len(shape) == 4, \
        "EIIE dense layer input shape incorrect: {}".format(shape)
    layer = nn.Sequential(
        nn.Conv2d(
            in_channels=shape[1],
            out_channels=layer_conf["filter_number"],
            kernel_size=(1, shape[3]),
            stride=(1, 1)
        ),
        getattr(nn.modules.activation,
                layer_conf["activation_function"])()
    )
    shape = [shape[0], layer_conf["filter_number"], shape[2], 1]
    return shape, layer


def conv2d(shape, layer_conf):
    # shape: [batch, feature, coin, window]
    assert len(shape) == 4, \
        "Conv2d layer input shape incorrect: {}".format(shape)
    if layer_conf["padding"] == "same":
        layer = nn.Sequential(
            nn.ZeroPad2d(get_conv_same_pad_size(layer_conf["filter_shape"],
                                                layer_conf["strides"])),
            nn.Conv2d(in_channels=shape[1],
                      out_channels=layer_conf["filter_number"],
                      kernel_size=tuple(layer_conf["filter_shape"]),
                      stride=tuple(layer_conf["strides"])),
            getattr(nn.modules.activation,
                    layer_conf["activation_function"])()
        )
    else:
        layer = nn.Sequential(
            nn.Conv2d(in_channels=shape[1],
                      out_channels=layer_conf["filter_number"],
                      kernel_size=tuple(layer_conf["filter_shape"]),
                      stride=tuple(layer_conf["strides"])),
            getattr(nn.modules.activation,
                    layer_conf["activation_function"])()
        )
    shape = get_conv_output_size(shape,
                                 layer_conf["filter_shape"],
                                 layer_conf["strides"],
                                 layer_conf["padding"])
    return shape, layer


def max_pool_2d(shape, layer_conf):
    # shape: [batch, feature, coin, window]
    assert len(shape) == 4, \
        "MaxPool2d layer input shape incorrect: {}".format(shape)
    if layer_conf["padding"] == "same":
        layer = nn.Sequential(
            nn.ZeroPad2d(get_conv_same_pad_size(layer_conf["filter_shape"],
                                                layer_conf["strides"])),
            nn.MaxPool2d(kernel_size=tuple(layer_conf["filter_shape"]),
                         stride=tuple(layer_conf["strides"])),
        )
    else:
        layer = nn.MaxPool2d(kernel_size=tuple(layer_conf["filter_shape"]),
                             stride=tuple(layer_conf["strides"]))
    shape = get_conv_output_size(shape,
                                 layer_conf["filter_shape"],
                                 layer_conf["strides"],
                                 layer_conf["padding"])
    return shape, layer


def avg_pool_2d(shape, layer_conf):
    # shape: [batch, feature, coin, window]
    assert len(shape) == 4, \
        "AvgPool2d layer input shape incorrect: {}".format(shape)
    if layer_conf["padding"] == "same":
        layer = nn.Sequential(
            nn.ZeroPad2d(get_conv_same_pad_size(layer_conf["filter_shape"],
                                                layer_conf["strides"])),
            nn.AvgPool2d(kernel_size=tuple(layer_conf["filter_shape"]),
                         stride=tuple(layer_conf["strides"])),
        )
    else:
        layer = nn.AvgPool2d(kernel_size=tuple(layer_conf["filter_shape"]),
                             stride=tuple(layer_conf["strides"]))
    shape = get_conv_output_size(shape,
                                 layer_conf["filter_shape"],
                                 layer_conf["strides"],
                                 layer_conf["padding"])
    return shape, layer


def local_response_normalization(shape):
    layer = nn.LocalResponseNorm(5)
    return shape, layer


def EIIE_recurrent(shape, layer_conf):
    # shape: [batch, feature, coin, window]
    assert len(shape) == 4, \
        "EIIE-recurrent layer input shape incorrect: {}".format(shape)
    layer = EIIE_RecurrentModule(shape, layer_conf)
    shape = [shape[0], layer_conf["neuron_number"], shape[2], shape[3]]
    return shape, layer


def output_with_w(shape):
    # shape: [batch, feature, coin, window]
    assert len(shape) == 4, \
        "Output-with-w layer input shape incorrect: {}".format(shape)
    layer = OutputWithWModule(shape)
    shape = [shape[0], shape[2] + 1]
    return shape, layer


def EIIE_output(shape):
    # shape: [batch, feature, coin, window]
    assert len(shape) == 4, \
        "EIIE-output layer input shape incorrect: {}".format(shape)
    layer = EIIE_OutputModule(shape)
    shape = [shape[0], shape[2] + 1]
    return shape, layer


def EIIE_output_with_w(shape):
    # shape: [batch, feature, coin, window]
    assert len(shape) == 4, \
        "EIIE-output-with-w layer input shape incorrect: {}".format(shape)
    layer = EIIE_OutputWithWModule(shape)
    shape = [shape[0], shape[2] + 1]
    return shape, layer


class OutputWithWModule(nn.Module):
    def __init__(self, shape):
        # shape: [batch, feature, coin, window]
        super(OutputWithWModule, self).__init__()
        self.fc = nn.Linear(in_features=np.product(shape[1:]) + shape[2] + 1,
                            out_features=shape[2] + 1)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, last_w):
        return self.sm(self.fc(
            t.cat([input.flatten(start_dim=1), last_w], dim=1)
        ))


class EIIE_OutputModule(nn.Module):
    def __init__(self, shape):
        # shape: [batch, feature, coin, window]
        super(EIIE_OutputModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=shape[1],
                              out_channels=1,
                              kernel_size=(1, shape[3]))
        self.sm = nn.Softmax(dim=1)

    def forward(self, input):
        btc_bias = t.ones((input.shape[0], 1), device=input.device)
        return self.sm(
            t.cat([btc_bias, self.conv(input)[:, :, 0, 0]], dim=1)
        )


class EIIE_OutputWithWModule(nn.Module):
    def __init__(self, shape):
        # shape: [batch, feature, coin, window]
        super(EIIE_OutputWithWModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=shape[1] * shape[3] + 1,
                              out_channels=1,
                              kernel_size=(1, 1))
        self.sm = nn.Softmax(dim=1)
        self.shape = shape

    def forward(self, input, last_w):
        # first permute input to [batch, feature, window, coin]
        # then flatten new dim 1, 2: (feature, window)
        input = t.cat([input.permute(0, 1, 3, 2)
                       .flatten(start_dim=1, end_dim=2).unsqueeze(-1),
                       last_w.to(input.device)
                      .view([-1, 1, self.shape[2], 1])], dim=1)
        input = self.conv(input)
        btc_bias = t.zeros((input.shape[0], 1), device=input.device)
        return self.sm(
            t.cat([btc_bias, input[:, 0, :, 0]], dim=1)
        )


class EIIE_RecurrentModule(nn.Module):
    def __init__(self, shape, layer_conf):
        # shape: [batch, feature, coin, window]
        super(EIIE_RecurrentModule, self).__init__()
        self.shape = shape
        self.hidden_size = layer_conf["neuron_number"]
        if layer_conf["type"] == "EIIE_LSTM":
            self.rec_mods = [
                nn.LSTM(input_size=shape[1],
                        hidden_size=layer_conf["neuron_number"],
                        dropout=layer_conf["dropouts"])
            ]
        else:
            self.rec_mods = [
                nn.RNN(input_size=shape[1],
                       hidden_size=layer_conf["neuron_number"],
                       dropout=layer_conf["dropouts"])
            ]

    def forward(self, input: t.Tensor):
        # now input becomes [coin, window, batch, feature]
        input = input.permute(2, 3, 0, 1)
        results = [self.rec_mods[i](input[i]) for i in range(self.shape[2])]
        input = t.stack(results)
        # turns back to [batch, hidden, coin, window]
        input = input.permute(2, 3, 0, 1)
        return input.view([-1, self.hidden_size, self.shape[2], self.shape[3]])


class CNN(nn.Module):
    def __init__(self, feature_number, coin_number, window_size, layers):
        # input_shape (features, rows, columns)
        # or (features, coin_number, window_size)
        super(CNN, self).__init__()
        self.feature_number = feature_number
        self.coin_number = coin_number
        self.window_size = window_size
        self.layers = []
        self.layer_types = []
        self._build_network(layers)

    def forward(self, x, last_w=None):
        # normalize all features in all periods
        # with "closing price" (dim 0) feature from the latest period
        # shape: [batch, feature, coin, time]
        x = x / x[:, 0, None, :, -1, None]
        for layer in self.layers[:-1]:
            x = layer(x)
        if "WithW" in self.layer_types[-1]:
            return self.layers[-1](x, last_w)
        else:
            return self.layers[-1](x)

    def _build_network(self, layers):
        # corresponds to [N, C, H, W]
        shape = [None, self.feature_number, self.coin_number, self.window_size]

        for layer in layers:
            self.layer_types.append(layer["type"])
            if layer["type"] == "DenseLayer":
                shape, l = dense(shape, layer)
                self.layers.append(l)
            elif layer["type"] == "DropOut":
                shape, l = dropout(shape, layer)
                self.layers.append(l)
            elif layer["type"] == "EIIE_Dense":
                shape, l = EIIE_dense(shape, layer)
                self.layers.append(l)
            elif layer["type"] == "ConvLayer":
                shape, l = conv2d(shape, layer)
                self.layers.append(l)
            elif layer["type"] == "MaxPooling":
                shape, l = max_pool_2d(shape, layer)
                self.layers.append(l)
            elif layer["type"] == "AveragePooling":
                shape, l = avg_pool_2d(shape, layer)
                self.layers.append(l)
            elif layer["type"] == "LocalResponseNormalization":
                shape, l = local_response_normalization(shape)
                self.layers.append(l)
            elif layer["type"] in ("EIIE_LSTM", "EIIE_RNN"):
                shape, l = EIIE_recurrent(shape, layer)
                self.layers.append(l)
            elif layer["type"] == "EIIE_Output":
                shape, l = EIIE_output(shape)
                self.layers.append(l)
            elif layer["type"] == "Output_WithW":
                shape, l = output_with_w(shape)
                self.layers.append(l)
            elif layer["type"] == "EIIE_Output_WithW":
                shape, l = EIIE_output_with_w(shape)
                self.layers.append(l)
            else:
                raise ValueError(
                    "Layer {} is not supported.".format(layer["type"]))

        for i, (layer, layer_t) in \
            enumerate(zip(self.layers, self.layer_types)):
            self.add_module("layer_{}_{}".format(i, layer_t), layer)
