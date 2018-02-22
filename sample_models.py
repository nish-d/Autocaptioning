from keras import backend as K
# from keras import optimizers
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Add)


#model_0
def simple_rnn_model(input_dim=13, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_1
def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # Batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_2
def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

# model_3
def deep_rnn_model(input_dim, units, recur_layers, neurons, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Recurrent layers, each with batch normalization
    previous_input = input_data
    simp_rnn = []
    bn_cnn = []
    
    for i in range(0, recur_layers):
        simp_rnn_active = GRU(units=neurons[i], dropout=0.35, return_sequences=True, implementation=2, name='rnn'+str(i))(previous_input)
        simp_rnn.append(simp_rnn_active)
        # Batch normalization
        bn_cnn_active = BatchNormalization(name="bn_conv_1d"+str(i))(simp_rnn[i])
        bn_cnn.append(bn_cnn_active)
        previous_input = bn_cnn[i]
        #units=units-40
   
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_cnn[recur_layers-1])
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_4
def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units,
        return_sequences=True, implementation=2, name='rnn'))(input_data)
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_5
def lstm_model(input_dim, units, output_dim=29):
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    lstm_rnn = Bidirectional(LSTM(units, return_sequences=True, implementation=2, name='rnn'))(input_data)
    # TODO: Add batch normalization 
    bn_rnn  = BatchNormalization()(lstm_rnn)
    # Time series:
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_6 - deep bidirectional rnn
def deep_bidirectional_rnn(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    previous_input = input_data
    bidir_rnn = []
    for i in range (0,recur_layers):
        bidir_rnn_active = Bidirectional(GRU(units, return_sequences=True, implementation=2, name="rnn"+str(i)))(previous_input)
        bidir_rnn.append(bidir_rnn_active)
        previous_input = bidir_rnn[i]
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn[recur_layers-1])
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_bidirectional_rnn(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    previous_input = input_data
    bidir_rnn = []
    for i in range (0,recur_layers):
        bidir_rnn_active = Bidirectional(GRU(units, return_sequences=True, implementation=2, name="rnn"+str(i)))(previous_input)
        bidir_rnn.append(bidir_rnn_active)
        previous_input = bidir_rnn[i]
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn[recur_layers-1])
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


# Final model
"""def final_model(input_dim, units, recur_layers, output_dim=29):
     Build a deep network for speech 
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    previous_input = input_data
    bidir_rnn = []
    for i in range (0,recur_layers):
        bidir_rnn_active = Bidirectional(GRU(units, return_sequences=True, implementation=2, name="rnn"+str(i)))(previous_input)
        bidir_rnn.append(bidir_rnn_active)
        previous_input = bidir_rnn[i]
        #units=units-40
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn[recur_layers-1])
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model"""
    
def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=2, 
    cell=GRU, activation='tanh'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    #print("shape", shape)
    print("input dim", input_dim)
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)


    if number_of_layers == 1:
        layer = cell(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = cell(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate)(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)
    

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

    
def experi_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, number_of_layers, 
    cell=GRU, activation='relu', output_dim=29, dropout_rate=0.4):
    
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    print(input_data.shape)
    """
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d',
                    dilation_rate=1)(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d_experi')(conv_1d)"""
    #print(bn_cnn.shape)
    recur_layers=number_of_layers
    #print("henlo frens")
    # TODO: Add bidirectional recurrent layer
    previous_input = input_data
    
    bidir_rnn = []
    bn_brnn = []
    
    for i in range (0,recur_layers):
        print(i)
        #bidir_rnn_active = Bidirectional(GRU(units, return_sequences=True, implementation=2, name="rnn"+str(i), dropout=dropout_rate))(previous_input)
        bidir_rnn_active = Bidirectional(GRU(units, return_sequences=True, implementation=2, name= "rnn"+str(i), dropout=dropout_rate, activation='relu'), merge_mode='concat')(previous_input)
        bidir_rnn.append(bidir_rnn_active)
        previous_input = bidir_rnn[i]
    #added=Add()([bidir_rnn[recur_layers-2],bidir_rnn[recur_layers-1]])
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn[recur_layers-1])
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def deepBidir_GRU_LSTM_model(input_dim, units, number_of_layers, 
    cell, dropout_rate, activation='relu', output_dim=29, ):
    
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    recur_layers=number_of_layers
    
    # TODO: Add bidirectional recurrent layer
    previous_input = input_data
    
    bidir_rnn = []
    bn_brnn = []
    
    for i in range (0,recur_layers):
        print(i)
        bidir_rnn_active = Bidirectional(cell(units, return_sequences=True, implementation=2, name= "rnn"+str(i), dropout=dropout_rate, activation='relu'), merge_mode='concat')(previous_input)
        bidir_rnn.append(bidir_rnn_active)
        previous_input = bidir_rnn[i]
    
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn[recur_layers-1])
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model