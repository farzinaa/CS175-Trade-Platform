from trade_platform.src.agent.agent_thread import agent_thread
from ...src.util.util import *
import numpy as np
from ...src.util.mrkt_data import mrkt_data
from tensorflow.keras.layers import Dense, add, Lambda, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, LSTM
from tensorflow.keras import Input, Model, backend
from tensorflow.keras.losses import binary_crossentropy
from sklearn import preprocessing
from ...src.util.Data_parsing.data_parsing import parse
import tensorflow as tf

#From https://github.com/philipperemy/keras-tcn
from tcn import TCN, tcn_full_summary

def custom_loss(y, y_hat):
    return binary_crossentropy(y, y_hat)
class tcn_agent(agent_thread):

    '''
    TCN network takes in the number of moments to look at
    '''
    def __init__(self, moments = 600, batch_size = None, input_dim = 4 ):
        agent_thread.__init__(self)
        self.moments = moments #Number of moments looked back to decide next value
        self.holding_time = 0
        self.batch_size = batch_size
        self.input_dim = input_dim #Dimensions of input default to 1
        self.built = False
        self.model() #Compile model
        self.training = True
        self.networth = 0
        self.amount = 0
        self.relevant_past = list()

    def _find_decision(self):
        if len(self.market_history) > self.moments:
            predicted_value = self.run_model()
            #print("Open is : ", self.market_history[self.time_counter - 1].open)
            #print("Close is : ", self.market_history[self.time_counter - 1].close)
            #print("Low is : ", self.market_history[self.time_counter - 1].low)
            #print("High is : ", self.market_history[self.time_counter - 1].high)
            if not self.holding and (predicted_value[0] > 0 and predicted_value[1] > predicted_value[0]):
                self.amount = 100#int(np.abs(1000*(predicted_value[1]-predicted_value[0])/(predicted_value[3]-predicted_value[2])))
                if self.amount < 0:
                    self.amount = 1
                self.act = action.BUY
                self.holding = True
                self.holding_time = self.time_counter
                self.buy_in_price = self.market_history[self.time_counter - 1].price
                self.networth -= self.market_history[self.time_counter - 1].price * self.amount
                #print("buy  at time " + str(self.time_counter) + "\t price : " + str(
                    #self.market_history[self.time_counter - 1].price))
            elif self.holding and (predicted_value[0] < 0 or predicted_value[1] < predicted_value[0]):
                self.act = action.SELL
                self.networth += self.market_history[self.time_counter - 1].price * self.amount
                self.amount = 0
                #print("sell at time " + str(self.time_counter) + "\t price : " + str(
                    #self.market_history[self.time_counter - 1].price))
                print("Current networth is: ", self.networth)
                self.holding = False
            elif self.holding_time - self.time_counter > 20:
                self.act = action.SELL
                print("hold 2 long  " + str(self.time_counter) + "\t price : " + str(
                    self.market_history[self.time_counter - 1].price))
                self.holding = False
            else:
                self.act = action.HOLD
            return self.act
        else:
            pass

    def model(self):
        #build model around TCN
        #As a general rule (keeping kernel_size fixed at 2) and dilations increasing with a factor of 2
        #The equation to find the ideal sizes for the last layer would be previous_layer + 2*(kernel_size-1)*(2^n-1)
        #For the nth residual block
        self.built = True
        i = Input(batch_shape=(self.batch_size, self.moments-1, self.input_dim))
        #Model 1: 5*10^-6 error,
        #########################################################################
        #x1 = TCN(return_sequences=True, nb_filters=32, nb_stacks = 1, dropout_rate=.0, kernel_size=2)(i)
        #x1 = Dense(4, activation='linear')(x1)
        #o = LSTM(4, dropout=.3)(x1)
        #########################################################################
        # Model 2: 1*10^-6 error, average networth change per tick = 23/(774-60) = shit
        #########################################################################
        #o = LSTM(4, dropout=.3)(i)
        #########################################################################
        #Model 3: 5.6779e-06 error, average networth change per tick = 863+503+628.5/(774-60)*3 = .9311, 50/(774-600) = .2874
        #########################################################################
        x1 = TCN(return_sequences=True, nb_filters=64, dilations = [1, 2, 4, 8, 16, 32], nb_stacks=1, dropout_rate=.1, kernel_size=3)(i)
        x2 = Lambda(lambda z: backend.reverse(z, axes=0))(i)
        x2 = TCN(return_sequences=True, nb_filters=64, dilations = [1, 2, 4, 8, 16, 32], nb_stacks=1, dropout_rate=.1, kernel_size=3)(x2)
        x = add([x1, x2])
        o = LSTM(4, dropout=.1)(x)
        #########################################################################
        # Model 4: 5.6150e-06 error, average networth change per tick = 569+511/(774-60) = .7969
        #########################################################################
        #x1 = TCN(return_sequences=True, nb_filters = 64, dilations = [1, 2, 4, 8, 16, 32], nb_stacks = 1, dropout_rate=.0, kernel_size=2)(i)
        #x1 = TCN(return_sequences=True, nb_filters = 64, dilations = [1, 2, 4, 8, 16, 32], nb_stacks = 1, dropout_rate=.0, kernel_size=2)(x1)
        #x1 = Dense(4, activation='linear')(x1)
        #x2 = LSTM(4, dropout=.3)(i)
        #x = add([x1, x2])
        #o = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(4, activation='linear')(o)
        #########################################################################
        # Model 5: 5.4011e-05 error, average networth change per tick = 354/(774-60) = .4958
        #########################################################################
        #x1 = TCN(return_sequences=True, nb_filters=64, dilations =[1, 2, 4, 8], nb_stacks=1, dropout_rate=.1, kernel_size=2)(i)
        #x2 = Lambda(lambda z: backend.reverse(z, axes=0))(i)
        #x2 = TCN(return_sequences=True, nb_filters=64, dilations =[1, 2, 4, 8], nb_stacks=1, dropout_rate=.1, kernel_size=2)(x2)
        #x = add([x1, x2])
        #x1 = TCN(return_sequences=True, nb_filters=64, dilations =[1, 2, 4, 8], nb_stacks=1, dropout_rate=.1, kernel_size=2)(x)
        #x2 = Lambda(lambda z: backend.reverse(z, axes=0))(x)
        #x2 = TCN(return_sequences=True, nb_filters=64, dilations =[1, 2, 4, 8], nb_stacks=1, dropout_rate=.1, kernel_size=2)(x2)
        #x = add([x1, x2])
        #o = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(4, activation='linear')(o)
        self.m = Model(inputs=i, outputs=o)
        self.m.compile(optimizer='adam', loss='mse') #optimizer and loss can be changed to what we want
        #self.m.compile(optimizer='adam', loss= 'binary_crossentropy')

    def split_data(self, input, moments, lookahead = 1):
        # Split data into groups for training and testing
        x = np.array([])
        y = np.array([])
        input = [(i.open, i.close, i.low, i.high) for i in input]
        input = np.array(input)
        input = np.atleast_2d(input)
        #Normalize data
        input = self.normalization(input, 'custom')
        for i in range(input.shape[0] - moments+1):
            x_values = np.array(input[i:moments + i - lookahead])
            y_values = np.array(input[i+moments-lookahead:i+moments])
            if (x.shape[0] == 0):
                x = x_values
                if len(y_values.shape) == 1:
                    y = [y_values]
                else:
                    y = y_values
            else:
                if i == 1:
                    x = np.concatenate(([x], [x_values]))
                    y = np.concatenate(([y], [y_values]))
                else:
                    x = np.concatenate((x, [x_values]))
                    y = np.concatenate((y, [y_values]))
                #x = np.vstack((x, x_values))
                #y = np.vstack((y, [y_values]))
        #y = np.array(y)
        if len(x.shape) < 3:
            x = x.reshape((1, x.shape[0], x.shape[1]))
        return x, y

    def normalization(self, data, mode = 'default'):
        #To be added to with normalization methods
        if mode == 'default':
            scaler = preprocessing.MinMaxScaler()
            return scaler.fit_transform(data)
        elif mode == 'lognormal':
            normalized_data = list()
            for i, data_pt in enumerate(data):
                inner_data = list()
                for j, value in enumerate(data_pt):
                    if (i == 0):
                        inner_data.append(np.log(value / value))
                    else:
                        inner_data.append(np.log(value / data[i-1][j]))
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        elif mode == 'custom':
            normalized_data = list()
            for i, data_pt in enumerate(data):
                inner_data = list()
                for j, value in enumerate(data_pt):
                    if (i == 0):
                        if (j < 1):
                            inner_data.append(np.log(value / value))
                        else:
                            inner_data.append(np.log(value/data_pt[0]))
                    else:
                        if (j < 1):
                            inner_data.append(np.log(value / data[i - 1][j]))
                        else:
                            inner_data.append(np.log(value/data_pt[0]))
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        elif mode == "percentile":
            normalized_data = list()
            for i, data_pt in enumerate(data):
                inner_data = list()
                for j, value in enumerate(data_pt):
                    if (i == 0):
                        if (j < 1):
                            inner_data.append(0)
                        else:
                            inner_data.append((value - data_pt[0])/data_pt[0])
                    else:
                        if (j < 1):
                            inner_data.append((value - data[i-1][j+1]/ data[i - 1][j+1]))
                        else:
                            inner_data.append((value - data_pt[0])/data_pt[0])
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        return data

    def train(self, data_path):
        def read_data():
            training_data = list()
            values = parse(data_path).values
            for i, val in enumerate(values):
                training_data.append(mrkt_data(val, time=i));
            return training_data
        inputs = read_data()
        x, y = self.split_data(inputs, self.moments)
        #x = np.array(x)
        #x = np.atleast_3d(x)
        #y = np.atleast_2d(y)
        self.m.fit(x, y, epochs=20, validation_split=0.1)
        
    def run_model(self):
        inputs = self.market_history[-self.moments:]
        x, y = self.split_data(inputs, self.moments)
        #x = np.array(x)
        #x = np.atleast_3d(x)
        #y = np.atleast_2d(y)
        y_hat = self.m.predict(x)
        print("Predicting next price to be: ", y_hat[0])
        print("Real next price was: ", y[0])
        y_normal = 10**(y_hat[0][0]/self.market_history[-2].price)*self.market_history[-2].price
        print("Converted back predicted value is ", y_normal)
        print("Read normal value is ", self.market_history[-1].price)
        return y_hat[0]