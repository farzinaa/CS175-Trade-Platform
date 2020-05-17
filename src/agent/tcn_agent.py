from trade_platform.src.agent.agent_thread import agent_thread
from ...src.util.util import *
import numpy as np
from ...src.util.mrkt_data import mrkt_data
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from sklearn import preprocessing
from ...src.util.Data_parsing.data_parsing import parse
import keras.backend as K
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Convolution1D, add

#From https://github.com/philipperemy/keras-tcn
from tcn import TCN, tcn_full_summary

class tcn_agent(agent_thread):

    '''
    TCN network takes in the number of moments to look at
    '''
    def __init__(self, moments = 100, batch_size = None, input_dim = 4 ):
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

    def _find_decision(self):
        if len(self.market_history) > self.moments:
            predicted_value = self.run_model()
            print("Open is : ", self.market_history[self.time_counter - 1].open)
            print("Close is : ", self.market_history[self.time_counter - 1].close)
            print("Low is : ", self.market_history[self.time_counter - 1].low)
            print("High is : ", self.market_history[self.time_counter - 1].high)
            if not self.holding and (predicted_value[1] - predicted_value[0] > 0 and predicted_value[1] > 0):
                self.amount = 10
                self.act = action.BUY
                self.holding = True
                self.holding_time = self.time_counter
                self.buy_in_price = self.market_history[self.time_counter - 1].price
                self.networth -= self.market_history[self.time_counter - 1].price * self.amount
                #print("buy  at time " + str(self.time_counter) + "\t price : " + str(
                    #self.market_history[self.time_counter - 1].price))
            elif self.holding and (predicted_value[1] < 0 or predicted_value[0] - predicted_value[1] > 0):
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
        self.built = True
        i = Input(batch_shape=(self.batch_size, self.moments-1, self.input_dim))
        x1 = TCN(return_sequences=True)(i)
        o = TCN(return_sequences=False)(x1)
        o = Dense(4)(o)
        self.m = Model(inputs=i, outputs=o)
        self.m.compile(optimizer='adam', loss='mse') #optimizer and loss can be changed to what we want

    def split_data(self, input, moments, lookahead = 1):
        # Split data into groups for training and testing
        x = np.array([])
        y = np.array([])
        input = [(i.open, i.close, i.low, i.high) for i in input]
        input = np.array(input)
        input = np.atleast_2d(input)
        #Normalize data
        input = self.normalization(input, 'lognormal')
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
            return preprocessing.normalize(data)
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
        return preprocessing.normalize(data)

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
        self.m.fit(x, y, epochs=10, validation_split=0.2)
    def run_model(self):
        inputs = self.market_history[-self.moments:]
        x, y = self.split_data(inputs, self.moments)
        #x = np.array(x)
        #x = np.atleast_3d(x)
        #y = np.atleast_2d(y)
        y_hat = self.m.predict(x)
        print("Predicting next price to be: ", y_hat[0])
        print("Real next price was: ", y[0])
        return y_hat[0]