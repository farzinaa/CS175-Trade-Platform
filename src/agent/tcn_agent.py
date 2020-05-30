from trade_platform.src.agent.agent_thread import agent_thread
from ...src.util.util import *
import numpy as np
from ...src.util.mrkt_data import mrkt_data
from tensorflow.keras.layers import Dense, add, Lambda, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, Conv1D
from tensorflow.keras.layers import concatenate, LSTM, Activation, multiply, Convolution1D, MaxPooling1D, Flatten
from tensorflow.keras import Input, Model, backend
from tensorflow.keras.losses import binary_crossentropy
from sklearn import preprocessing
from ...src.util.Data_parsing.data_parsing import parse
import tensorflow as tf
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
#From https://github.com/philipperemy/keras-tcn
from tcn import TCN, tcn_full_summary

def wave_net_activation(x): #https://www.kaggle.com/christofhenkel/temporal-cnn
    # type: (Layer) -> Layer
    """This method defines the activation used for WaveNet
    described in https://deepmind.com/blog/wavenet-generative-model-raw-audio/
    Args:
        x: The layer we want to apply the activation to
    Returns:
        A new layer with the wavenet activation applied
    """
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return multiply([tanh_out, sigm_out])

#Loss function for punishing wrong decision of buy or sell and magnitude
def custom_loss(y, y_hat):
    mask = tf.math.multiply(y, y_hat)
    #print(mask)
    mask = tf.cast(mask < 0, mask.dtype) * mask
    mask = tf.math.abs(mask)
    yy = tf.math.multiply(y, mask)
    yy_hat = tf.math.multiply(y_hat, mask)
    #return backend.mean(backend.square(yy - yy_hat))
    return (backend.mean(backend.square(yy - yy_hat)) + backend.mean(backend.square(y - y_hat)))/2

class tcn_agent(agent_thread):

    '''
    TCN network takes in the number of moments to look at
    '''
    def __init__(self, moments = 17, batch_size = None, input_dim = 7):
        agent_thread.__init__(self)
        self.moments = moments #Number of moments looked back to decide next value
        self.holding_time = 0
        self.batch_size = batch_size
        self.input_dim = input_dim #Dimensions of input default to 1
        self.built = False
        self.model() #Compile model
        self.training = True
        self.networth = 0
        self.amount = 0 #stocks bought
        self.correct_guess = 0 #tracker to see when guessing correct of open prices
        self.sergei = 0
        self.arima = []
        self.features = []
        self.scaler = 0

    def _find_decision(self):
        #Right now these below if statement are for training and testing on the same data set with training on
        #the first self.moments*10 and testing on the remaining
        if len(self.market_history) == self.moments*100 + 50:
            self.train()
        if len(self.market_history) > self.moments*100 + 50:
            predicted_value = self.run_model()
            print("Correct guess chance is: ", self.correct_guess*100/(self.time_counter - self.moments*100-50), "%")
            if not self.holding and predicted_value[0] > self.market_history[self.time_counter-2].price:#(predicted_value[0] > 0 and predicted_value[1] > 0):
                self.amount = 100 #amount to buy is set to fix for now but can be changed
                if self.amount < 0:
                    self.amount = 1
                self.act = action.BUY
                self.holding = True
                self.holding_time = self.time_counter
                ###Sergei's algorithm
                # looking at self.time_counter-2 so we can check the real value while guessing the real value
                # as well as be able to use high and low of current market point for Sergei's algorithm
                if (self.market_history[self.time_counter - 3].close >= self.market_history[self.time_counter - 3].open
                        and self.market_history[self.time_counter - 2].close <= self.market_history[self.time_counter - 2].open
                        and (self.market_history[self.time_counter - 3].high >= self.market_history[self.time_counter - 1].low
                        and self.market_history[self.time_counter - 3].high <= self.market_history[self.time_counter - 1].high)):
                    self.buy_in_price = self.market_history[self.time_counter - 3].high
                    print("Used Sergei's algorithm and gained/lost ", self.market_history[self.time_counter - 2].price - self.buy_in_price, " per share")
                    self.sergei += (self.market_history[self.time_counter - 2].price - self.buy_in_price) * self.amount
                else:
                    self.buy_in_price = self.market_history[self.time_counter - 2].price #base price
                #print(self.buy_in_price)
                self.networth -= self.buy_in_price * self.amount
                print("buy  at time " + str(self.time_counter) + "\t price : " + str(self.buy_in_price))
            elif self.holding and predicted_value[0] < self.market_history[self.time_counter-2].price:#(predicted_value[0] < 0 or predicted_value[1] < 0):
                self.act = action.SELL
                ###Sergei's algorithm
                if (self.market_history[self.time_counter - 3].close <= self.market_history[self.time_counter - 3].open
                        and self.market_history[self.time_counter - 2].close >= self.market_history[self.time_counter - 2].open
                        and (self.market_history[self.time_counter - 3].low >= self.market_history[self.time_counter - 1].low
                        and self.market_history[self.time_counter - 3].low <= self.market_history[self.time_counter - 1].high)):
                    self.sell_price = self.market_history[self.time_counter - 3].low
                    print("Used Sergei's algorithm and gained/lost ", self.buy_in_price - self.market_history[self.time_counter - 2].price, " per share")
                    self.sergei += (self.buy_in_price - self.market_history[self.time_counter - 2].price) * self.amount
                else:
                    self.sell_price = self.market_history[self.time_counter - 2].price
                self.networth += self.sell_price * self.amount#base price
                #print(self.market_history[self.time_counter - 2].price) #selling price base
                self.amount = 0 #reset amount to account for varying amounts
                print("sell at time " + str(self.time_counter) + "\t price : " + str(
                    self.sell_price))
                print("Current networth is: ", self.networth)
                print("Sergei's networth: ", self.sergei)
                self.holding = False
            elif self.holding and self.time_counter - self.holding_time > 20:
                self.act = action.SELL
                self.networth += self.market_history[self.time_counter - 2].price * self.amount
                print("hold 2 long  " + str(self.time_counter) + "\t price : " + str(
                    self.market_history[self.time_counter - 1].price))
                self.holding = False
            else:
                print(self.time_counter - self.holding_time)
                self.act = action.HOLD
            return self.act
        else:
            pass

    def model(self):
        #build model around TCN
        #As a general rule (keeping kernel_size fixed at 2) and dilations increasing with a factor of 2
        #The equation to find the ideal sizes is receptive field = nb_stacks_of_residuals_blocks(nb_stacks) * kernel_size * last_dilation)
        #Each layer adds linearly to the receptive field
        self.built = True
        i = Input(batch_shape=(self.batch_size, self.moments-1, self.input_dim))
        #Model 1: Simple TNN self-build
        #for TCN compressed down for LSTM, build for self.moments between 40-80, networth and accuracy can be used for testing
        #########################################################################

        #x = Conv1D(filters= 64, padding = 'causal', dilation_rate= 2, kernel_size= 2)(i)
        #x = MaxPooling1D(2, strides=2)(x)
        #x = Conv1D(filters= 32, padding= 'causal', dilation_rate= 2, kernel_size= 2)(x)
        #x = MaxPooling1D(2, strides=2)(x)
        #x = Dropout(.2)(x)
        #x = Flatten()(x)
        #x = Dense(16)(x)
        #x = Dropout(.2)(x)
        #x = Activation('relu')(x)
        #x = Dense(1)(x)
        #o = Activation('relu')(x)

        #########################################################################
        #x = TCN(return_sequences=True, nb_filters=128, dilations=[1, 2, 4, 8, 16], nb_stacks=2, dropout_rate=.3,
        #        kernel_size=8)(i)
        #x2 = TCN(return_sequences=True, nb_filters=128, dilations=[1, 2, 4, 8], nb_stacks=2, dropout_rate=.27,
        #        kernel_size=8, activation= wave_net_activation)(i)
        #x = add([x1,x2])
        #x = LSTM(4, dropout=.3, return_sequences=True)(x)
        #x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(1, activation='linear')(x)
        # Model 2: 1*10^-6 error, average networth change per tick = 23/(774-60) = shit, build for self.moments between 40-80, networth and accuracy can be used for testing
        #########################################################################
        #x = LSTM(256, dropout=.3, return_sequences=True)(i) #optional addition to try stacked LSTM not added yet
        #x = LSTM(16, dropout=.3, activation = wave_net_activation)(x)
        #o = Dense(2, activation='linear')(x)
        #########################################################################

        #Model 3: LSTM Model
        #########################################################################
        #x = TCN(return_sequences=True, nb_filters=8, dilations=[1, 2, 4, 8, 16], nb_stacks=2, dropout_rate=.3,
        #        kernel_size=4, activation= wave_net_activation)(i)
        #x = TCN(return_sequences=True, nb_filters=16, dilations=[1, 2, 4], nb_stacks=2, dropout_rate=.3,
        #        kernel_size=2, activation=wave_net_activation)(x)
        #x = Dense(64, activation='linear')(x)
        #x = TCN(return_sequences=True, nb_filters=128, dilations=[1, 2, 4], nb_stacks=1, dropout_rate=.3,
        #        kernel_size=4)(x)
        #x = LSTM(4, return_sequences=True, dropout=.3)(x)#, activation= 'relu')(x)#wave_net_activation)(x)
        #x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(2, activation='linear')(x)
        #########################################################################

        # Model 4: Layered TCN with LSTM on top, build for self.moments between 20-40, networth and accuracy can be used for testing
        #x1 = TCN(return_sequences=True, nb_filters = 64, dilations = [1, 2, 4, 8], nb_stacks = 2, dropout_rate=.3, kernel_size=2)(i)
        #x1 = LSTM(16, return_sequences=True, dropout=.3)(x1)
        #x2 = TCN(return_sequences=True, nb_filters = 64, dilations=[1, 2, 4, 8], nb_stacks=2, dropout_rate=.3,
        #        kernel_size=2)(i)
        #x = add([x1, x2])
        #x = Dense(16, activation='linear')(x)
        #x = TCN(return_sequences=True, nb_filters=16, dilations=[1, 2, 4, 8], nb_stacks=1, dropout_rate=.1,
        #        kernel_size=2, activation= wave_net_activation)(x)
        #x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(2, activation='linear')(x)
        #########################################################################

        # Model 5: Dual TCN layered, build for self.moments between 40-80, networth and accuracy can be used for testing
        #########################################################################
        #x1 = TCN(return_sequences=True, nb_filters=15, dilations =[1, 2, 4, 8, 16], padding='same', nb_stacks=1, dropout_rate=.3, kernel_size=3)(i)
        #x2 = TCN(return_sequences=True, nb_filters=5, dilations =[1, 2, 4, 8], padding='same', nb_stacks=1, dropout_rate=.3, kernel_size=3, activation= wave_net_activation)(i)
        #x1 = Dense(5, activation='linear')(x1)
        #x = add([x1,x2])
        #x = LSTM(5, return_sequences=True, dropout=.1)(x)
        #x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(2, activation='linear')(x)

        #Model 6:
        #########################################################################
        #x = TCN(return_sequences=True, nb_filters=64, dilations=[1, 2, 4, 8], nb_stacks=1, dropout_rate=.3,  activation= wave_net_activation,
        #         kernel_size=2)(i)
        #x = Dense(32, activation='linear')(x)
        #x1 = LSTM(8, return_sequences=True, dropout=.1)(x)
        #x2 = TCN(return_sequences=True, nb_filters=8, dilations=[1, 2, 4, 8], nb_stacks=1, dropout_rate=.3)(i)
        #x = add([x1,x2])
        #o = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(2, activation='linear')(o)
        #########################################################################

        # Model 7: Layered TCN with LSTM on top, build for self.moments between 20-40, networth and accuracy can be used for testing
        #########################################################################
        x1 = TCN(return_sequences=False, nb_filters = 15, dilations = [1, 2, 4], nb_stacks = 2, dropout_rate=.3, kernel_size=2)(i)
        x2 = Lambda(lambda z: backend.reverse(z, axes=-1))(i)
        x2 = TCN(return_sequences=False, nb_filters=15, dilations=[1, 2, 4], nb_stacks=2, dropout_rate=.1,
                 kernel_size=2)(x2)
        #x1 = add([x1,x2])
        #x1 = TCN(return_sequences=True, nb_filters = 64, dilations = [1, 2, 4, 8], nb_stacks = 2, dropout_rate=.3, kernel_size=8)(x1)
        #x2 = LSTM(15, return_sequences=True, dropout=.3)(i)
        ##x2 = LSTM(64, return_sequences=True, dropout=.3)(x2)
        x = add([x1, x2])
        #x = Dense(8, activation='linear')(x)
        #x = TCN(return_sequences=False, nb_filters=4, dilations=[1, 2, 4], nb_stacks=1, dropout_rate=.3, kernel_size=4, activation= wave_net_activation)(x)
        #x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        o = Dense(1, activation='linear')(x)
        #########################################################################
        # Model 7: Layered TCN with LSTM on top, build for self.moments between 20-40, networth and accuracy can be used for testing
        #########################################################################
        #x = TCN(return_sequences=True, nb_filters=128, dilations=[1, 2, 4, 8, 16], nb_stacks=2, dropout_rate=.3,
        #         kernel_size=8)(i)
        #x1 = TCN(return_sequences=True, nb_filters = 64, dilations = [1, 2, 4, 8, 16], nb_stacks = 2, dropout_rate=.3, kernel_size=8)(x)
        #x2 = LSTM(128, return_sequences=True, dropout=.3)(i)
        #x2 = LSTM(64, return_sequences=True, dropout=.3)(x)
        #x = add([x1, x2])
        #x = Dense(32, activation='linear')(x)
        #x = TCN(return_sequences=True, nb_filters=16, dilations=[1, 2, 4], nb_stacks=1, dropout_rate=.3,
        #        kernel_size=2, activation=wave_net_activation)(x)
        #x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(2, activation='linear')(x)

        self.m = Model(inputs=i, outputs=o)
        self.m.compile(optimizer='adam', loss='mse') #optimizer and loss can be changed to what we want
        #########################################################################

    def arima_feature(self, i):
        if len(self.arima) < 25:
            m = ARIMA(self.arima[:i], order=(5, 1, 0))
        else:
            m = ARIMA(self.arima[i-25:i], order=(5,1,0))
        m_fit = m.fit(disp=0)
        output = m_fit.forecast()
        #print(i)
        #print(output[0][0])
        return output[0][0]

    def get_technical_indicators(self, i):
        if len(self.market_history) >= 8 and i >= 8:
            last7 = self.market_history[i-8:i-1]
        else:
            last7 = self.market_history[:]
        if len(self.market_history) >= 28 and i >= 28:
            last21 =  self.market_history[i-28:i-1]
        else:
            last21 = self.market_history[:]
        last7 = [i.price for i in last7]
        last21 = [i.price for i in last21]
        malast7 = np.mean(last7)
        malast21 = np.mean(last21)
        #explast7 = pd.ewm(last7)
        #explast21 = pd.ewm(last21)

        return malast7, malast21#, explast7, explast21)


    def split_data(self, size, moments, lookahead = 1):
        # Split data into groups for training and testing
        self.prepare_data()
        input = self.features[-size:]
        input = np.array(input)
        input = np.atleast_2d(input)
        x = np.array([])
        y = np.array([])
        #print(input)
        #Normalize data)
        input = self.normalization(input, 'default')
        #print(input)
        for i in range(input.shape[0] - moments+1):
            x_values = np.array(input[i:moments + i - lookahead])
            y_values = np.array(input[i+moments-lookahead:i+moments][0][0])
            #print(y_values.shape)
            #print(y_values)
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
        if len(x.shape) < 3:
            x = x.reshape((1, x.shape[0], x.shape[1]))
        return x, y

    def normalization(self, data, mode = 'default'):
        #To be added to with normalization methods
        #Time consuming can be improved
        if mode == 'default':
            if self.scaler == 0:
                self.scaler = preprocessing.StandardScaler()
                return self.scaler.fit_transform( data )
            return self.scaler.transform( data )
            #if self.scaler == 0:
            #    self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            #    self.scaler = self.scaler.fit(data)
            #    return self.scaler.fit_transform(data)
            #return self.scaler.transform(data)
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
                            inner_data.append(np.log(value / data[i - 1][1]))
                        else:
                            inner_data.append(np.log(value/ (data[i - 1][0])))
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        elif mode == "percentile":
            normalized_data = list()
            for i, data_pt in enumerate(data):
                inner_data = list()
                for j, value in enumerate(data_pt):
                    if (i == 0):
                        if (j < 1):
                            inner_data.append((value / value)-1)
                        else:
                            inner_data.append((value / data_pt[j])-1)
                    else:
                        if (j < 1):
                            inner_data.append((value / data[i - 1][j])-1)
                        else:
                            inner_data.append((value / data[i-1][j])-1)#data[i - 1][0])-1)
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        return data

    def prepare_data(self):
        difference = len(self.features) - len(self.market_history) + 50
        #print(difference)
        if difference < 0:
            input = [[i.open, i.close, i.low, i.high] for i in self.market_history[difference:]]
            # input = [[i.open, i.close, (i.high-i.close+offset)*2/(i.close+i.high)] for i in input]
            # input = [[i.open, i.close] for i in input]
            for i in range(len(input)):
                input[i].append(self.arima_feature(len(self.arima) - len(input) + i))
                input[i].append(self.get_technical_indicators(len(self.arima) - len(input) + i)[0])
                input[i].append(self.get_technical_indicators(len(self.arima) - len(input) + i)[1])
                self.features.append(input[i])
            #self.features.append(input)

    def train(self, data_path = None):
        def read_data():
            if data_path == None:
                return data_path
            training_data = list()
            values = parse(data_path).values
            for i, val in enumerate(values):
                training_data.append(mrkt_data(val, time=i));
            return training_data
        inputs = read_data()
        if inputs == None:
            inputs = self.market_history[100:]
        x, y = self.split_data(self.moments*100, self.moments)
        #print(x)
        #print(y)
        self.m.fit(x, y, epochs=400, validation_split=0.1)
        
    def run_model(self):
        #if self.log_percentage != []:
            #inputs = self.get_log_percentages(-self.moments)
        #else:
        #inputs = self.market_history[-self.moments:]

        x, y = self.split_data(self.moments, self.moments)
        #print(x)
        #print(y)
        y_hat = self.m.predict(x)
        print("Predicting next price to be: ", y_hat[0][0])
        print("Real next price was: ", y)
        if ((y-self.market_history[-2].price) * (y_hat[0][0]-self.market_history[-2].price) > 0): #checks if agent guessed right on opening going up or down
            self.correct_guess +=1
        y_normal = 0
        if self.percentage != []:
            y_normal = (y_hat[0][0])*self.market_history[-2].price
        else:
            y_normal = 10*(y_hat[0][0]/self.market_history[-2].price)*self.market_history[-2].price
        #print("Converted back predicted value is ", y_normal) #To be changed to function to account for different normalized data
        #print("Read normal value is ", self.market_history[-1].price)
        return y_hat[0]

    def update_market_history(self, data):
        # undate for 1 unit of time
        self.next_time()
        self.market_history.append(data)
        self.arima.append(data.price)

    def set_exit(self, value=True):
        # used to exit this agent thread
        self.exit = value
        print("Sergei's networth: ", self.sergei)
        print("Networth: ", self.networth)