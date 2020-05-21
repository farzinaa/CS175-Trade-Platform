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

#Loss function for punishing wrong decision of buy or sell and magnitude
def custom_loss(y, y_hat):
    mask = tf.math.multiply(y, y_hat)
    #print(mask)
    mask = tf.cast(mask < 0, mask.dtype) * mask
    mask = tf.math.abs(mask)
    yy = tf.math.multiply(y, mask)
    yy_hat = tf.math.multiply(y_hat, mask)
    return backend.mean(backend.square(yy - yy_hat))
    #return (backend.mean(backend.square(yy - yy_hat)) + backend.mean(backend.square(y - y_hat)))/2

class tcn_agent(agent_thread):

    '''
    TCN network takes in the number of moments to look at
    '''
    def __init__(self, moments = 50, batch_size = None, input_dim = 4 ):
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

    def _find_decision(self):
        #Right now these below if statement are for training and testing on the same data set with training on
        #the first self.moments*10 and testing on the remaining
        if len(self.market_history) == self.moments*10:
            self.train()
        if len(self.market_history) > self.moments*10:
            predicted_value = self.run_model()
            #print("Correct guess chance is: ", self.correct_guess*100/(self.time_counter - self.moments*10-1), "%")
            if not self.holding and (predicted_value[0] > 0 and predicted_value[1] > predicted_value[0]):
                self.amount = 100 #amount to buy is set to fix for now but can be changed
                if self.amount < 0:
                    self.amount = 1
                self.act = action.BUY
                self.holding = True
                self.holding_time = self.time_counter
                ###Sergei's algorithm to be input
                #looking at self.time_counter-2 so we can check the real value while guessing the real value
                #as well as be able to use high and low of current market point for Sergei's algorithm
                self.buy_in_price = self.market_history[self.time_counter - 2].price #base price
                #print(self.buy_in_price)
                self.networth -= self.market_history[self.time_counter-2].price * self.amount
                #print("buy  at time " + str(self.time_counter) + "\t price : " + str(
                    #self.market_history[self.time_counter - 1].price))
            elif self.holding and (predicted_value[0] < 0 or predicted_value[1] < predicted_value[0]):
                self.act = action.SELL
                self.networth += self.market_history[self.time_counter - 2].price * self.amount#base price
                ###Sergei's algorithm to be input
                #print(self.market_history[self.time_counter - 2].price) #selling price base
                self.amount = 0 #reset amount to account for varying amounts
                #print("sell at time " + str(self.time_counter) + "\t price : " + str(
                    #self.market_history[self.time_counter - 1].price))
                #print("Current networth is: ", self.networth)
                self.holding = False
            elif self.holding and self.time_counter - self.holding_time > 20:
                self.act = action.SELL
                self.networth += self.market_history[self.time_counter - 2].price * self.amount
                #print("hold 2 long  " + str(self.time_counter) + "\t price : " + str(
                    #self.market_history[self.time_counter - 1].price))
                self.holding = False
            else:
                #print(self.time_counter - self.holding_time)
                self.act = action.HOLD
            return self.act
            print("Current networth = ", self.networth)
        else:
            pass

    def model(self):
        #build model around TCN
        #As a general rule (keeping kernel_size fixed at 2) and dilations increasing with a factor of 2
        #The equation to find the ideal sizes is receptive field = nb_stacks_of_residuals_blocks(nb_stacks) * kernel_size * last_dilation)
        #Each layer adds linearly to the receptive field
        self.built = True
        i = Input(batch_shape=(self.batch_size, self.moments-1, self.input_dim))
        #Model 1: Simple TCN for lower layer and LSTM for upper, set to handel a receptive field of around 64
        #for TCN compressed down for LSTM, build for self.moments between 40-80, networth and accuracy can be used for testing
        #########################################################################
        #x1 = TCN(return_sequences=True, nb_filters=32, nb_stacks = 1, dropout_rate=.0, kernel_size=2)(i)
        #x1 = Dense(4, activation='linear')(x1)
        #o = LSTM(4, dropout=.3)(x1)
        #########################################################################

        # Model 2: 1*10^-6 error, average networth change per tick = 23/(774-60) = shit, build for self.moments between 40-80, networth and accuracy can be used for testing
        #########################################################################
        #i = LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features))(i) #optional addition to try stacked LSTM not added yet
        #x = LSTM(50, dropout=.3, activation='relu')(i)
        #o = Dense(4, activation='softmax')(x)
        #########################################################################

        #Model 3: TCN with LSTM on top with dual bottom layer, build for self.moments between 40-80, networth and accuracy can be used for testing
        #########################################################################
        x1 = TCN(return_sequences=True, nb_filters=64, dilations = [1, 2, 4, 8, 16, 32], nb_stacks=1, dropout_rate=.1, kernel_size=2)(i)
        x2 = Lambda(lambda z: backend.reverse(z, axes=0))(i)
        x2 = TCN(return_sequences=True, nb_filters=64, dilations = [1, 2, 4, 8, 16, 32], nb_stacks=1, dropout_rate=.1, kernel_size=2)(x2)
        x = add([x1, x2])
        o = LSTM(4, dropout=.1)(x)
        #########################################################################

        # Model 4: Layered TCN with LSTM on top, build for self.moments between 40-80, networth and accuracy can be used for testing
        #########################################################################
        #x1 = TCN(return_sequences=True, nb_filters = 64, dilations = [1, 2, 4, 8, 16, 32], nb_stacks = 1, dropout_rate=.1, kernel_size=2)(i)
        #x1 = TCN(return_sequences=True, nb_filters = 64, dilations = [1, 2, 4, 8, 16, 32], nb_stacks = 1, dropout_rate=.1, kernel_size=2)(x1)
        #x1 = Dense(4, activation='linear')(x1)
        #x2 = LSTM(4, dropout=.3)(i)
        #x = add([x1, x2])
        #o = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(4, activation='linear')(o)
        #########################################################################

        # Model 5: Dual TCN layered, build for self.moments between 40-80, networth and accuracy can be used for testing
        #########################################################################
        #x1 = TCN(return_sequences=True, nb_filters=64, dilations =[1, 2, 4, 8, 16, 32], nb_stacks=1, dropout_rate=.1, kernel_size=1)(i)
        #x2 = Lambda(lambda z: backend.reverse(z, axes=0))(i)
        #x2 = TCN(return_sequences=True, nb_filters=64, dilations =[1, 2, 4, 8, 16, 32], nb_stacks=1, dropout_rate=.1, kernel_size=1)(x2)
        #x = add([x1, x2])
        #x1 = TCN(return_sequences=True, nb_filters=64, dilations =[1, 2, 4, 8], nb_stacks=1, dropout_rate=.1, kernel_size=1)(x)
        #x2 = Lambda(lambda z: backend.reverse(z, axes=0))(x)
        #x2 = TCN(return_sequences=True, nb_filters=64, dilations =[1, 2, 4, 8], nb_stacks=1, dropout_rate=.1, kernel_size=1)(x2)
        #x = add([x1, x2])
        #o = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
        #o = Dense(4, activation='linear')(o)
        self.m = Model(inputs=i, outputs=o)
        self.m.compile(optimizer='adam', loss=custom_loss) #optimizer and loss can be changed to what we want

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
        if len(x.shape) < 3:
            x = x.reshape((1, x.shape[0], x.shape[1]))
        return x, y

    def normalization(self, data, mode = 'default'):
        #To be added to with normalization methods
        #Time consuming can be improved
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
                            inner_data.append(np.log(value/ (data[i - 1][0])))
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        elif mode == "percentile":
            return data
            normalized_data = list()
            for i, data_pt in enumerate(data):
                inner_data = list()
                for j, value in enumerate(data_pt):
                    inner_data.append(value)
                normalized_data.append(inner_data)
            return np.array(normalized_data)
        return data

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
            inputs = self.market_history[-self.moments*10:]
        x, y = self.split_data(inputs, self.moments)
        self.m.fit(x, y, epochs=20, validation_split=0.1)
        
    def run_model(self):
        #if self.log_percentage != []:
            #inputs = self.get_log_percentages(-self.moments)
        #else:
        inputs = self.market_history[-self.moments:]
        x, y = self.split_data(inputs, self.moments)
        y_hat = self.m.predict(x)
        #print("Predicting next price to be: ", y_hat[0])
        #print("Real next price was: ", y[0])
        if (y[0][0]*y_hat[0][0] > 0): #checks if agent guessed right on opening going up or down
            self.correct_guess +=1
        y_normal = 0
        if self.percentage != []:
            y_normal = (y_hat[0][0])*self.market_history[-2].price
        else:
            y_normal = 10*(y_hat[0][0]/self.market_history[-2].price)*self.market_history[-2].price
        #print("Converted back predicted value is ", y_normal) #To be changed to function to account for different normalized data
        #print("Read normal value is ", self.market_history[-1].price)
        return y_hat[0]