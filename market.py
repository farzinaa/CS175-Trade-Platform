from threading import Thread,Lock
import time
import  numpy as np
from util import *

class market():
    '''/*
     * This is the base object for market
     * it will only consider price
    */'''
    def __init__(self , random=False,length=0):
        self.time_counter = 0
        self.market_values = list()
        self.length = length

        self.ended = False # status of simulation
        #randomly generate price
        if random:
            r = np.random.random(self.length) * 5 + 20
            for i,d in enumerate(r):
                self.market_values.append(mrkt_data(d,i))



    def get_current_value(self):
        return self.market_values[self.time_counter]

    def get_all_value(self):
        # historical price, including current
        return list(self.market_values)

    def get_ranged_value(self, range):
        # return market value from current time_counter to time_counter-range
        # if range = 0, return entire list
        if range < 0:
            raise Exception("get_ranged_value : negative")
        if range == 0:
            return self.get_all_value()
        if(range >= self.time_counter + 1):
            return np.append(np.array([mrkt_data(0, None)]*(range-self.time_counter-1)), np.array(self.market_values[0:self.time_counter+1]))
        return np.array(self.market_values[self.time_counter - range : self.time_counter])

    def set_time(self,value):
        # to skip forward, meant for debug, no guarantee
        # be careful for the mutex lock
        self.time_counter = value

    def get_time(self):
        return  self.time_counter

    def next_time(self):
        # step into next unit of time
        if not self.ended:
            self.time_counter += 1
        else:
            print("warning: market: simulation ended time = ", self.time_counter)
        if self.time_counter == self.length - 1:
            print("warning: market: simulation ended time = ", self.time_counter)
            self.ended = True



class market_thread(market, Thread):
    def __init__(self, sync = True,  animation_speed = 1.0, random=False,length=0, graph=False, graph_span=50):
        '''/*
         * sync: True: wait for all agent finish to step into next unit of time
         * number_of_agent : in case you want more than one agent

         * animation_speed: if not sync, the market will move forward animation_speed unit per second
         * eg: if animation_speed = 0.5 , then market will move forward 1 unit of time every 2 seconds.

         * graph: enable real time market grpah
         * graph_span: How many past units of time are included in the graph. 0 : Entire history
        */'''
        market.__init__(self, random=random, length=length)
        Thread.__init__(self)
        self.sync = sync
        #self.next_time_status = False
        self.next_time_mutex = Lock()

        self.exit = False

        self.animation_speed = animation_speed


    def set_next_time_status(self, value=True):

        self.next_time_mutex.acquire()
        while (self.next_time_mutex.locked()):
            pass
        #self.next_time_status = value

    def set_exit(self, value=True):
        # used to end the thread
        self.exit = value

    def run(self) -> None:
        while True:
            if self.ended: # if the simulation runs out
                break
            if self.exit: # be terminated
                break
            if(self.sync):
                if(self.next_time_mutex.locked()):
                    self.next_time()
                    self.next_time_mutex.release()

            else:
                time.sleep(1/self.animation_speed)
                self.next_time()

# m = market_thread(random=True,length=1000)
# m.start()
# print()
# m.set_next_time_status()
# m.get_time()
# l = m.get_ranged_value(50)
# for i in l:
#     print(i.price)