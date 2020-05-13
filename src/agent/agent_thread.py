import time
from threading import Thread
from ...src.util.util import *

class agent_thread(Thread):
    def __init__(self, sync=True,         currently_holding = False, buy_in_price = 0, time_ = 0, market_history = []):
        Thread.__init__(self)

        if time_ != len(market_history):
            time_ = len(market_history)
        print("Warning: unmatch time and market_history len. set time = market_history len")
        self.time_counter = time_

        self.synchronized = sync
        self.market_history = market_history

        self.act = action.HOLD
        self.offer_price = None

        self.holding = currently_holding
        self.buy_in_price = buy_in_price

        self.next_time_flag = False

        self.exit = False

    def _make_decision(self):
        # private function. To signal that a decision is made
        self.next_time_flag = False

    def _need_decision(self):
        return self.next_time_flag

    def _find_decision(self):
        # An agent should overwrite run or _find_decision
        # the main logic
        time.sleep(0.1)
        self.act = action.HOLD
        return self.act

    def get_action(self):
        # return action
        # used by tp
        # return action.HOLD
        return self.act,self.offer_price
    '''
    def set_time(self, value):
        # for debug only
        if (value != self.time_counter):
            if (value - self.time_counter != 1):
                print("wanrning : set_time : increment > 1")
            self.next_time_flag = True
            self.time_counter = value
    '''
    def next_time(self):

        if self.synchronized:
            while (self.next_time_flag):
                pass
                # to prevent clock tick more than once
                # and to prevent time from changing while running
        self.time_counter += 1
        self.next_time_flag = True
        self.act = action.BLOCK

    ''' #USE WITH DISCRETION -_- DEBUG ONLY
    def set_market_history(self, time, data):
        #set market history at one point
        #USE WITH DISCRETION -_- DEBUG ONLY

        self.market_history[time] = data

    def set_all_market_history(self, data):
        # set entire market history
        #USE WITH DISCRETION -_- DEBUG ONLY
        self.market_history = data
        self.time_counter = len(data)
    '''

    def update_market_history(self, data):
        # undate for 1 unit of time
        self.next_time()
        self.market_history.append(data)

    def set_exit(self, value=True):
        # used to exit this agent thread
        self.exit = value

    def run(self):
        # An agent should overwrite run or _find_decision
        print("agent started")
        last_time = 0
        while True:
            if self.time_counter - last_time > 1:
                raise Exception("ERROR: Agent: not Sync ")
            if self.exit:
                print("client terminated")
                break
            if not self._need_decision():
                # the market has not updated. i.e. the time not changed
                continue
            # make a simply move

            # first we set action = block, so the market knows we need more time
            # self.act = action.BLOCK
            #This line was moved to next_time function

            # _find_decision
            self.act = self._find_decision()

            self._make_decision()
            last_time = self.time_counter
