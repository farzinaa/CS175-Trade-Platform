from threading import Thread
from src.util.util import *

class agent_thread(Thread):
    def __init__(self, sync=True, currently_holding=0, buy_in_price=0, time_=0, market_history=[]):
        Thread.__init__(self)
        self.buy_in_price = buy_in_price
        self.currently_holding = currently_holding

        if time_ != len(market_history):
            time_ = len(market_history)
            print("Warning: unmatch time and market_history len. set time = market_history len")
        self.time_counter = time_

        self.synchronized = sync
        self.market_history = market_history

        self.act = action.BLOCK
        self.holding = False
        self.next_time_flag = False

        self.exit = False

    def _make_decision(self):
        # private function. To signal that a decision is made
        self.next_time_flag = False

    def _need_decision(self):
        return self.next_time_flag

    def get_action(self):
        # return action
        return action.HOLD

    def set_time(self, value):
        # for debug only
        if (value != self.time_counter):
            if (value - self.time_counter != 1):
                print("wanrning : set_time : increment > 1")
            self.next_time_flag = True
            self.time_counter = value

    def next_time(self):

        if self.synchronized:
            while (self.next_time_flag):
                pass
                # to prevent clock tick more than once
                # and to prevent time from changing while running
        self.time_counter += 1
        self.next_time_flag = True

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
        self.exit = value

    def _find_decision(self):
        # the main logic
        return action.HOLD
    def run(self):
        # a simply AI
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
            self.act = action.BLOCK
            self.act = self._find_decision()
            last_time = self.time_counter

            self._make_decision()
