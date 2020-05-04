from threading import Thread, Lock
import time

from market import market_thread
from util import *
from agent_thread import agent_thread

from simple_agent import simple_agent
# Define Macros:





class trade_platform(Thread):
    class ACB:
        # agent control block
        # job : double check a agent does not double buy, or sell
        # when in synchronized mode,
        #
        # each agent has one
        def __init__(self, agent):
            self.agent = agent
            self.act = action.HOLD
            self.holding = False  # True if brought, and cannot buy more until sell current

            # other functionality such as logging transactions, prices will be left for agent itself
            # but can be added here

    def __init__(self, synchronized=True, random=True, length=1000):
        Thread.__init__(self)

        self.acb = []
        self.market = market_thread(sync=synchronized, random=random, length=length)
        self.synchronized = synchronized

        self.exit = False

    def add_agent(self, ag):
        # needs to be added before Thread start
        # if(self.started):
        if (self._started.is_set()):
            raise Exception("add_agenr : cannot add to runing platform")
        self.acb.append(self.ACB(ag))

    def run(self):
        # init
        self.market.start()
        for ag in self.acb:
            ag.agent.start()

        last_time = self.market.get_time()
        while True:
            if self.market.ended:
                print("Info: platform: simulation finished")
                self.end_market_agent()
                break
            if self.exit:
                self.end_market_agent()
                break
            cur_time = self.market.get_time()
            if cur_time - last_time > 1:
                raise Exception("ERROR : threads sync. Skipped one ",cur_time, last_time)

            # synchronous
            # market will wait until all agents make decisons
            if self.synchronized:
                # feed agents with data
                if cur_time != last_time:
                    market_data = self.market.get_current_value()
                    for ag in self.acb:
                        ag.agent.update_market_history(market_data)  # this will update time as well

                # retrieve action from agent
                for ag in self.acb:
                    ag.act = ag.agent.get_action()

                # check if there is any blocking
                blk = False
                for ag in self.acb:
                    blk = blk or ag.act == action.BLOCK
                if blk:
                    last_time = cur_time
                    continue  # if there is any blocking, do nothing and go back to the loop
                else:
                    self.market.set_next_time_status(True)


                # register agents' transaction to ACB
                for ag in self.acb:
                    if ag.act == action.BUY:
                        if ag.holding:
                            raise Exception("Buying w/ holding")
                        else:
                            ag.holding = True
                    elif ag.act == action.SELL:
                        if ag.holding:
                            ag.holding = False
                        else:
                            raise Exception("Selling w/o holding")
                    # reset their action
                    ag.act = action.BLOCK

            # asynchronous TODO not tested
            # it will ignore blocking, nor control market speed
            # it will only register agents' transaction to ACB when the time change
            # this section needs to be right after the time change(edge)
            if not self.synchronized:
                if cur_time != last_time:
                    # update acb
                    for ag in self.acb:
                        if ag.act == action.BUY:
                            if ag.holding:
                                raise Exception("Buying w/ holding")
                            else:
                                ag.holding = True
                        elif ag.act == action.SELL:
                            if ag.holding:
                                ag.holding = False
                            else:
                                raise Exception("Selling w/o holding")
                        # reset their action
                        ag.act = action.HOLD

                    # feed agents with data
                    if (cur_time != last_time):
                        market_data = self.market.get_current_value()
                        for ag in self.acb:
                            ag.agent.update_market_history(market_data)

                    # retrieve action from agent
                    for ag in self.acb:
                        ag.act = ag.agent.get_action()

            last_time = cur_time

    def set_exit(self, value=True):
        # used to exit the thread
        self.exit = value

    def end_market_agent(self):
        self.market.set_exit()
        for ag in self.acb:
            ag.agent.set_exit()


t = trade_platform(length=1000)
t.add_agent(simple_agent())
t.start()

# foo()