from enum import Enum

class mrkt_data:
    # store all the data at a unit of time
    def __init__(self, args,time = 0 ):
        self.price = args[0]
        if len(args)> 1:
            # Back ward compatibility for plotting
            self.open = args[0]
            self.close= args[1]
            self.low  = args[2]
            self.high = args[3]
        else:
            self.open = None
            self.close = None
            self.low = None
            self.high = None
        self.time  = time # time here is only as a reference. Use with discretion

class action(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2
    BLOCK = 3



# unrelated
LENGTH = 1000  # Duration of test.
TIME_PER_TRADE = 1  # the intervel between each test
# If TIME_PER_TRADE = 1, then it can make 1 trade every unit of time.


# Macros for testing
RANDOM_MARKET_VALUE = True  # market will generate ranndam value
IMPORT_VALUE_FROM_FILE = False  # if not RANDOM_MARKET_VALUE, market object will import
IMPORT_PATH = ''

'''/*
 *
 *
 *
*/'''
