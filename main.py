

from simple_agent import simple_agent
from trade_platform import trade_platform

if __name__ == "__main__":
    t = trade_platform(length=5000)
    t.add_agent(simple_agent())
    t.start()

'''/*
 * Synchronous / Asynchronous
 * Market / Agent 
 *    
 * Trade platform
 *
*/'''
