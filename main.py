
from src.agent.simple_agent import simple_agent
from src.trade_platform.trade_platform import trade_platform

if __name__ == "__main__":
    t = trade_platform(length=5000, data_path='sample_data/a.csv', enable_plot=True)
    t.add_agent(simple_agent())
    t.start()

'''/*
 * Synchronous / Asynchronous
 * Market / Agent 
 *    
 * Trade platform
 *
*/'''
