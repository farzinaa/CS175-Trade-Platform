# CS175-Trade-Platform

Provide trade logic, as submodule of CS175 Project.\
<h3>
Update 5/13: Changes to fit candle stick data
</h3>

- Fix Bugs
- Change default mrkt_data to 4 fields:open, close, low, high. But it also backward support for only one field(price), depending on the format of input file.
- Add method to customize mrkt_data. By pass a new __init__ in `trade_platform` 
- The agent now provide a price(offer) when return a SELL or BUY action, and the trade_logic 
will check if the offered price is valid (between that day's low and high) Change member variable offer_price to enable this.  