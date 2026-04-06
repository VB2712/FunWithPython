#===== Project Overview =====
#A general project focused on working with an order book to generate various analytics
#Work with a class object

#===== Part I - Order book class =====
#Create an order book class that stores bids and asks messages and create an order book

class OrderBook:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.order_map = {} 

    def add_order(self, side, price, quantity, order_id):
        if side == "BID":
            self.bids[price] = self.bids.get(price, 0) + quantity #We use the get() method which is cleaner than adding further if then conditions

        elif side == "ASK":
            self.asks[price] = self.asks.get(price, 0) + quantity

        self.order_map[order_id] = {'price': price, 'side': side, 'quantity': quantity}

    def cancel_orders(self, order_id):

        order_info = self.order_map.get(order_id)

        if order_info:
            price=order_info["price"]
            side=order_info["side"]
            quantity=order_info["quantity"]

            if side == "BID":
                self.bids[price] -= quantity
                if self.bids[price] <= 0: del self.bids[price]
            else:
                self.asks[price] -= quantity
                if self.asks[price] <= 0: del self.asks[price] 

            del self.order_map[order_id]

    def get_best_bid(self):
        if not self.bids:
            return None
        else:
            return max(self.bids.keys())

    def get_best_ask(self):
        if not self.asks:
            return None
        else:
            return min(self.asks.keys())

    def get_mid(self):
        best_bid=self.get_best_bid()
        best_ask=self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None, None
        else:
            spread=best_ask-best_bid
            mid=(best_bid+best_ask)/2
            return spread, mid

    def execute_order(self,price,size):
        if price in self.asks:
            self.asks[price] -= size
            if self.asks[price] <= 0: del self.asks[price]

        elif price in self.bids:
            self.bids[price] -= size
            if self.bids[price] <= 0: del self.bids[price]

    def buy_order(self,quantity_to_buy):
        
        if quantity_to_buy > sum(self.asks.values()): return "Error - Not enough liquidity"

        remaining_quantity=quantity_to_buy
        total_cost=0

        for price in sorted(self.asks.keys()):
            quantity_at_price=self.asks[price]
            if quantity_at_price >= remaining_quantity:
                total_cost += remaining_quantity * price
                break
            else:
                total_cost += quantity_at_price * price
                remaining_quantity -= quantity_at_price

        return total_cost/quantity_to_buy

    def sell_order(self,quantity_to_sell):
        
        if quantity_to_sell > sum(self.bids.values()): return "Error - Not enough liquidity"

        remaining_quantity=quantity_to_sell
        total_cost=0

        for price in sorted(self.bids.keys(), reverse=True):
            quantity_at_price=self.bids[price]
            if quantity_at_price >= remaining_quantity:
                total_cost += remaining_quantity * price
                break
            else:
                total_cost += quantity_at_price * price
                remaining_quantity -= quantity_at_price

        return total_cost/quantity_to_sell


#===== Part II - Process dummy events =====

# Sample Event Data
events = [
    {"type": "ADD", "side": "BID", "price": 100.50, "size": 10, "order_id": 1},
    {"type": "ADD", "side": "ASK", "price": 100.55, "size": 5, "order_id": 2},
    {"type": "ADD", "side": "ASK", "price": 100.60, "size": 10, "order_id": 3},
    {"type": "ADD", "side": "ASK", "price": 100.65, "size": 20, "order_id": 4},
    {"type": "CANCEL","order_id":1},
    {"type": "ADD", "side": "BID", "price": 100.50, "size": 10, "order_id": 5},
    {"type": "ADD", "side": "BID", "price": 100.49, "size": 100, "order_id": 6},
    {"type": "EXECUTE", "price": 100.55, "size": 2}
]

#Initialise the class
ob=OrderBook()

#Process the events
for event in events:
    etype = event["type"]
    if etype == "ADD":
        ob.add_order(event["side"],event["price"],event["size"],event["order_id"])
    elif etype == "CANCEL":
        ob.cancel_orders(event["order_id"])
    elif etype == "EXECUTE":
        ob.execute_order(event["price"],event["size"])

    spread, mid = ob.get_mid()
    side = event.get("side", "N/A")
    price = event.get("price", "N/A")
    
    print(f"Action: {etype} {side} at {price}")
    print(f"Best Bid: {ob.get_best_bid()} | Best Ask: {ob.get_best_ask()}")
    print(f"Spread: {spread} | Mid-Price: {mid}")
    print("-" * 30)

#Try the slippage functions
testbuy=ob.buy_order(100)
print(testbuy)

testsell=ob.sell_order(111)
print(testsell)