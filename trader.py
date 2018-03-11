
ACTION_LIST = {'HOLD': 1, 'IDLE': 0, 'SHORT': -1}

class StockValueError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr('WRONG' + ACTION_LIST[self.value])


class Trader():

    def __init__(self, init_state = 0, data=None):
        self.current_state = 0
        self._state = [-1, 0, 1]
        self.money = 0
        self._DATA = data

    def load_data(self, data):
        self._DATA = data

    def data_len(self):
        return len(self._DATA['open'])

    def reset(self):
        self.record = 0
        self.current_state = 0

    def set_state(self, state):
        self.current_state = state

    def predict_action(self, trend, i):
        if self.current_state == 0:
            if trend == 4 or trend == 3:
                # up
                # BUY
                self.action = ACTION_LIST['HOLD']
            if trend == 0 or trend == 1:
                # large down
                self.action = ACTION_LIST['SHORT']
            else:
                self.action = ACTION_LIST['IDLE']
        elif self.current_state == 1:
            if trend == 0 or trend == 1:
                # large down
                # SOLD!
                self.action = ACTION_LIST['IDLE']
            else:
                self.action = ACTION_LIST['HOLD']
        else:
            if trend == 4 or trend == 3:
                # up
                self.action = ACTION_LIST['IDLE']
            else:
                # maintain the SHORT
                self.action = ACTION_LIST['SHORT']
        return self.action

    def reaction(self, today):
        today_price = self._DATA['open'][today]
        if self.current_state == 0:
            if self.action == ACTION_LIST['HOLD']:
                # Buy
                self.money -= today_price
                self.current_state = 1
            elif self.action == ACTION_LIST['IDLE']:
                pass
            else:
                # short
                self.money += today_price
                self.current_state = -1
        if self.current_state == 1:
            if self.action == ACTION_LIST['HOLD']:
                pass
            elif self.action == ACTION_LIST['IDLE']:
                self.money += today_price
                self.current_state = 0
            else:
                raise StockValueError(ACTION_LIST['SHORT'])
        if self.current_state == -1:
            if self.action == ACTION_LIST['HOLD']:
                raise StockValueError(ACTION_LIST['HOLD'])
            elif self.action == ACTION_LIST['IDLE']:
                self.money -= today_price
                self.current_state = 0
            else:
                pass

    def get_money(self):
        return self.money
