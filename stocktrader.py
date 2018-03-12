
ACTION_LIST = {'HOLD': 1, 'IDLE': 0, 'SHORT': -1}

class StockValueError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr('WRONG AT ' + self.value)


class StockTrader():

    def __init__(self, init_state = 0, data=None):
        self.current_state = init_state
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
            if trend > 3:
                # up
                # BUY
                action = ACTION_LIST['HOLD']
            if trend < 2:
                # large down
                action = ACTION_LIST['SHORT']
            else:
                action = ACTION_LIST['IDLE']
        elif self.current_state == 1:
            if trend == 0:
                # large down
                # SOLD!
                action = ACTION_LIST['IDLE']
            else:
                action = ACTION_LIST['HOLD']
        else:
            if trend > 2:
                # up
                action = ACTION_LIST['IDLE']
            else:
                # maintain the SHORT
                action = ACTION_LIST['SHORT']
        return action

    def reaction(self, today, action):
        today_price = self._DATA['open'][today]
        if self.current_state == 0:
            if action == ACTION_LIST['HOLD']:
                # Buy
                print('----\nTAKE ACTION {} in day{} \n----'.format(action, today))
                self.money -= today_price
                self.current_state = 1
            elif action == ACTION_LIST['IDLE']:
                pass
            else:
                # short
                print('----\nTAKE ACTION {} in day{} \n----'.format(action, today))
                self.money += today_price
                self.current_state = -1
        if self.current_state == 1:
            if action == ACTION_LIST['HOLD']:
                pass
            elif action == ACTION_LIST['IDLE']:
                print('----\nTAKE ACTION {} in day{} \n----'.format(action, today))
                self.money += today_price
                self.current_state = 0
            else:
                raise StockValueError(ACTION_LIST['SHORT'])
        if self.current_state == -1:
            if action == ACTION_LIST['HOLD']:
                raise StockValueError(ACTION_LIST['HOLD'])
            elif action == ACTION_LIST['IDLE']:
                print('----\nTAKE ACTION {} in day{} \n----'.format(action, today))
                self.money -= today_price
                self.current_state = 0
            else:
                pass

    def get_money(self):
        return self.money

    def get_today_price(self, today):
        return self._DATA['open'][today]

