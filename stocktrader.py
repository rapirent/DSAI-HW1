
ACTION_LIST = {'BUY': 1, 'IDLE': 0, 'SELL': -1}

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
            if trend > 2:
                # up
                # BUY
                action = ACTION_LIST['BUY']
            if trend < 2:
                # large down
                action = ACTION_LIST['SELL']
            else:
                action = ACTION_LIST['IDLE']
        elif self.current_state == 1:
            if trend < 1:
                # large down
                # SOLD!
                action = ACTION_LIST['SELL']
            else:
                action = ACTION_LIST['IDLE']
        else:
            if trend > 2:
                # up
                action = ACTION_LIST['BUY']
            else:
                # maintain the
                action = ACTION_LIST['IDLE']
        return action

    def reaction(self, today, action):
        today_price = self._DATA['open'][today]
        if self.current_state == 0:
            if action == ACTION_LIST['BUY']:
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
        elif self.current_state == 1:
            if action == ACTION_LIST['SELL']:
                print('----\nTAKE ACTION {} in day{} \n----'.format(action, today))
                self.money += today_price
                self.current_state = 0
            elif action == ACTION_LIST['IDLE']:
                pass
            else:
                raise StockValueError(ACTION_LIST['BUY'])
        else:
            if action == ACTION_LIST['BUY']:
                print('----\nTAKE ACTION {} in day{} \n----'.format(action, today))
                self.money -= today_price
                self.current_state = 0
            elif action == ACTION_LIST['IDLE']:
                pass
            else:
                raise StockValueError(ACTION_LIST['SELL'])

    def get_money(self):
        return self.money

    def get_today_price(self, today):
        return self._DATA['open'][today]

