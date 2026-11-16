class MinMaxStats:
    def __init__(self, min_value, max_value):
        if min_value == max_value:
            raise ValueError('min max cannot have the same value')
        self.min = min_value
        self.max = max_value

    def update(self, value):
        if value < self.min:
            self.min = value
        elif value > self.max:
            self.max = value

    def normalize(self, value):
        return (value - self.min) / (self.max - self.min)
