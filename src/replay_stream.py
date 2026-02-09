class ReplayStream:
    def __init__(self, data):
        self.data = data
        self.ptr = 0

    def next(self):
        frame = self.data[self.ptr]
        self.ptr = (self.ptr + 1) % len(self.data)
        return frame
