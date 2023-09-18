import time

UL_MAX = 4294967295
MILLI_SECONDS = False
MICRO_SECONDS = True


class TimerTicks:
    def __init__(self):
        self.timeBench = 0
        self.period = 0
        self.timeDiff = 0
        self.us = False

    def begin(self, timeout, micros=False):
        self.us = micros
        self.update(timeout)

    def update(self, timeout):
        self.period = timeout
        self.start()

    def start(self):
        if self.us:
            self.timeBench = int(time.time() * 1000000)  # Microseconds
        else:
            self.timeBench = int(time.time() * 1000)  # Milliseconds

    def reset(self):
        self.timeBench += self.period

    def tick(self, reset=True):
        current_time = 0

        if self.us:
            current_time = int(time.time() * 1000000)  # Microseconds
        else:
            current_time = int(time.time() * 1000)  # Milliseconds

        if current_time < self.timeBench:
            self.timeDiff = (UL_MAX - self.timeBench) + current_time
        else:
            self.timeDiff = current_time - self.timeBench

        if self.timeDiff >= self.period:
            if reset:
                self.reset()
            return True

        return False
