import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def get_current(self):
        return self.get_time_hhmmss(self.start)

    def get_time_hhmmss(self, start=None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None:
            return time.strftime("%Y/%m/%d %H:%M:%S")
        end = time.time()
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str
