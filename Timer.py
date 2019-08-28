from timeit import default_timer

class Timer:
    def __init__(self, text: str = None):
        self.text = text

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, *args):
        self.end = default_timer()
        self.interval = self.end - self.start
        if self.text is not None:
            print(self.text.format(self.interval))
