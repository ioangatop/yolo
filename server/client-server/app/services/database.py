import queue

class AsyncDatabase(queue.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

