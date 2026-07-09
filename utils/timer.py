import time
from collections import OrderedDict


class _TimerContext:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.timer.record(self.name, elapsed)
        print(f'[{self.name}] 耗时: {elapsed:.2f}s')


class Timer:
    def __init__(self):
        self._times = OrderedDict()

    def __call__(self, name):
        return _TimerContext(self, name)

    def record(self, name, elapsed):
        self._times.setdefault(name, []).append(elapsed)

    def summary(self):
        total = 0
        print('\n===== 耗时统计 =====')
        for name, times in self._times.items():
            t = sum(times)
            total += t
            print(f'  {name:<20} {t:.2f}s  ({len(times)} 次调用)')
        print(f'  {"总计":<20} {total:.2f}s')
        print('====================\n')
        return total
