import time
from pprint import pprint

__all__ = ['TimeCost']


class TimeCost(object):

    def __init__(self, close=True):
        self.close = close
        self.tag_list = []
        self.t0 = time.time()
        self.add_tag("t0", self.t0)

    def clear(self):
        self.tag_list.clear()
        self.t0 = time.time()
        self.add_tag("t0", self.t0)

    def add_tag(self, tag_name, tag_time=None):
        if self.close:
            return
        self.tag_list.append(
            {
                "tag_name": tag_name,
                "tag_time": time.time(),
            }
        )

    def print_tag(self):
        if self.close:
            return
        for t in self.tag_list:
            pprint(t)

    def print_time_cost(self):
        if self.close:
            return
        last_t = self.t0
        for i, tag in enumerate(self.tag_list):
            tn = tag["tag_name"]
            tt = tag["tag_time"]
            print("time cost {:0.4f}s, \t tag is {}".format(tt - last_t, tn))
            last_t = tt
        print("-" * 100)
        self.clear()


"""
TEST FUNCTION
"""


def test():
    tc = TimeCost(close=False)
    tc.add_tag('t1')
    for i in range(9999999):
        n = i * i
    tc.add_tag('tttttt2')

    for i in range(99999999):
        n = i * i
    tc.add_tag('t3')

    tc.print_tag()
    tc.print_time_cost()


if __name__ == "__main__":
    test()
