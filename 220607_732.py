class MyCalendarThree:

    def __init__(self):
        self.dct = dict()

    def book(self, start: int, end: int) -> int:
        self.dct[start] = self.dct.setdefault(start, 0) + 1
        self.dct[end] = self.dct.setdefault(end, 0) - 1
        lst = sorted(self.dct.items(), key=lambda x:x[0])
        # sorted(dct): return list
        ans = tmp = 0
        for _, v in lst:
            tmp += v
            ans = max(ans, tmp)

        return ans

# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)


