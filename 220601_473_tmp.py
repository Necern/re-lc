# 不太懂

class Solution:
    def makesquare(self, matchsticks: List[int]) -> bool:
        def foo(index):
            if index == len(matchsticks):
                return True
            for i in range(4):
                e[i] += matchsticks[index]
                if e[i] <= total // 4 and foo(index + 1):
                    return True
                e[i] -= matchsticks[index]
            return False

        total = sum(matchsticks)
        if total % 4:
            return False
        matchsticks.sort(reverse=True)
        e = [0] * 4
        return foo(0)


