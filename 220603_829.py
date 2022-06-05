class Solution:
    def consecutiveNumbersSum(self, n: int) -> int:
        ans = 0
        i = 1
        while i <= n:
            if n % i == 0:
                ans += 1
            n -= i
            i += 1
        return ans
                
# 9 % 1 T
# 8 % 2 T
# 6 % 3 T
# 3 % 4

# 15 % 1 T
# 14 % 2 T
# 12 % 3 T
#  9 % 4 
#  5 % 5 T
