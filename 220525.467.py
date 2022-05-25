# class Solution:
#     def findSubstringInWraproundString(self, p: str) -> int:
#         # abcda
#         # a
#         # b ab
#         # c bc abc
#         # d cd bcd abcd
#         n = len(p)
#         dp = defaultdict(int)
#         tmp = 0
#         for i in range(n):
#             if i > 0 and ((ord(p[i]) - ord(p[i - 1]) == 1) or (ord(p[i]) - ord(p[i - 1]) == -25)):
#                 tmp += 1
#             else:
#                 tmp = 1
#             dp[p[i]] = max(dp[p[i]], tmp)


#         return sum(dp.values())


class Solution:
    def findSubstringInWraproundString(self, p: str) -> int:
        dp = defaultdict(int)
        k = 0
        # enumerate !!
        for i, ch in enumerate(p):
            # -25 % 26 = 1
            if i > 0 and (ord(ch) - ord(p[i - 1])) % 26 == 1:
                k += 1
            else:
                k = 1
            dp[ch] = max(dp[ch], k)
        return sum(dp.values())
