# class Solution:
#     def strStr(self, haystack: str, needle: str) -> int:
#         for i in range(len(haystack)):
#             if haystack[i] == needle[0]:
#                 ans, j = i, 0
#                 while j + i < len(haystack) and j < len(needle) and needle[j] == haystack[j + i]:
#                     j += 1
#                 if j == len(needle):
#                     return ans
        
#         return -1

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # 0 1 2 3 4 index
        # a b a b c needle
        # 0 0 1 2 0 nxt
        n, m = len(haystack), len(needle)
        i, j = 1, 0
        nxt = [0] * m
        for i in range(1, m):
            # 2. 不同且不为初始状态，j是当前匹配失败位置，所以要j-1,通过nxt跳到上一个位置
            while j > 0 and needle[i] != needle[j]:
                j = nxt[j - 1]
            # 1. 相同的话j往下走
            if needle[i] == needle[j]:
                j += 1
            # 3. 因为nxt数组匹配成功部分是+1递增的，失败部分是返回上一个位置。直接赋值j的状态 (1 or 2)
            nxt[i] = j

        j = 0
        for i in range(n):
            while j > 0 and haystack[i] != needle[j]: 
                j = nxt[j - 1]
            if haystack[i] == needle[j]:
                j += 1
            # 全部匹配后，haystack长度 - 匹配串长度 + 1
            if j == m:
                return i - m + 1

        return -1
