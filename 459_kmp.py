# class Solution:
#     def repeatedSubstringPattern(self, s: str) -> bool:
#         n = len(s)
#         l = n >> 1
#         for i in range(1, l + 1):
#             if n % i == 0:
#                 if all(s[j] == s[j - i] for j in range(i, n)):
#                     return True
#         return False


class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        # 0 1 2 3 4 5 index
        # a b a b a b needle
        # 0 0 1 2 3 4 nxt
        n = len(s)
        i, j = 1, 0
        nxt = [0] * n
        for i in range(1, n):
            # 2. 不同且不为初始状态，j是当前匹配失败位置，所以要j-1,通过nxt跳到上一个位置
            while j > 0 and s[i] != s[j]:
                j = nxt[j - 1]
            # 1. 相同的话j往下走
            if s[i] == s[j]:
                j += 1
            # 3. 因为nxt数组匹配成功部分是+1递增的，失败部分是返回上一个位置。直接赋值j的状态 (1 or 2)
            nxt[i] = j
        
        # nxt[-1] == 0, 意味着不构成重复
        # n - nxt[-1], 重复段的长度，再判断能否整除
        # 6 - 4 = 2 -> ab
        # a b a b a b
        # 0 0 1 2 3 4
        return True if nxt[-1] != 0 and n % (n - (nxt[-1])) == 0 else False

