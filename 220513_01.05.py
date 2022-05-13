# class Solution:
#     def oneEditAway(self, first: str, second: str) -> bool:
#         # insert +1 delete -1 replace == 0
#         fir = len(first)
#         sec = len(second)
#         # replace
#         if fir - sec == 0:
#             tag = 0
#             for i in range(fir):
#                 if first[i] != second[i]:
#                     tag += 1
#                 if tag > 1:
#                     return False
#         elif fir - sec == -1:
#             # insert
#             l, r = 0, fir
#             while l < fir and first[l] == second[l]:
#                 l += 1
#             if l == fir:
#                 return True
#             while r >= 0 and first[r - 1] == second[r]:
#                 r -= 1
#             if r == -1:
#                 return True
#             if l != r:
#                 return False
#         elif fir - sec == 1:
#             # delete
#             l, r = 0, sec
#             while l < sec and first[l] == second[l]:
#                 l += 1
#             if l == sec:
#                 return True
#             while r >= 0 and first[r] == second[r - 1]:
#                 r -= 1
#             if r == -1:
#                 return True
#             if l != r:
#                 return False
#         else:
#             return False            
#             # ab    ab      ab
#             # abc   cab     acb
#             # 1 2   0 1     1 1
#             # abc   abc     abc
#             # ab    bc      ac
#             # 1 
#         return True

class Solution:
    def oneEditAway(self, first: str, second: str) -> bool:
        fir, sec = len(first), len(second)
        if abs(fir - sec) > 1:
            return False
        # 调用自身省下分开判断长短，然后双指针 fir >= sec
        if fir < sec:
            return self.oneEditAway(second, first)
        i, j, ans = 0, 0, 0
        while i < fir and j < sec:
            if first[i] == second[j]:
                i += 1
                j += 1
            else:
                #fir >= sec
                if fir == sec:
                    i += 1
                    j += 1
                    ans += 1
                else:
                    # fir > sec
                    i += 1
                    ans += 1
        
        return True if ans < 2 else False
        
# # 卧槽 还可以这样递归。。。
# class Solution:
#     def oneEditAway(self, first: str, second: str) -> bool:
#         m, n = len(first), len(second)
#         if m < n:
#             return self.oneEditAway(second, first)
#         if m - n > 1:
#             return False
#         for i, (x, y) in enumerate(zip(first, second)):
#             if x != y:
#                 return first[i + 1:] == second[i + 1:] if m == n else first[i + 1:] == second[i:]
#         return True


