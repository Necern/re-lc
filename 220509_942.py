class Solution:
    def diStringMatch(self, s: str) -> List[int]:
        l, r = 0, len(s)
        ans = []

        for i in s:
            if i == 'I':
                ans.append(l)
                l += 1
            elif i == 'D':
                ans.append(r)
                r -= 1

        ans.append(l)
        return ans
