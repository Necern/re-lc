class Solution:
    def repeatedNTimes(self, nums: List[int]) -> int:
        n = len(nums) // 2
        # dct = {}
        # for x in nums:
        #     if x not in dct:
        #         dct[x] = 1
        #     else:
        #         dct[x] += 1
        #     if dct[x] == n:
        #         return x
        
        # nums 包含 n + 1 个 不同的 元素
        s = set()
        for x in nums:
            if x in s:
                return x
            s.add(x)
