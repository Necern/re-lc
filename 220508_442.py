# class Solution:
#     def findDuplicates(self, nums: List[int]) -> List[int]:
#         ans = []
#         # dct = collections.Counter(nums)
#         # for k, v in dct.items():
#         hashmap = set()
#         for n in nums:
#             if not n in hashmap:
#                 hashmap.add(n)
#             else:
#                 ans.append(n)

#         return ans

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        # O(1) space complexity
        # [1, n], | * -1
        # swap(n, n - 1)

        ans = []
        for n in nums:
            if nums[abs(n) - 1] < 0:
                ans.append(abs(n))
            nums[abs(n) - 1] *= -1

        return ans
    
