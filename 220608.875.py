class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # mid 去除每一堆 piles[i] 计算时间 总和与h对比
        # sum > h: l = mid + 1 总时长大于h, 底数小了 左加一
        # sum <= h: r = mid 总时长小于h, 吃快了 (要找刚好等于h的位置)
        def check(mid, h):
            ans = 0
            for n in piles:
                ans += math.ceil(n / mid)
            return ans > h

        l, r = 1, 10 ** 9
        while l < r:
            mid = l + ((r - l) >> 1)
            if check(mid, h):
                l = mid + 1
            else:
                r = mid
        
        return r
