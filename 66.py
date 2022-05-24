class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        # 逆序遍历
        # 如果不是9, 正常 +1 123-> 124
        # 如果是9, 找到第一个不是9的位置,然后置0 1299 -> 1399 -> 1300
        # 全是9, 置为0, 然后加一位 99 -> 00 -> 1 00
        n = len(digits)
        for i in range(n - 1, -1, -1):
            if digits[i] != 9:
                digits[i] += 1
                for j in range(i + 1, n):
                    digits[j] = 0
                return digits
        return [1] + [0] * n
