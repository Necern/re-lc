# class Solution:
#     def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
#         if desiredTotal < maxChoosableInteger:
#             return True
#         if maxChoosableInteger * (maxChoosableInteger + 1) // 2 < desiredTotal:
#             return False
#         @cache
#         def foo(n, total):
#             for i in range(maxChoosableInteger):
#                 if ((1 << i) & n) == 0:
#                     if total + i + 1 >= desiredTotal or not foo((1 << i) | n,  total + i + 1):
#                         return True
#             return False

#         return foo(0, 0)

# 核心思路：缓存，把复杂度从n!降低到2^n*n（空间换时间，2^n是状态数，n是每个状态的计算复杂度）

# 二进制状态压缩的前提下，进一步优化的方向：
# 1. 递归转递推 => 从所有数都已经被使用开始(1<<n)-1 一直计算到 所有数都还没有被使用0
# 按照二进制从大到小的顺序遍历，我们遍历到任何一个状态的时候，它所有的可达状态都已经被计算过了
# 这样就可以完全不用递归，直接用2^n * n的循环解决

# 2. 优化剪枝，减少不必要的计算
# 一个状态所能到达的所有状态，只要有一个是必败态，那么当前就是必胜态，否则就是必败态
# 我们能不能优先去找更可能到达必败态的状态，不再计算一些不必要的状态
#     比如说 先选择 一次就能超过上限的数（改成从大到小遍历）

# 3. 根据 maxChoosableInteger desiredTotal 选择合适的算法

# by 黑聚
