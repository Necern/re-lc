import imp
from typing import TypeVar
import typing
from fractions import Fraction
import collections
import math
from collections import defaultdict
from collections import deque
from re import T
import re
# from sortedcontainers import SortedDict


# http://ddia.vonng.com/#/ch3  

# 467. 环绕字符串中唯一的子字符串
# 把字符串 s 看作是 “abcdefghijklmnopqrstuvwxyz” 的无限环绕字符串，所以 s 看起来是这样的：
# "...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd...." . 
# 现在给定另一个字符串 p 。返回 s 中 唯一 的 p 的 非空子串 的数量 。 
# 示例 1:
# 输入: p = "a"
# 输出: 1
# 解释: 字符串 s 中只有一个"a"子字符。
# 示例 2:
# 输入: p = "cac"
# 输出: 2
# 解释: 字符串 s 中的字符串“cac”只有两个子串“a”、“c”。.
# 示例 3:
# 输入: p = "zab"
# 输出: 6
# 解释: 在字符串 s 中有六个子串“z”、“a”、“b”、“za”、“ab”、“zab”。
 
# 1 <= p.length <= 105
# p 由小写英文字母构成
# (0, 25), 1 - 0 = 1, 0 - 25 = -25

# abcdz
# a
# b ab 
# c bc abc
# d cd bcd abcd

# for i in n:
# if i - [i - 1] == 1:
# a += 1 
# ans += sum(range(a + 1))


# class Solution:
#     def findSubstringInWraproundString(self, p: str) -> int:
#         s = set()
#         s.add(p[0])
#         n = len(p)
#         tmp = 0
#         ans = 0
#         for i in range(n):
#             # -25 % 26 = 1
#             if i > 0 and (ord(i) - ord(i - 1)) % 26 == 1:
#                 tmp += 1
#             else:
#                 ans += sum(range(tmp + 1))

#         # k = 1
#         # for i in range(1, n):
#         #     for j in range(i + 1, n):
#         #         if (ord(j) - ord(j - 1) == 1 or ord(j) - ord(j - 1) == -25):
#         #             k += 1
#         #             if p[i: j] not in s:
#         #                 s.add(p[i: j])
#         #         else:
        
#         return ans

# class Solution:
#     def findSubstringInWraproundString(self, p: str) -> int:
#         dp = defaultdict(int)
#         k = 0
#         for i, ch in enumerate(p):
#             if i > 0 and (ord(ch) - ord(p[i - 1])) % 26 == 1:  # 字符之差为 1 或 -25
#                 k += 1
#             else:
#                 k = 1
#             dp[ch] = max(dp[ch], k)
#         return sum(dp.values())


# 354. 俄罗斯套娃信封问题
# 给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
# 当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
# 请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
# 注意：不允许旋转信封。
# 示例 1：
# 输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
# 输出：3
# 解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
# 示例 2：
# 输入：envelopes = [[1,1],[1,1],[1,1]]
# 输出：1
# 提示：
# 1 <= envelopes.length <= 105
# envelopes[i].length == 2
# 1 <= wi, hi <= 105



# # f(231, 15)
# def foo(x, y):
#     if y == 0:
#         return x 
#     elif y > 0:
#         foo(y, x % y)

# return foo(231, 15)

# #   x   y  x % y   f(231, 15)
# # 231  15      6   f(15, 6)
# #  15   6      3   f(6, 3)
# #   6   3      0   f(3, 0)
# #   3   0   


# 699. 掉落的方块
# 在无限长的数轴（即 x 轴）上，我们根据给定的顺序放置对应的正方形方块。
# 第 i 个掉落的方块（positions[i] = (left, side_length)）是正方形，
# 其中 left 表示该方块最左边的点位置(positions[i][0])，side_length 表示该方块的边长(positions[i][1])。
# 每个方块的底部边缘平行于数轴（即 x 轴），并且从一个比目前所有的落地方块更高的高度掉落而下。
# 在上一个方块结束掉落，并保持静止后，才开始掉落新方块。
# 方块的底边具有非常大的粘性，并将保持固定在它们所接触的任何长度表面上（无论是数轴还是其他方块）。
# 邻接掉落的边不会过早地粘合在一起，因为只有底边才具有粘性。
# 返回一个堆叠高度列表 ans 。
# 每一个堆叠高度 ans[i] 表示在通过 positions[0], positions[1], ..., positions[i] 表示的方块掉落结束后，
# 目前所有已经落稳的方块堆叠的最高高度。
# 示例 1:
# 输入: [[1, 2], [2, 3], [6, 1]]
# 输出: [2, 5, 5]
# 解释:
# 第一个方块 positions[0] = [1, 2] 掉落：
# _aa
# _aa
# -------
# 方块最大高度为 2 。
# 第二个方块 positions[1] = [2, 3] 掉落：
# __aaa
# __aaa
# __aaa
# _aa__
# _aa__
# --------------
# 方块最大高度为5。
# 大的方块保持在较小的方块的顶部，不论它的重心在哪里，因为方块的底部边缘有非常大的粘性。
# 第三个方块 positions[1] = [6, 1] 掉落：
# __aaa
# __aaa
# __aaa
# _aa
# _aa___a
# -------------- 
# 方块最大高度为5。
# 因此，我们返回结果[2, 5, 5]。

# 示例 2:
# 输入: [[100, 100], [200, 100]]
# 输出: [100, 100]
# 解释: 相邻的方块不会过早地卡住，只有它们的底部边缘才能粘在表面上。
# 注意:
# 1 <= positions.length <= 1000.
# 1 <= positions[i][0] <= 10^8.
# 1 <= positions[i][1] <= 10^6.

# class Solution:
#     def fallingSquares(self, positions: List[List[int]]) -> List[int]:
#         n = len(positions)
#         heights = [0] * n
#         for i, (left1, side1) in enumerate(positions):
#             right1 = left1 + side1 - 1
#             heights[i] = side1
#             for j in range(i):
#                 left2, right2 = positions[j][0], positions[j][0] + positions[j][1] - 1
#                 if right1 >= left2 and right2 >= left1:
#                     heights[i] = max(heights[i], heights[j] + side1)
#         for i in range(1, n):
#             heights[i] = max(heights[i], heights[i - 1])
#         return heights

# class Solution:
#     def fallingSquares(self, positions: List[List[int]]) -> List[int]:
#         ans = []
#         x, y = 0, 0
#         for i, j in positions:
#             # x < i => 在外面 a__a
#             if x < i:
#                 x = i + j - 1
#                 y = max(y, j)
#                 ans.append(y)
#             else:
#                 x = max(x, i + j - 1)
#                 y += j
#                 ans.append(y)
#         return ans

# [[1, 2], [2, 3], [6, 1]]        
# # x y i j
# # 0 0 1 2
# # 2 2 
# #     2 3 
# # 4 5
# #     6 1
# # 6 5
# # 0


# 面试题 17.11. 单词距离
# 有个内含单词的超大文本文件，给定任意两个不同的单词，找出在这个文件中这两个单词的最短距离(相隔单词数)。如果寻找过程在这个文件中会重复多次，而每次寻找的单词不同，你能对此优化吗?
# 示例：
# 输入：words = ["I","am","a","student","from","a","university","in","a","city"], word1 = "a", word2 = "student"
# 输出：1
# 提示：
# words.length <= 100000

# class Solution:
#     def findClosest(self, words: List[str], word1: str, word2: str) -> int:
#         dct = defaultdict(list)
#         for i, ele in enumerate(words):
#             dct[ele].append(i)
        
#         ans = float('inf')
#         i, j = 0, 0
#         while i < len(dct[word1]) and j < len(dct[word2]):
#             ans = min(ans, abs(dct[word1][i] - dct[word2][j]))
#             if dct[word1][i] < dct[word2][j]:
#                 i += 1
#             else:
#                 j += 1
        
#         return ans


# class Solution:
#     def findClosest(self, words: List[str], word1: str, word2: str) -> int:
#         ans = len(words)
#         index1, index2 = -1, -1
#         for i, word in enumerate(words):
#             if word == word1:
#                 index1 = i
#             elif word == word2:
#                 index2 = i
#             if index1 >= 0 and index2 >= 0:
#                 ans = min(ans, abs(index1 - index2))
#         return ans

# 43. 字符串相乘
# 给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
# 注意：不能使用任何内置的 BigInteger 库或直接将输入转换为整数。
# 示例 1:
# 输入: num1 = "2", num2 = "3"
# 输出: "6"
# 示例 2:
# 输入: num1 = "123", num2 = "456"
# 输出: "56088"

# 1 <= num1.length, num2.length <= 200
# num1 和 num2 只能由数字组成。
# num1 和 num2 都不包含任何前导零，除了数字0本身。

# 123
# 456
# ---
#  18

# class Solution:
#     def multiply(self, num1: str, num2: str) -> int:
#         #


# class Solution:
#     def cutOffTree(self, forest: List[List[int]]) -> int:
#         def bfs(sx: int, sy: int, tx: int, ty: int) -> int:
#             m, n = len(forest), len(forest[0])
#             q = deque([(0, sx, sy)])
#             vis = {(sx, sy)}
#             while q:
#                 d, x, y = q.popleft()
#                 if x == tx and y == ty:
#                     return d
#                 for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
#                     if 0 <= nx < m and 0 <= ny < n and forest[nx][ny] and (nx, ny) not in vis:
#                         vis.add((nx, ny))
#                         q.append((d + 1, nx, ny))
#             return -1

#         trees = sorted((h, i, j) for i, row in enumerate(forest) for j, h in enumerate(row) if h > 1)
#         ans = preI = preJ = 0
#         for _, i, j in trees:
#             d = bfs(preI, preJ, i, j)
#             if d < 0:
#                 return -1
#             ans += d
#             preI, preJ = i, j
#         return ans


# class Solution:
#     def cutOffTree(self, forest: List[List[int]]) -> int:
#         def bfs(sx: int, sy: int, tx: int, ty: int) -> int:
#             m, n = len(forest), len(forest[0])
#             q = deque([(0, sx, sy)])
#             vis = {(sx, sy)}
#             while q:
#                 d, x, y = q.popleft()
#                 if x == tx and y == ty:
#                     return d
#                 for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
#                     if 0 <= nx < m and 0 <= ny < n and forest[nx][ny] and (nx, ny) not in vis:
#                         vis.add((nx, ny))
#                         q.append((d + 1, nx, ny))
#             return -1
#         # forest = [[1,2,3],[0,0,4],[7,6,5]]
#         trees = sorted((h, i, j) for i, row in enumerate(forest) for j, h in enumerate(row) if h > 1)
#         ans = preI = preJ = 0
#         for _, i, j in trees:
#             d = bfs(preI, preJ, i, j)
#             if d < 0:
#                 return -1
#             ans += d
#             preI, preJ = i, j
#         return ans



# class Solution {
#     int N = 50;
#     int[][] g = new int[N][N];
#     int n, m;
#     public int cutOffTree(List<List<Integer>> forest) {
#         n = forest.size(); m = forest.get(0).size();
#         List<int[]> list = new ArrayList<>();
#         for (int i = 0; i < n; i++) {
#             for (int j = 0; j < m; j++) {
#                 g[i][j] = forest.get(i).get(j);
#                 if (g[i][j] > 1) list.add(new int[]{g[i][j], i, j});
#             }
#         }
#         Collections.sort(list, (a,b)->a[0]-b[0]);
#         int x = 0, y = 0, ans = 0;
#         for (int[] ne : list) {
#             int nx = ne[1], ny = ne[2];
#             int d = astar(x, y, nx, ny);
#             if (d == -1) return -1;
#             ans += d;
#             x = nx; y = ny;
#         }
#         return ans;
#     }
#     int[][] dirs = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
#     int getIdx(int x, int y) {
#         return x * m + y;
#     }
#     int f(int X, int Y, int P, int Q) {
#         return Math.abs(X - P) + Math.abs(Y - Q);
#     }
#     int astar(int X, int Y, int P, int Q) {
#         if (X == P && Y == Q) return 0;
#         Map<Integer, Integer> map = new HashMap<>();
#         PriorityQueue<int[]> q = new PriorityQueue<>((a,b)->a[0]-b[0]);
#         q.add(new int[]{f(X, Y, P, Q), X, Y});
#         map.put(getIdx(X, Y), 0);
#         while (!q.isEmpty()) {
#             int[] info = q.poll();
#             int x = info[1], y = info[2], step = map.get(getIdx(x, y));
#             for (int[] di : dirs) {
#                 int nx = x + di[0], ny = y + di[1], nidx = getIdx(nx, ny);
#                 if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
#                 if (g[nx][ny] == 0) continue;
#                 if (nx == P && ny == Q) return step + 1;
#                 if (!map.containsKey(nidx) || map.get(nidx) > step + 1) {
#                     q.add(new int[]{step + 1 + f(nx, ny, P, Q), nx, ny});
#                     map.put(nidx, step + 1);
#                 }
#             }
#         }
#         return -1;
#     }
# }

# DIRS = (0, 1), (1, 0), (0, -1), (-1, 0)
# class Solution:
#     def cutOffTree(self, forest: List[List[int]]) -> int:
#         m, n = len(forest), len(forest[0])
#         trees = sorted((forest[i][j], i, j) for i, j in product(range(m), range(n)) if forest[i][j] > 1)

#         def bfs(start, end):
#             queue = deque([(start, 0)])
#             # forest = [[1,2,3],[0,0,4],[7,6,5]]
#             explored = [list(r) for r in forest]
#             while queue:
#                 cur, cost = queue.popleft()
#                 if cur == end:
#                     return cost
#                 x, y = cur
#                 cost += 1
#                 for dx, dy in DIRS:
#                     if 0 <= (nx := x + dx) < m and 0 <= (ny := y + dy) < n and explored[nx][ny]:
#                             [nx][ny] = 0
#                         queue.append(((nx, ny), cost))
#             return -1
        
#         ans = bfs((0, 0), trees[0][1:])
#         for a, b in pairwise(trees):
#             if (res := bfs(a[1:], b[1:])) == -1:
#                 return -1
#             ans += res
#         return ans

# #
# DIRS = (0, 1), (1, 0), (0, -1), (-1, 0)
# class Foo:
#     def foo(self, forest):
#         m, n = len(forest), len(forest[0])
#         idx_map = {forest[i][j]: (i, j) for i in range(m) for j in range(n) if forest[i][j] > 1}
#         nums = sorted(idx_map.keys())
#         explored = defaultdict(lambda: inf)

#         def h(p, idx):
#             x1, y1 = idx_map[nums[idx]]
#             x2, y2 = p
#             return abs(x2 - x1) + abs(y2 - y1)

#         # f(n) = h(n) + g(n), g(n), idx, point
#         pq = [(h((0, 0), 0), 0, 0, (0, 0))]
#         explored[(0, 0, 0)] = 0
#         mem = set()
#         while pq:
#             _, cost, idx, point = heappop(pq)
#             if idx in mem:
#                 continue
#             x, y = point
#             if forest[x][y] == nums[idx]: # zuobiao
#                 if idx == len(nums) - 1:
#                     return cost
#                 forest[x][y] = 1
#                 mem.add(idx)
#                 idx += 1
#             cost += 1
#             for dx, dy in DIRS:
#                 if 0 <= (nx := x + dx) < m and 0 <= (ny := y + dy) < n and forest[nx][ny] and explored[(nx, ny, idx)] > cost:
#                     explored[(nx, ny, idx)] = cost
#                     heappush(pq, (h((nx, ny), idx) + cost, cost, idx, (nx, ny)))
#         return -1


# 43
# 67
# 989
# 739
# 58
# 48
# 1886
# 54
# 973
# 1630
# 429
# 503
# 556
# 1376
# 49
##
# 438
# 713
# 304
# 910
# 143
# 138
# 2
# 445
# 61
# 173
# 1845
# 860
# 155
# 341
# 1797
# 707
# 380
# 622
# 729

# 875. 爱吃香蕉的珂珂
# 珂珂喜欢吃香蕉。这里有 n 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 h 小时后回来。
# 珂珂可以决定她吃香蕉的速度 k （单位：根/小时）。每个小时，她将会选择一堆香蕉，
# 从中吃掉 k 根。如果这堆香蕉少于 k 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。  
# 珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
# 返回她可以在 h 小时内吃掉所有香蕉的最小速度 k（k 为整数）。
# 示例 1：
# 输入：piles = [3,6,7,11], h = 8
# 输出：4
# 示例 2：
# 输入：piles = [30,11,23,4,20], h = 5
# 输出：30
# 示例 3：
# 输入：piles = [30,11,23,4,20], h = 6
# 输出：23
# 提示：
# 1 <= piles.length <= 104
# piles.length <= h <= 109
# 1 <= piles[i] <= 109
# 通过次数86,354提交次数171,212


#
# class Solution:
#     def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
#         ans = []
#         m, n = len(mat), len(mat[0])
#         for i in range(m + n - 1):
#             if i % 2:
#                 x = 0 if i < n else i - n + 1
#                 y = i if i < n else n - 1
#                 while x < m and y >= 0:
#                     ans.append(mat[x][y])
#                     x += 1
#                     y -= 1
#             else:
#                 x = i if i < m else m - 1
#                 y = 0 if i < m else i - m + 1
#                 while x >= 0 and y < n:
#                     ans.append(mat[x][y])
#                     x -= 1
#                     y += 1
#         return ans


#  1  2  3  4  5
#  6  7  8  9 10
# 11 12 13 14 15
# 16 17 18 19 20
# m = 4, n = 5
# 1, 2 6, 11 7 3, 4 8 12 16, 17 13 9 5, 
# 4 + 5 - 1
# 0 [0, 0]
# 1 [0, 1] [1, 0]
# 2 [2, 0] [1, 1] [0, 2]
# 3 [0, 3] [1, 2] [2, 1] [1, 0]
# m
# 4 [3, 1] [2, 2] [1, 3] [0, 4]
# 5 [1, 4] [2, 3] [3, 2]
# 6 [3, 3] [2, 4]
# 7 [3, 4]

# for i in range(m + n - 1):
#     if i % 2: # 1   
#         while x <= i and y >= 0:
#             x += 1
#             y -= 1
#     else: # 0
#         if i < m:
#             x, y = 0, i
#         else:
#             pass
#         while x >= 0 and y <= i:
#             x -= 1
#             y += 1




# class Solution:
#     def findLUSlength(self, strs: List[str]) -> int:
#         def is_subseq(s: str, t: str) -> bool:
#             pt_s = pt_t = 0
#             while pt_s < len(s) and pt_t < len(t):
#                 if s[pt_s] == t[pt_t]:
#                     pt_s += 1
#                 pt_t += 1
#             return pt_s == len(s)
        
#         ans = -1
#         for i, s in enumerate(strs):
#             check = True
#             for j, t in enumerate(strs):
#                 if i != j and is_subseq(s, t):
#                     check = False
#                     break
#             if check:
#                 ans = max(ans, len(s))
        
#         return ans

# class Solution:
#     def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
#         n = len(arr)
#         arr.sort()

#         best, ans = float('inf'), list()
#         for i in range(n - 1):
#             if (delta := arr[i + 1] - arr[i]) < best:
#                 best = delta
#                 ans = [[arr[i], arr[i + 1]]]
#             elif delta == best:
#                 ans.append([arr[i], arr[i + 1]])
        
#         return ans


# class MyCalendar:
#     def __init__(self):
#         self.booked = SortedDict()

#     def book(self, start: int, end: int) -> bool:
#         i = self.booked.bisect_left(end)
#         if i == 0 or self.booked.items()[i - 1][1] <= start:
#             self.booked[start] = end
#             return True
#         return False


# from enum import Enum, auto

# class ExprStatus(Enum):
#     VALUE = auto()  # 初始状态
#     NONE  = auto()  # 表达式类型未知
#     LET   = auto()  # let 表达式
#     LET1  = auto()  # let 表达式已经解析了 vi 变量
#     LET2  = auto()  # let 表达式已经解析了最后一个表达式 expr
#     ADD   = auto()  # add 表达式
#     ADD1  = auto()  # add 表达式已经解析了 e1 表达式
#     ADD2  = auto()  # add 表达式已经解析了 e2 表达式
#     MULT  = auto()  # mult 表达式
#     MULT1 = auto()  # mult 表达式已经解析了 e1 表达式
#     MULT2 = auto()  # mult 表达式已经解析了 e2 表达式
#     DONE  = auto()  # 解析完成

# class Expr:
#     __slots__ = 'status', 'var', 'value', 'e1', 'e2'

#     def __init__(self, status):
#         self.status = status
#         self.var = ''  # let 的变量 vi
#         self.value = 0  # VALUE 状态的数值，或者 LET2 状态最后一个表达式的数值
#         self.e1 = self.e2 = 0  # add 或 mult 表达式的两个表达式 e1 和 e2 的数值

# class Solution:
#     def evaluate(self, expression: str) -> int:
#         scope = defaultdict(list)

#         def calculateToken(token: str) -> int:
#             return scope[token][-1] if token[0].islower() else int(token)

#         vars = []
#         s = []
#         cur = Expr(ExprStatus.VALUE)
#         i, n = 0, len(expression)
#         while i < n:
#             if expression[i] == ' ':
#                 i += 1  # 去掉空格
#                 continue
#             if expression[i] == '(':
#                 i += 1  # 去掉左括号
#                 s.append(cur)
#                 cur = Expr(ExprStatus.NONE)
#                 continue
#             if expression[i] == ')':  # 本质上是把表达式转成一个 token
#                 i += 1  # 去掉右括号
#                 if cur.status is ExprStatus.LET2:
#                     token = str(cur.value)
#                     for var in vars[-1]:
#                         scope[var].pop()  # 清除作用域
#                     vars.pop()
#                 elif cur.status is ExprStatus.ADD2:
#                     token = str(cur.e1 + cur.e2)
#                 else:
#                     token = str(cur.e1 * cur.e2)
#                 cur = s.pop()  # 获取上层状态
#             else:l
#                 i0 = i
#                 while i < n and expression[i] != ' ' and expression[i] != ')':
#                     i += 1
#                 token = expression[i0:i]

#             if cur.status is ExprStatus.VALUE:
#                 cur.value = int(token)
#                 cur.status = ExprStatus.DONE
#             elif cur.status is ExprStatus.NONE:
#                 if token == "let":
#                     cur.status = ExprStatus.LET
#                     vars.append([])  # 记录该层作用域的所有变量, 方便后续的清除
#                 elif token == "add":
#                     cur.status = ExprStatus.ADD
#                 elif token == "mult":
#                     cur.status = ExprStatus.MULT
#             elif cur.status is ExprStatus.LET:
#                 if expression[i] == ')':  # let 表达式的最后一个 expr 表达式
#                     cur.value = calculateToken(token)
#                     cur.status = ExprStatus.LET2
#                 else:
#                     cur.var = token
#                     vars[-1].append(token)  # 记录该层作用域的所有变量, 方便后续的清除
#                     cur.status = ExprStatus.LET1
#             elif cur.status is ExprStatus.LET1:
#                 scope[cur.var].append(calculateToken(token))
#                 cur.status = ExprStatus.LET
#             elif cur.status is ExprStatus.ADD:
#                 cur.e1 = calculateToken(token)
#                 cur.status = ExprStatus.ADD1
#             elif cur.status is ExprStatus.ADD1:
#                 cur.e2 = calculateToken(token)
#                 cur.status = ExprStatus.ADD2
#             elif cur.status is ExprStatus.MULT:
#                 cur.e1 = calculateToken(token)
#                 cur.status = ExprStatus.MULT1
#             elif cur.status is ExprStatus.MULT1:
#                 cur.e2 = calculateToken(token)
#                 cur.status = ExprStatus.MULT2
#         return cur.value



# class Solution:
#     def replaceWords(self, dictionary: List[str], sentence: str) -> str:
#         dictionarySet = set(dictionary)
#         words = sentence.split(' ')
#         for i, word in enumerate(words):
#             for j in range(1, len(words) + 1):
#                 if word[:j] in dictionarySet:
#                     words[i] = word[:j]
#                     break
#         return ' '.join(words)


# class Solution:
#     def replaceWords(self, dictionary: List[str], sentence: str) -> str:
#         trie = {}
#         for word in dictionary:
#             cur = trie
#             for c in word:
#                 if c not in cur:
#                     cur[c] = {}
#                 cur = cur[c]
#             cur['#'] = {}

#         words = sentence.split(' ')
#         for i, word in enumerate(words):
#             cur = trie
#             for j, c in enumerate(word):
#                 if '#' in cur:
#                     words[i] = word[:j]
#                     break
#                 if c not in cur:
#                     break
#                 cur = cur[c]
#         return ' '.join(words)


# class Solution:
#     def minCostToMoveChips(self, position: List[int]) -> int:
#         cnt = Counter(p % 2 for p in position)  # 根据模 2 后的余数来统计奇偶个数
#         return min(cnt[0], cnt[1])

# class MagicDictionary:

#     def __init__(self):
#         self.words = list()

#     def buildDict(self, dictionary: List[str]) -> None:
#         self.words = dictionary

#     def search(self, searchWord: str) -> bool:
#         for word in self.words:
#             if len(word) != len(searchWord):
#                 continue
            
#             diff = 0
#             for chx, chy in zip(word, searchWord):
#                 if chx != chy:
#                     diff += 1
#                     if diff > 1:
#                         break
            
#             if diff == 1:
#                 return True
        
#         return False
#
#
#

# 字典树思路
    # tree -> done = boolean() .child = dict()
    # root.child = [a]
    # a.child = [d]
    # d.child = [d]
    # a.d.d.done = True
    # root.child = [b]
    # b.child = [a]
    # a.child = [n]
# class Trie:
#     def __init__(self):
#         self.is_finished = False
#         self.child = dict()

# # 建树: 遍历dic的每个单词的每个字母
# # if ch in root.child # child 是字典
# # root = root.child[ch] 继续往下找
# # else: done = True

# # 查找: 需要当前递归的位置 pos = int()，是否修改过 modified = boolean()，以及当前位置 node = root.child()?
# # 递归退出条件: len(word) == pos # 当前递归位置
# # return 是否修改过 and 树中的done # 不用做 +-1
# # ch = word[pos]
# # if ch in node.child #字典
# # dfs(pos + 1, modified, node.child[ch])
# # else:
# # modified == True: return False
# # modified == False: 
# # 需要遍历 node.child中的每一个chdic 
# # for chdic in node.child:
# # dfs(pos + 1, modified = True, node.child[chdic]) 
# class MagicDictionar:
#     def __init__(self):
#         self.root = Trie()

#     def buildDict(self, dictionary: List[str]) -> None:
#         for word in dictionary:
#             cur = self.root
#             for ch in word:
#                 if ch not in cur.child:
#                     cur.child[ch] = Trie()
#                 cur = cur.child[ch]
#             cur.is_finished = True

#     def search(self, searchWord: str) -> bool:
#         def dfs(node: Trie, pos: int, modified: bool) -> bool:
#             if pos == len(searchWord):
#                 return modified and node.is_finished
            
#             ch = searchWord[pos]
#             if ch in node.child:
#                 if dfs(node.child[ch], pos + 1, modified):
#                     return True
                
#             if not modified:
#                 for cnext in node.child:
#                     if ch != cnext: # 
#                         if dfs(node.child[cnext], pos + 1, True):
#                             return True
            
#             return False
        
#         return dfs(self.root, 0, False)

# n, m list() 记录每一行列的值
# 

# class Solution:
#     def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
#         row = [0 for _ in range(m)]
#         col = [0 for _ in range(n)]

#         for r, c in indices:
#             row[r] += 1
#             col[c] += 1
        
#         a = sum(r % 2 == 1 for r in row)
#         b = len(row) - a
#         c = sum(c % 2 == 1 for c in col)
#         d = len(col) - c
#         return a * d + b * c

# 2, 3, [[0, 1], [1, 1]]
# 1 3 1
# 1 3 1
# r = [1, 1]
# c = [0, 2, 0]
# a = 2, b = 0, c = 0, d = 3

# 2, 2, [[1, 1]]
# 0 1
# 1 2
# r = [0, 1]
# c = [0, 1]

# 735
# 输入：asteroids = [5,10,-5]
# 输出：[5,10]
# 解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。

# 双指针 遍历 
# 同号 继续
# 异号 
# 判断大小 
#    小的指针后移
#    等于同时后移
#    继续判断符号

# ans = []
# i, j = 0, 1
# while 0 <= i and j < len(a):
#     if (a[i] > 0 and a[j] < 0) or (a[i] < 0 and a[j] > 0):
#         if a[i] < a[j]:
#             ans.append(a[j])
#             if i == 0:
#                 i = j
#             else:
#                 i -= 1
#         elif a[i] > a[j]:
#             ans.append(a[i])
#             j += 1
#         else: a[i] == a[j]:
            
#             i -= 1
#             j -= 1

# 735
# 输入：asteroids = [5,10,-5]
# 输出：[5,10]
# 解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。

# 从左到右遍历一次 [1, len(a))
# if not ans: ans.append(a[i]), i++
# 同号 ans.append(a[i])
# 异号 判断大小 
# ans[-1] > a[i]: i++
# ans[-1] == a[i]: ans.pop(), i++ 
# ans[-1] < a[i]: ans.pop(), ans.append(a[i])
# class Solution:
#     def asteroidCollision(self, asteroids: List[int]) -> List[int]:
#         ans, i = list(), 0
#         while i < len(asteroids):
#             if not ans:
#                 ans.append(asteroids[i])
#                 i += 1
#             if (asteroids[i] > 0 and ans[-1] < 0) or (asteroids[i] < 0 and ans[-1] > 0):
#                 if ans[-1] - a[i] > 0:
#                     i += 1
#                 elif ans[-1] - a[i] == 0:
#                     ans.pop()
#                     i += 1
#                 else:
#                     ans.pop()
#                     ans.append(asteroids[i])
#             else:
#                 ans.append(asteroids[i])
#                 i += 1
#         return ans

# class Solution:
#     def asteroidCollision(self, asteroids: List[int]) -> List[int]:
#         st = []
#         for aster in asteroids:
#             alive = True
#             while alive and aster < 0 and st and st[-1] > 0:
#                 alive = st[-1] < -aster
#                 if st[-1] <= -aster:
#                     st.pop()
#             if alive:
#                 st.append(aster)
#         return st

# 745 method 1
# class WordFilter:

#     def __init__(self, words: List[str]):
#         self.d = {}
#         for i, word in enumerate(words):
#             m = len(word)
#             for prefixLength in range(1, m + 1):
#                 for suffixLength in range(1, m + 1):
#                     self.d[word[:prefixLength] + '#' + word[-suffixLength:]] = i


#     def f(self, pref: str, suff: str) -> int:
#         return self.d.get(pref + '#' + suff, -1)

# 745 method 2
# class WordFilter:
#     def __init__(self, words: List[str]):
#         self.trie = {}
#         self.weightKey = ('#', '#')
#         for i, word in enumerate(words):
#             cur = self.trie
#             m = len(word)
#             for j in range(m):
#                 tmp = cur
#                 for k in range(j, m):
#                     key = (word[k], '#')
#                     if key not in tmp:
#                         tmp[key] = {}
#                     tmp = tmp[key]
#                     tmp[self.weightKey] = i
#                 tmp = cur
#                 for k in range(j, m):
#                     key = ('#', word[-k - 1])
#                     if key not in tmp:
#                         tmp[key] = {}
#                     tmp = tmp[key]
#                     tmp[self.weightKey] = i
#                 key = (word[j], word[-j - 1])
#                 if key not in cur:
#                     cur[key] = {}
#                 cur = cur[key]
#                 cur[self.weightKey] = i
                
#     def f(self, pref: str, suff: str) -> int:
#         cur = self.trie
#         for key in zip_longest(pref, suff[::-1], fillvalue='#'):
#             if key not in cur: 
#                 return -1
#             cur = cur[key]
#         return cur[self.weightKey]


###########################################################################


# 
# dir_c = (1, -1, 1j, -1j)

# class Solution:
#     def containVirus(self, isInfected: List[List[int]]) -> int:
#         book = [set(), set()]
#         for i, row in enumerate(isInfected):
#             for j, c in enumerate(row):
#                 book[c].add(complex(i, j))
#         blank, virus = book

#         def dfs(u, infect, neib):
#             if u in neib: return 0
#             neib.add(u)
#             wall = 0
#             for v in (u + 1, u - 1, u + 1j, u - 1j):
#                 if v in blank:
#                     infect.add(v)
#                     wall += 1
#                 elif v in virus:
#                     wall += dfs(v, infect, neib)
#             return wall

#         res = 0
#         while virus and blank:
#             seen = set()
#             hi = (set(), set(), 0)
#             book = []
#             for u in virus:
#                 if u in seen: continue
#                 infect, neib = set(), set()
#                 walls = dfs(u, infect, neib)
#                 seen.update(neib)
#                 book.append((infect, neib, walls))
#                 if len(hi[0]) < len(infect):
#                     hi = (infect, neib, walls)
            
#             res += hi[2]
#             virus -= hi[1]
#             for infect, neib, walls in book:
#                 if len(hi[0]) == len(infect): continue
#                 if infect:
#                     virus |= infect
#                     blank -= infect
#                 else:
#                     virus -= neib           

#         return res




#

# MyCalendar.book(10, 20); // returns true
# MyCalendar.book(50, 60); // returns true
# MyCalendar.book(10, 40); // returns true
# MyCalendar.book(5, 15); // returns false
# MyCalendar.book(5, 10); // returns true
# MyCalendar.book(25, 55); // returns true




# class MyCalendarTwo:
#     def __init__(self):
#         self.booked = []
#         self.overlaps = []

#     def book(self, start: int, end: int) -> bool:
#         if any(s < end and start < e for s, e in self.overlaps):
#             return False
#         for s, e in self.booked:
#             if s < end and start < e:
#                 self.overlaps.append((max(s, start), min(e, end)))
#         self.booked.append((start, end))
#         return True


# 5 10 15 20 25 40 50 55 60
# 1  1    -1    -1  1    -1

# class MyCalendarTwo:
#     def __init__(self):
#         self.cnt = SortedDict()

#     def book(self, start: int, end: int) -> bool:
#         self.cnt[start] = self.cnt.get(start, 0) + 1
#         self.cnt[end] = self.cnt.get(end, 0) - 1
#         maxBook = 0
#         for c in self.cnt.values():
#             maxBook += c
#             if maxBook > 2:
#                 self.cnt[start] = self.cnt.get(start, 0) - 1
#                 self.cnt[end] = self.cnt.get(end, 0) + 1
#                 return False
#         return True

##
# class MyCalendarTwo:
#     def __init__(self):
#         self.tree = {}

#     def update(self, start: int, end: int, val: int, l: int, r: int, idx: int) -> None:
#         if r < start or end < l:
#             return
#         if start <= l and r <= end:
#             p = self.tree.get(idx, [0, 0])
#             p[0] += val
#             p[1] += val
#             self.tree[idx] = p
#             return
#         mid = (l + r) // 2
#         self.update(start, end, val, l, mid, 2 * idx)
#         self.update(start, end, val, mid + 1, r, 2 * idx + 1)
#         p = self.tree.get(idx, [0, 0])
#         p[0] = p[1] + max(self.tree.get(2 * idx, (0,))[0], self.tree.get(2 * idx + 1, (0,))[0])
#         self.tree[idx] = p

#     def book(self, start: int, end: int) -> bool:
#         self.update(start, end - 1, 1, 0, 10 ** 9, 1)
#         if self.tree[1][0] > 2:
#             self.update(start, end - 1, -1, 0, 10 ** 9, 1)
#             return False
#         return True

# 
# class Solution:
#     def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
#         m, n = len(grid), len(grid[0])
#         ans = [[0] * n for _ in range(m)]
#         for i, row in enumerate(grid):
#             for j, v in enumerate(row):
#                 index1 = (i * n + j + k) % (m * n)
#                 ans[index1 // n][index1 % n] = v
#         return ans


# 542


# class Solution:
#     def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
#         ans = 0
#         tag = -1
#         intervals.sort(key=lambda x: (x[0], x[1]))
#         a, b = intervals[0]

#         for x, y in intervals:
#             if b < x:
#                 ans += 2
#                 if tag == a:
#                     ans -= 1
#                     tag = -1
#                 a, b = x, y # a b x y #输出
#             elif b == x:
#                 if tag == a:
#                     ans -= 1
#                     tag = -1
#                 ans += 2
#                 tag = b
#                 a, b = x, y #特殊处理
#             elif b < y: 
#                 a, b = x, b # a x b y
#             elif b >= y: 
#                 a, b = x, y # a x y b
        
#         if tag == a:
#             ans -= 1
#         ans += 2
    
#         return ans 
            

# class Solution:
#     def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
#         intervals.sort(key=lambda x: (x[0], -x[1]))
#         ans, n, m = 0, len(intervals), 2
#         vals = [[] for _ in range(n)]
#         for i in range(n - 1, -1, -1):
#             j = intervals[i][0]
#             for k in range(len(vals[i]), m):
#                 ans += 1
#                 for p in range(i - 1, -1, -1):
#                     if intervals[p][1] < j:
#                         break
#                     vals[p].append(j)
#                 j += 1
#         return ans


# a x y b
# a x b y
# a b x y

# x y a b
# x a y b
# x a b y 

# base line = [a, b]
# a <= x < y <= b: [x, y]
# a <= x < b <= y: [x, b]
# a < x <= b < y: [x – 1, b + 1]
# a < b <= x < y: [b - 1, x + 1]

# x <= a < b <= y: [a, b]
# x <= a < y <= b: [a, y]
# x < a <= y < b: [a - 1, y + 1]
# x < y <= a < b: [y - 1, a + 1]


# class CBTInserter:

#     def __init__(self, root: TreeNode):
#         self.root = root
#         self.candidate = deque()

#         q = deque([root])
#         while q:
#             node = q.popleft()
#             if node.left:
#                 q.append(node.left)
#             if node.right:
#                 q.append(node.right)
#             if not (node.left and node.right):
#                 self.candidate.append(node)

#     def insert(self, val: int) -> int:
#         candidate_ = self.candidate

#         child = TreeNode(val)
#         node = candidate_[0]
#         ret = node.val
        
#         if not node.left:
#             node.left = child
#         else:
#             node.right = child
#             candidate_.popleft()
        
#         candidate_.append(child)
#         return ret

#     def get_root(self) -> TreeNode:
#         return self.root


#
#
# skip list 
# MAX_LEVEL = 32
# P_FACTOR = 0.25 

# def random_level() -> int:
#     lv = 1
#     while lv < MAX_LEVEL and random.random() < P_FACTOR:
#         lv += 1
#     return lv

# class SkiplistNode:
#     __slots__ = 'val', 'forward'

#     def __init__(self, val: int, max_level=MAX_LEVEL):
#         self.val = val
#         self.forward = [None] * max_level

# class Skiplist:
#     def __init__(self):
#         self.head = SkiplistNode(-1)
#         self.level = 0

#     def search(self, target: int) -> bool:
#         curr = self.head
#         for i in range(self.level - 1, -1, -1):
#             # 找到第 i 层小于且最接近 target 的元素
#             while curr.forward[i] and curr.forward[i].val < target:
#                 curr = curr.forward[i]
#         curr = curr.forward[0]
#         # 检测当前元素的值是否等于 target
#         return curr is not None and curr.val == target

#     def add(self, num: int) -> None:
#         update = [self.head] * MAX_LEVEL
#         curr = self.head
#         for i in range(self.level - 1, -1, -1):
#             # 找到第 i 层小于且最接近 num 的元素
#             while curr.forward[i] and curr.forward[i].val < num:
#                 curr = curr.forward[i]
#             update[i] = curr
#         lv = random_level()
#         self.level = max(self.level, lv)
#         new_node = SkiplistNode(num, lv)
#         for i in range(lv):
#             # 对第 i 层的状态进行更新，将当前元素的 forward 指向新的节点
#             new_node.forward[i] = update[i].forward[i]
#             update[i].forward[i] = new_node

#     def erase(self, num: int) -> bool:
#         update = [None] * MAX_LEVEL
#         curr = self.head
#         for i in range(self.level - 1, -1, -1):
#             # 找到第 i 层小于且最接近 num 的元素
#             while curr.forward[i] and curr.forward[i].val < num:
#                 curr = curr.forward[i]
#             update[i] = curr
#         curr = curr.forward[0]
#         if curr is None or curr.val != num:  # 值不存在
#             return False
#         for i in range(self.level):
#             if update[i].forward[i] != curr:
#                 break
#             # 对第 i 层的状态进行更新，将 forward 指向被删除节点的下一跳
#             update[i].forward[i] = curr.forward[i]
#         # 更新当前的 level
#         while self.level > 1 and self.head.forward[self.level - 1] is None:
#             self.level -= 1
#         return True

# 0727 592

# x1/y1 + x2/y2 = (x1y2 + x2y1) / y1y2

# class Solution:
#     def fractionAddition(self, expression: str) -> str:
#         denominator, numerator = 0, 1  # 分子，分母
#         i, n = 0, len(expression)
#         while i < n:
#             # 读取分子
#             denominator1, sign = 0, 1
#             if expression[i] == '-' or expression[i] == '+':
#                 if expression[i] == '-':
#                     sign = -1
#                 i += 1
#             while i < n and expression[i].isdigit():
#                 denominator1 = denominator1 * 10 + int(expression[i])
#                 i += 1
#             denominator1 = sign * denominator1
#             i += 1

#             # 读取分母
#             numerator1 = 0
#             while i < n and expression[i].isdigit():
#                 numerator1 = numerator1 * 10 + int(expression[i])
#                 i += 1

#             denominator = denominator * numerator1 + denominator1 * numerator
#             numerator *= numerator1
#         if denominator == 0:
#             return "0/1"
#         g = gcd(abs(denominator), numerator)
#         return f"{denominator // g}/{numerator // g}"

#xx
# ptn = re.compile(r'[+-]?\d+/\d+')
# class Solution:
#     def fractionAddition(self, expr: str) -> str:
#         res = sum(Fraction(f) for f in ptn.findall(expr))
#         return f'{res.numerator}/{res.denominator}'

# a = Solution()
# a.fractionAddition('1/2+2/2+3/2+5/6')        

##
# def arrayRankTransform(self, arr: List[int]) -> List[int]:
#     a = sorted(set(arr))
#     d = {}
#     for i, v in enumerate(a, 1):
#         d[v] = i
#     return [d[i] for i in arr]

# class Solution:
#     def arrayRankTransform(self, arr: List[int]) -> List[int]:
#         ranks = {v: i for i, v in enumerate(sorted(set(arr)), 1)}
#         return [ranks[v] for v in arr]


# 622
# class MyCircularQueue:
#     def __init__(self, k: int):
#         self.front = self.rear = 0
#         self.elements = [0] * (k + 1)

#     def enQueue(self, value: int) -> bool:
#         if self.isFull():
#             return False
#         self.elements[self.rear] = value
#         self.rear = (self.rear + 1) % len(self.elements)
#         return True

#     def deQueue(self) -> bool:
#         if self.isEmpty():
#             return False
#         self.front = (self.front + 1) % len(self.elements)
#         return True

#     def Front(self) -> int:
#         return -1 if self.isEmpty() else self.elements[self.front]

#     def Rear(self) -> int:
#         return -1 if self.isEmpty() else self.elements[(self.rear - 1) % len(self.elements)]

#     def isEmpty(self) -> bool:
#         return self.rear == self.front

#     def isFull(self) -> bool:
#         return (self.rear + 1) % len(self.elements) == self.front

# 1302

# class Solution:
#     def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
#         q = deque([root])
#         while q:
#             ans = 0
#             for _ in range(len(q)):
#                 node = q.popleft()
#                 ans += node.val
#                 if node.left:
#                     q.append(node.left)
#                 if node.right:
#                     q.append(node.right)
#         return ans

# 782. 变为棋盘

# class Solution:
#     def movesToChessboard(self, board: List[List[int]]) -> int:
#         n = len(board)
#         # 棋盘的第一行与第一列
#         rowMask = colMask = 0
#         for i in range(n):
#             rowMask |= board[0][i] << i
#             colMask |= board[i][0] << i
#         reverseRowMask = ((1 << n) - 1) ^ rowMask
#         reverseColMask = ((1 << n) - 1) ^ colMask
#         rowCnt = colCnt = 0
#         for i in range(n):
#             currRowMask = currColMask = 0
#             for j in range(n):
#                 currRowMask |= board[i][j] << j
#                 currColMask |= board[j][i] << j
#             # 检测每一行和每一列的状态是否合法
#             if currRowMask != rowMask and currRowMask != reverseRowMask or \
#                currColMask != colMask and currColMask != reverseColMask:
#                 return -1
#             rowCnt += currRowMask == rowMask  # 记录与第一行相同的行数
#             colCnt += currColMask == colMask  # 记录与第一列相同的列数

#         def getMoves(mask: int, count: int) -> int:
#             ones = mask.bit_count()
#             if n & 1:
#                 # 如果 n 为奇数，则每一行中 1 与 0 的数目相差为 1，且满足相邻行交替
#                 if abs(n - 2 * ones) != 1 or abs(n - 2 * count) != 1:
#                     return -1
#                 if ones == n // 2:
#                     # 偶数位变为 1 的最小交换次数
#                     return n // 2 - (mask & 0xAAAAAAAA).bit_count()
#                 else:
#                     # 奇数位变为 1 的最小交换次数
#                     return (n + 1) // 2 - (mask & 0x55555555).bit_count()
#             else:
#                 # 如果 n 为偶数，则每一行中 1 与 0 的数目相等，且满足相邻行交替
#                 if ones != n // 2 or count != n // 2:
#                     return -1
#                 # 偶数位变为 1 的最小交换次数
#                 count0 = n // 2 - (mask & 0xAAAAAAAA).bit_count()
#                 # 奇数位变为 1 的最小交换次数
#                 count1 = n // 2 - (mask & 0x55555555).bit_count()
#                 return min(count0, count1)

#         rowMoves = getMoves(rowMask, rowCnt)
#         colMoves = getMoves(colMask, colCnt)
#         return -1 if rowMoves == -1 or colMoves == -1 else rowMoves + colMoves


# # 658. 找到 K 个最接近的元素

# class Solution:
#     def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
#         right = bisect_left(arr, x)
#         left = right - 1
#         for _ in range(k):
#             if left < 0:
#                 right += 1
#             elif right >= len(arr) or x - arr[left] <= arr[right] - x:
#                 left -= 1
#             else:
#                 right += 1
#         return arr[left + 1: right]


# # 1464. 数组中两元素的最大乘积

# class Solution:
#     def maxProduct(self, nums: List[int]) -> int:
#         a, b = nums[0], nums[1]
#         if a < b:
#             a, b = b, a
#         for i in range(2, len(nums)):
#             num = nums[i]
#             if num > a:
#                 a, b = num, a
#             elif num > b:
#                 b = num
#         return (a - 1) * (b - 1)


# # 998. 最大二叉树 II

# class Solution:
#     def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
#         parent, cur = None, root
#         while cur:
#             if val > cur.val:
#                 if not parent:
#                     return TreeNode(val, root, None)
#                 node = TreeNode(val, cur, None)
#                 parent.right = node
#                 return root
#             else:
#                 parent = cur
#                 cur = cur.right
        
#         parent.right = TreeNode(val)
#         return root


# # 946. 验证栈序列

# class Solution:
#     def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
#         st, j = [], 0
#         for x in pushed:
#             st.append(x)
#             while st and st[-1] == popped[j]:
#                 st.pop()
#                 j += 1
#         return len(st) == 0


# 1 2 3 4 5
# 4 5 3 2 1
# 4 5 3 1 2

# 1 2 3 4 -> for 1 2 3 -> 1 2 3 5 -> 1 2 3 -> 1 2 -> 1 -> 0

# 1475. 商品折扣后的最终价格

# class Solution:
#     def finalPrices(self, prices: List[int]) -> List[int]:
#         n = len(prices)
#         ans = [0] * n
#         st = [0]
#         for i in range(n - 1, -1, -1):
#             p = prices[i]
#             while len(st) > 1 and st[-1] > p:
#                 st.pop()
#             ans[i] = p - st[-1]
#             st.append(p)
#         return ans

# 687. 最长同值路径

# class Solution:
#     def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
#         ans = 0
#         def dfs(node: Optional[TreeNode]) -> int:
#             if node is None:
#                 return 0
#             left = dfs(node.left)
#             right = dfs(node.right)
#             left1 = left + 1 if node.left and node.left.val == node.val else 0
#             right1 = right + 1 if node.right and node.right.val == node.val else 0
#             nonlocal ans
#             ans = max(ans, left1 + right1)
#             return max(left1, right1)
#         dfs(root)
#         return ans

# 652. 寻找重复的子树

# class Solution:
#     def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
#         def dfs(node: Optional[TreeNode]) -> str:
#             if not node:
#                 return ""
            
#             serial = "".join([str(node.val), "(", dfs(node.left), ")(", dfs(node.right), ")"])
#             if (tree := seen.get(serial, None)):
#                 repeat.add(tree)
#             else:
#                 seen[serial] = node
            
#             return serial
        
#         seen = dict()
#         repeat = set()

#         dfs(root)
#         return list(repeat)


# 1592. 重新排列单词间的空格

class Solution:
    def reorderSpaces(self, text: str) -> str:
        words = text.split()
        space = text.count(' ')
        if len(words) == 1:
            return words[0] + ' ' * space
        per_space, rest_space = divmod(space, len(words) - 1)
        return (' ' * per_space).join(words) + ' ' * rest_space

# 667. 优美的排列 II

class Solution:
    def constructArray(self, n: int, k: int):
        answer = list(range(1, n - k))
        i, j = n - k, n
        while i <= j:
            answer.append(i)
            if i != j:
                answer.append(j)
            i, j = i + 1, j - 1
        return answer

a = Solution()
a.constructArray(3, 2)

# 给你两个整数 n 和 k ，请你构造一个答案列表 answer ，该列表应当包含从 1 到 n 的 n 个不同正整数，并同时满足下述条件：

# 示例 1：

# 输入：n = 3, k = 1
# 输出：[1, 2, 3]
# 解释：[1, 2, 3] 包含 3 个范围在 1-3 的不同整数，并且 [1, 1] 中有且仅有 1 个不同整数：1
# 示例 2：

# 输入：n = 3, k = 2
# 输出：[1, 3, 2]
# 解释：[1, 3, 2] 包含 3 个范围在 1-3 的不同整数，并且 [2, 1] 中有且仅有 2 个不同整数：1 和 2



class Solution:
    def flipLights(self, n: int, presses: int) -> int:
        seen = set()
        for i in range(2**4):
            pressArr = [(i >> j) & 1 for j in range(4)]
            if sum(pressArr) % 2 == presses % 2 and sum(pressArr) <= presses:
                status = pressArr[0] ^ pressArr[1] ^ pressArr[3]
                if n >= 2:
                    status |= (pressArr[0] ^ pressArr[1]) << 1
                if n >= 3:
                    status |= (pressArr[0] ^ pressArr[2]) << 2
                if n >= 4:
                    status |= (pressArr[0] ^ pressArr[1] ^ pressArr[3]) << 3
                seen.add(status)
        return len(seen)

# 1640 easy

class Solution:
    def canFormArray(self, arr, pieces):
        index = {p[0]: i for i, p in enumerate(pieces)}
        i = 0
        while i < len(arr):
            if arr[i] not in index:
                return False
            p = pieces[index[arr[i]]]
            if arr[i: i + len(p)] != p:
                return False
            i += len(p)
        return True

# a = Solution()
# b = [91,4,64,78]
# c = [[78],[4,64],[91]]
# a.canFormArray(b, c)

# class Solution:
#     def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
#         n, m = len(arr), len(pieces)
#         hash = [0] * 110
#         for i in range(m):
#             hash[pieces[i][0]] = i
#         i = 0
#         while i < n:
#             cur = pieces[hash[arr[i]]]
#             sz, idx = len(cur), 0
#             while idx < sz and cur[idx] == arr[i + idx]:
#                 idx += 1
#             if idx == sz:
#                 i += sz
#             else:
#                 return False
#         return True

# 6189
# 解题思路
# 因为 a&b <= min(a,b) ，所以 k = max(nums)，只要求出 nums 中连续 k 个数的最大值即可

# 简单动态规划
# dp[i]=dp[i-1]+1(dp[i]=k)
# dp[i]=1(dp[i]!=k)

# class Solution:
#     def longestSubarray(self, nums: List[int]) -> int:
#         n = len(nums)
#         k = max(nums)  # 找出最大值
#         ans = cnt = 0
#         for i in nums:
#             # 统计每段连续 k 的长度
#             if i == k:
#                 cnt += 1
#             else:
#                 cnt = 0
#             ans = max(ans, cnt)
#         return ans



# 6190

# 我是时间复杂度O(3n),没那么快，但我觉得我的思路还是可以的
# 计算两个数组，
# 一个数组表示当前点左边非递增的最远下标
# 一个数组表示当前点右边非递减的最远下标
# 最后判断k~n-k中的每个点
# 左边一个点的值要比i-k小，右边一个值比i+k要大就满足
# 值得注意的是，不包含当前点，当前点可以超出

# class Solution(object):
#     def goodIndices(self, nums, k):
#         n = len(nums)
#         jian = [0] *n
#         zeng = [n-1] *n
#         res = []
#         for i in range(1,n):
#             jian[i] = i if nums[i] > nums[i-1] else jian[i-1]
#         for i in range(n-2,-1,-1):
#             zeng[i] = i if nums[i] > nums[i+1] else zeng[i+1]
#         print(jian,zeng)
#         for i in range(k,n-k):
#             if jian[i-1] <= i-k and zeng[i+1] >= i+k:
#                 res += [i]
#         return res

# nums = [2,1,1,1,3,4,1]
# k = 2
# a = Solution()
# a.goodIndices(nums, k)

# 921
# 遍历s，左括号+1，大于0的情况下右括号-1，否则答案+1，最后答案+剩余的

# class Solution:
#     def minAddToMakeValid(self, s: str) -> int:
#         score, ans = 0, 0
#         for c in s:
#             score += 1 if c == '(' else -1
#             if score < 0:
#                 score = 0
#                 ans += 1
#         return ans + score


# 1800
# class Solution:
#     def maxAscendingSum(self, nums) -> int:
#         ans = sums = nums[0]
#         for i in range(1, len(nums)):
#             if nums[i] > nums[i - 1]:
#                 sums += nums[i]
#             else:
#                 sums = nums[i]
#             ans = max(ans, sums)

#         return ans

# a = [10,20,30,5,10,50]
# b = Solution()
# b.maxAscendingSum(a)

# class Solution:
#     def maxAscendingSum(self, nums: List[int]) -> int:
#         ans = 0
#         i, n = 0, len(nums)
#         while i < n:
#             s = nums[i]
#             i += 1
#             while i < n and nums[i] > nums[i - 1]:
#                 s += nums[i]
#                 i += 1
#             ans = max(ans, s)
#         return ans



# 817
# num 存 hash 
# 遍历记录有几个部分
# 断开再存在+1
# class Solution:
#     def numComponents(self, head, nums) -> int:
#         ans = 0
#         nset = set([x for x in nums])
#         while head:
#             if head.val in nset:
#                 while head and head.val in nset:
#                     head = head.next
#                 ans += 1
#             else:
#                 head = head.next
#         return ans


# class Solution:
#     def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
#         numsSet = set(nums)
#         inSet = False
#         res = 0
#         while head:
#             if head.val not in numsSet:
#                 inSet = False
#             elif not inSet:
#                 inSet = True
#                 res += 1
#             head = head.next
#         return res


# class Solution:
#     def maxChunksToSorted(self, arr: List[int]) -> int:
#         n, ans = len(arr), 0
#         j, minv, maxv = 0, n, -1
#         for i in range(n):
#             minv, maxv = min(minv, arr[i]), max(maxv, arr[i])
#             if j == minv and i == maxv:
#                 ans, j, minv, maxv = ans + 1, i + 1, n, -1
#         return ans

# filename = '123454terf00'
# filename = filename[:-2]
# print(filename)


# 934 bfs * 2
class Solution:
    def shortestBridge(self, grid) -> int:
        n = len(grid)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v != 1:
                    continue
                island = []
                grid[i][j] = -1
                q = deque([(i, j)])
                while q:
                    x, y = q.popleft()
                    island.append((x, y))
                    for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                        if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
                            grid[nx][ny] = -1
                            q.append((nx, ny))

                step = 0
                q = island
                while True:
                    tmp = q
                    q = []
                    for x, y in tmp:
                        for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                            if 0 <= nx < n and 0 <= ny < n:
                                if grid[nx][ny] == 1:
                                    return step
                                if grid[nx][ny] == 0:
                                    grid[nx][ny] = -1
                                    q.append((nx, ny))
                    step += 1
# # 地图坐标
# grid = list()
# # 两个for 遍历找到的第一个为1的坐标的位置
# i = j = 1
# # bfs 的四个方向
# coordinate = [(0, 1), (0, -1), (1, 0), (-1, 0)]
# # 记录A岛的所有坐标
# islandA = list()
# # 队列
# q = deque([(i, j)])
# # bfs搜索 如果队列里有东西就继续 没有了就结束
# while q:
#     # 弹出来一组坐标
#     x, y = q.popleft()
#     # 记录A岛坐标
#     islandA.append((x, y))
#     # 遍历四个方向
#     for i, j in coordinate:
#         # 在板子的边界内且坐标为1，也就是A岛屿的部分
#         if 0 <= x + i < len(grid) and 0 <= y + j < len(grid[0]) and grid[x + i][y + j] == 1:
#             # 变成2，防止重复
#             grid[x + i][y + j] = 2
#             # 把这个坐标加入队列，回到最外面下一轮循环，遍历这个点的四个方向
#             q.append((x + i, y + j))

# for x, y in q:
#     for i, j in coordinate:
#         # 在板子的边界内且坐标为1，也就是A岛屿的部分
#         if 0 <= x + i < len(grid) and 0 <= y + j < len(grid[0]) and grid[x + i][y + j] == 1:
#             # 改这里 
#             if grid[x + i][y + j] == 0:
#                 grid[x + i][y + j] = 2
#             if grid[x + i][y + j] == 1:
#                 break
#             if grid[x + i][y + j] == 1:
#                 pass
#     ans += 1

T = TypeVar('T')
print(T)
def repeat(x: T, n: int):
    """Return a list containing n references to x."""
    return [x]*n


a = repeat(1,5)
print(a)

















