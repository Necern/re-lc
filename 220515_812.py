class Solution:
    def largestTriangleArea(self, points: List[List[int]]) -> float:
        # S = 1/2 * |(x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)|
        n = len(points)
        ans = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i][0], points[i][1]
                    x2, y2 = points[j][0], points[j][1]
                    x3, y3 = points[k][0], points[k][1]
                    tri = abs(((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / 2.0)
                    ans = max(ans, tri)
        return ans

