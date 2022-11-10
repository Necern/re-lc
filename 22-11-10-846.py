class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        m, n = len(grid), len(grid[0])
        keys = 0

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '@':
                    startX, startY = i, j
                    grid[i] = grid[i][:j] + '.' + grid[i][j + 1:]
                elif grid[i][j].islower():
                    keys += 1

        q = deque([[startX, startY, '']])
        ans = defaultdict(int)
        nk = ''

        while q:
            x, y, k = q.popleft()
            for i, j in ([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]):
                if 0 <= i < m and 0 <= j < n:
                    # 钥匙 a
                    if grid[i][j].islower() and (i, j, k) not in ans:
                        if grid[i][j] not in k:
                            nk = k
                            nk += grid[i][j]
                            ans[(i, j, nk)] = ans[(x, y, k)] + 1
                            q.append([i, j, nk])

                            if len(nk) == keys:
                                return ans[(i, j, nk)]
                        # 回头的情况
                        else:
                            ans[(i, j, k)] = ans[(x, y, k)] + 1
                            q.append([i, j, k])
                    # 锁 A
                    elif grid[i][j].isupper() and grid[i][j].lower() in k and (i, j, k) not in ans:
                        ans[(i, j, k)] = ans[(x, y, k)] + 1
                        q.append([i, j, k])
                    # 正常 .
                    elif grid[i][j] == '.' and (i, j, k) not in ans:
                            ans[(i, j, k)] = ans[(x, y, k)] + 1
                            q.append([i, j, k])

        return -1
