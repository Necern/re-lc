class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        # bfs? -> cal step
        # change start -> del bank -> if not bank, -1 | if end, step

        q = collections.deque()
        q.append((start, 0))
        while q:
            n = len(q)
            s, step = q.popleft()
            if s == end:
                return step

            while n:
                for i in range(len(s)):
                    for ch in ['A', 'C', 'G', 'T']:
                        tmp = s[0: i] + ch + s[i + 1:]
                        if tmp in bank:
                            q.append((tmp, step + 1))
                            bank.remove(tmp)
                n -= 1

        return -1
