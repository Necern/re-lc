class Solution:
    def findClosest(self, words: List[str], word1: str, word2: str) -> int:
        dct = defaultdict(list)
        for i, ch in enumerate(words):
            dct[ch].append(i)
        
        i1 = i2 = 0
        n1, n2 = len(dct[word1]), len(dct[word2])
        ans = len(words)
        while i1 < n1 and i2 < n2:
            ans = min(ans, abs(dct[word1][i1] - dct[word2][i2]))
            if dct[word1][i1] < dct[word2][i2]:
                i1 += 1
            else:
                i2 += 1

        return ans 

