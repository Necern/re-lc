class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        dct = {ele: n for n, ele in enumerate(order)}
        flag = 0
        for i in range(len(words) - 1):
            for j in range(min(len(words[i]), len(words[i + 1]))):
                if dct[words[i][j]] < dct[words[i + 1][j]]:
                    break
                elif dct[words[i][j]] == dct[words[i + 1][j]]:
                    continue
                else:
                    return False
            if dct[words[i][j]] == dct[words[i + 1][j]] and j < len(words[i]) - 1:
                return False            

        return True


# py 3.10 +
# itertools
# def pairwise(iterable):
#     # pairwise('ABCDEFG') --> AB BC CD DE EF FG
#     a, b = tee(iterable)
#     next(b, None)
#     return zip(a, b)

# def all(iterable):
#     for element in iterable:
#         if not element:
#             return False
#     return True

# class Solution:
#     def isAlienSorted(self, words: List[str], order: str) -> bool:
#         index = {c: i for i, c in enumerate(order)}
#         return all(s <= t for s, t in pairwise([index[c] for c in word] for word in words))
