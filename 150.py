class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        ans = 0
        for ch in tokens:
            if ch not in ['+', '-', '*', '/']:
                stack.append(int(ch))
            else:
                b = stack.pop()
                a = stack.pop()
                if ch == '+':
                    ans = a + b
                elif ch == '-':
                    ans = a - b
                elif ch == '*':
                    ans = a * b
                else:
                    ans = int(a / b)
                stack.append(ans)
        return stack.pop()
