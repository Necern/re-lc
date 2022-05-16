# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# class Solution:
#     def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> TreeNode:
#         q = []
#         cur = root
#         pre = None

#         while q or cur:
#             # 左
#             while cur:
#                 q.append(cur)
#                 cur = cur.left
#             # 判
#             cur = q.pop()
#             if pre == p:
#                 return cur
#             # 右
#             pre = cur
#             cur = cur.right

#         return None

class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> TreeNode:
        # bst
        def foo(root):
            nonlocal ans
            if not root:
                return
            if root.val > p.val:
                ans = root
                foo(root.left)
            else:
                foo(root.right)

        ans = None
        foo(root)
        return ans
