# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def isUnivalTree(self, root: TreeNode) -> bool:
#         def foo(root):
#             nonlocal val
#             if val == -1:
#                 val = root.val
#             if not root:
#                 return True
#             if root.val != val:
#                 return False
#             return foo(root.left) and foo(root.right)

#         val = -1
#         return foo(root)


class Solution:
    def isUnivalTree(self, root: TreeNode) -> bool:
        if not root:
            return True

        if root.left:
            if root.val != root.left.val:
                return False
            if not self.isUnivalTree(root.left):
                return False

        if root.right:
            if root.val != root.right.val:
                return False
            if not self.isUnivalTree(root.right):
                return False

        return True

