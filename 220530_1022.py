# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
#         def foo(root, val):
#             if not root:
#                 return 0
#             ans = 0
#             val = (val << 1) | root.val
#             if root.left:
#                 ans += foo(root.left, val)
#             if root.right:
#                 ans += foo(root.right, val)
#             return val if not root.left and not root.right else ans

#         return foo(root, 0)

# class Solution:
#     def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
#         def foo(root, val):
#             nonlocal ans
#             val = (val << 1) | root.val
#             if root.left:
#                 foo(root.left, val)
#             if root.right:
#                 foo(root.right, val)
#             if not root.left and not root.right:
#                 ans += val

#         ans = 0
#         foo(root, 0)
#         return ans

class Solution:
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        def foo(root, val):
            if not root:
                return 0
            ans = 0
            val = (val << 1) | root.val
            if not root.left and not root.right:
                return val
            return foo(root.left, val) + foo(root.right, val)

        return foo(root, 0)



