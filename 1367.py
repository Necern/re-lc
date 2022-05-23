# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def foo(self, node, root):
        if not node:
            return True
        if not root:
            return False
        if root.val != node.val:
            return False
        return self.foo(node.next, root.left) or self.foo(node.next, root.right)

    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        if not root:
            return False
        return self.foo(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)


