# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    # 前序遍历 + 分隔符 | 因为是二叉搜索树（有序: 左 < 根 < 右） 所以不需要 前中 / 中后
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.
        """
        ans = ''
        lst = []
        def foo(node, lst):
            if not node:
                return
            lst.append(str(node.val))
            foo(node.left, lst)
            foo(node.right, lst)
        
        foo(root, lst)
        return ','.join(map(str, lst))

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        if not data:
            return []
        # # lst = ['2', '1', '3', '']
        # lst = data.split(',')
        lst = list(map(int, data.split(',')))

        def foo(l, r):
            if l > r:
                return
            i = l + 1
            while i <= r and lst[i] <= lst[l]:
                i += 1

            node = TreeNode(lst[l])
            node.left = foo(l + 1, i - 1)
            node.right = foo(i, r)
            return node
        
        l, r = 0, len(lst) - 1
        root = foo(l, r)

        return root

# # Your Codec object will be instantiated and called as such:
# # Your Codec object will be instantiated and called as such:
# # ser = Codec()
# # deser = Codec()
# # tree = ser.serialize(root)
# # ans = deser.deserialize(tree)
# # return ans
