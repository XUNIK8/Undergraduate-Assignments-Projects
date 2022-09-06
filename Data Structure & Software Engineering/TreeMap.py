class TreeMap(object):
    class TreeNode(object):
        def __init__(self, key, value, parent=None, right=None, left=None):
            self.key = key
            self.value = value
            self.parent = parent
            self.right = right
            self.left = left

        def degree(self):
            deg = 0
            if self.left is not None:
                deg += 1
            if self.right is not None:
                deg += 1
            return deg

    def __init__(self):
        self.num = 0
        self.root = None
        self.stop = False

    def size(self):
        return self.num

    def isEmpty(self):
        return self.num == 0

    def clear(self):
        self.root = None
        self.num = 0

    def put(self, key, value):
        parent = self.root
        node = self.root
        if key is None:
            raise SystemExit('It failed!')
        if self.root is None:
            self.root = self.TreeNode(key, value)
            self.num = 1
            return None

        while node is not None:
            compare = self._compare(key, node.key)
            parent = node
            if compare > 0:
                node = node.right
            elif compare < 0:
                node = node.left
            else:
                node.key = key
                old = node.value
                node.value = value
                return old

        newNode = self.TreeNode(key, value, parent)
        if compare > 0:
            parent.right = newNode
        else:
            parent.left = newNode
        self.num += 1

    def get(self, key):
        node = self._node(key)
        return node.value if node is not None else None

    def remove(self, key):
        node = self._node(key)
        if node is None:
            return
        self._remove(node)

    def containsKey(self, key):
        return self._node(key) is not None

    def containsValue(self, value):
        if self.root is None:
            return False
        self._traversalFind(self.root, value)
        if self.stop:
            self.stop = False
            return True
        else:
            return False

    def traversal(self, visitor):
        if visitor is None:
            return
        self._traversalAll(self.root, visitor)

    # comparator
    def _compare(self, x, y):
        if x == y:
            return 0
        elif x < y:
            return -1
        else:
            return 1

    # find node by key
    def _node(self, key):
        if key is None:
            raise SystemExit('It failed!')
        node = self.root
        while node is not None:
            compare = self._compare(key, node.key)
            if compare > 0:
                node = node.right
            elif compare < 0:
                node = node.left
            else:
                return node
        return None

    # remove node by key
    def _remove(self, node):
        self.num -= 1

        degree = node.degree()
        if degree == 0:
            if node == self.root:
                self.root = None
            elif node == node.parent.left:
                node.parent.left = None
            else:
                node.parent.right = None
        elif degree == 1:
            child = node.left if node.left is not None else node.right
            if node == self.root:
                self.root = child
                self.root.parent = None
            else:
                child.parent = node.parent
                if node == node.parent.left:
                    node.parent.left = child
                else:
                    node.parent.right = child
        else:
            preNode = self._preNode(node)
            node.key = preNode.key
            self._remove(preNode)

    # finde predecesor node
    def _preNode(self, node):
        cur = node.left
        if cur is not None:
            while cur.right is not None:
                cur = cur.right
            return cur
        while node.parent is not None and node == node.parent.left:
            node = node.parent
        return node.parent

    # in-order traversal to find a specific node
    def _traversalFind(self, root, value):
        if root is None or self.stop:
            return
        self._traversalFind(root.left, value)
        if root.value == value:
            self.stop = True
        self._traversalFind(root.right, value)

    # in-order traversal of binary search tree
    def _traversalAll(self, root, visitor):
        if root is None or visitor.stop:
            return
        self._traversalAll(root.left, visitor)
        if visitor.stop:
            return
        visitor.stop = visitor.visit(root.key, root.value)
        self._traversalAll(root.right, visitor)

if __name__ == "__main__":
    class visitor:
        def __init__(self):
            self.stop = False

        def visit(self, key, value):
            print(str(key)+':'+str(value),end = ' ')
            return False

    treeMap = TreeMap()
    visitor = visitor()

    dict = {'f':2,
            'a':3,
            'b':1,
            'e':4,
            'b':5}
    for key in list(dict.keys()):
        treeMap.put(key, dict[key])

    # test all the functions
    # traversal
    treeMap.traversal(visitor)
    # size
    print('\nSize of treemap : ', treeMap.size())
    # isEmpty
    print('Isempty : ', treeMap.isEmpty())
    # remove
    # remove
    treeMap.remove('a')
    # size
    print('Size of treemap : ', treeMap.size())
    # get
    print('Value of \"e\" : ', treeMap.get('e'))
    # containsKey
    print('Contains Key \"a\" : ', treeMap.containsKey('a'))
    # containsValue
    print('Contains Value \"20\" : ', treeMap.containsValue(4))
    # traversal
    print('Traversal outcome :')
    treeMap.traversal(visitor)
    # clear
    treeMap.clear()
    print('\nTraversal outcome :',treeMap.traversal(visitor))
    # isEmpty
    print('IsEmpty : ', treeMap.isEmpty())
