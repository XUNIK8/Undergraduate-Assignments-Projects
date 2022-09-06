class TreeSet(object):
    class TreeNode(object):
        def __init__(self, value, parent=None, right=None, left=None):
            self.value = value
            self.right = right
            self.left = left
            self.parent = parent

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

    def size(self):
        return self.num

    def isEmpty(self):
        return self.num == 0

    def clear(self):
        self.root = None
        self.num = 0

    def add(self, value):
        parent = self.root
        node = self.root

        if value is None:
            raise SystemExit('It failed!')

        if self.root is None:
            self.root = self.TreeNode(value)
            self.num = 1
            return

        while node is not None:
            compare = self._compare(value, node.value)
            parent = node
            if compare > 0:
                node = node.right
            elif compare < 0:
                node = node.left
            else:
                node.value = value
                return

        newNode = self.TreeNode(value, parent)
        if compare > 0:
            parent.right = newNode
        else:
            parent.left = newNode
        self.num += 1

    def remove(self, value):
        node = self._node(value)
        if node is None:
            return
        self._remove(node)

    def contains(self, value):
        return self._node(value) is not None

    def traversal(self, visitor):
        if visitor is None:
            return
        self._traversalAll(self.root, visitor)

    def _compare(self, x, y):
        if x == y:
            return 0
        elif x < y:
            return -1
        else:
            return 1
            
    # find node
    def _node(self, value):
        if value is None:
            raise SystemExit('It failed!')
        node = self.root
        while node is not None:
            compare = self._compare(value, node.value)
            if compare > 0:
                node = node.right
            elif compare < 0:
                node = node.left
            else:
                return node
        return None

    # remove node
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
            node.value = preNode.value
            self._remove(preNode)

    # find predecesor node
    def _preNode(self, node):
        cur = node.left
        if cur is not None:
            while cur.right is not None:
                cur = cur.right
            return cur
        while node.parent is not None and node == node.parent.left:
            node = node.parent
        return node.parent

    # in-order traversal of binary search tree
    def _traversalAll(self, root, visitor):
        if root is None or visitor.stop:
            return
        self._traversalAll(root.left, visitor)
        if visitor.stop:
            return
        visitor.stop = visitor.visit(root.value)
        self._traversalAll(root.right, visitor)

if __name__ == "__main__":
    class visitor:
        def __init__(self):
            self.stop = False

        def visit(self, value):
            print(str(value),end = ' ')
            return False

    treeSet = TreeSet()
    visitor = visitor()

    List = [2,3,1,6,5,4]
    for value in List:
        treeSet.add(value)

    # test all the functions
    # size
    print('Size of treeset : ', treeSet.size())
    # isEmpty
    print('IsEmpty : ', treeSet.isEmpty())
    # remove
    treeSet.remove(3)
    # size
    print('Size of treeset : ', treeSet.size())
    # contains
    print('Contains \"3\" : ', treeSet.contains(3))
    print('Contains \"4\" : ', treeSet.contains(4))
    # traversal
    print('Traversal outcome:')
    treeSet.traversal(visitor)
    # clear
    treeSet.clear()
    print('\nTraversal outcome :',treeSet.traversal(visitor))
    # isEmpty
    print('IsEmpty : ', treeSet.isEmpty())
