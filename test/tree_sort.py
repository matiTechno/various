import random

class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

def insert_impl(node, value):

#    if node.value == value:
#        return

    if value > node.value:
        if not node.right:
            node.right = Node(value)
            return
        else:
            insert_impl(node.right, value)

    else:
        if not node.left:
            node.left = Node(value)
            return
        else:
            insert_impl(node.left, value)

def to_list_impl(list, node):
    if node.left:
        to_list_impl(list, node.left)

    list.append(node.value)

    if node.right:
        to_list_impl(list, node.right)

def bt_insert(tree, value):
    insert_impl(tree, value)

def bt_to_list(tree):
    list = []
    to_list_impl(list, tree)
    return list

if __name__ == '__main__':

    values = []

    for i in range(20):
        values.append(random.randint(0,100))

    print(values)

    root = Node(-1)

    for v in values:
        bt_insert(root, v)

    sorted = bt_to_list(root)
    sorted.pop(0)
    print(sorted)
