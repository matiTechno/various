import math

def query(list, key):
    assert len(list)
    left = 0
    right = len(list) - 1

    while left != right:
        mid = math.ceil((right + left) / 2)
        if key == list[mid]:
            left = right = mid
        elif key < list[mid]:
            right = mid - 1
        else:
            left = mid + 1

    mid = left
    ln = mid # left neighbour
    rn = mid

    if list[mid] != key:
        print(f'NO key ({key}) found')

        if list[mid] < key:
            rn += 1
        else:
            ln -= 1

    else:
        while ln > 0 and list[ln - 1] == key:
            ln -= 1

        while (rn < len(list) - 1) and list[rn + 1] == key:
            rn += 1

        if ln == rn:
            print(f'key ({key}) at position {mid}')
        else:
            print(f'key ({key}) at positions [{ln}, {rn}]')

        ln -= 1
        rn += 1

    if ln >= 0:
        v = list[ln]
        print(f'left neighbour: list[{ln}] = {v}')
    else:
        print('no left neighbour')

    if rn < len(list):
        v = list[rn]
        print(f'right neighbour: list[{rn}] = {v}')
    else:
        print('no right neighbour')

    print()

list = [0,2,8,9,15,16,17,18,19,19,19,20,25,33,44,100]
list.sort()
query(list, 0)
query(list, 2)
query(list, 100)
query(list, 19)
query(list, 14)
query(list, -1)
