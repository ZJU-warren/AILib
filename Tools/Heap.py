class Node:
    lson = None
    rson = None

    def __init__(self, key, info):
        self.key = key
        self.info = info

def swap(a, b):
    t = a
    a = b
    b = t
    return a, b

class Heap:
    tree = [Node(None, None)]
    cnt = 0

    def __init__(self, cmp_func):
        self.cmp_func = cmp_func

    def push(self, node):
        self.tree.append(node)
        self.cnt += 1
        idx = self.cnt

        while idx // 2 != 0:
            if self.cmp_func(self.tree[idx].key, self.tree[idx//2].key):
                self.tree[idx], self.tree[idx // 2] = swap(self.tree[idx], self.tree[idx // 2])
            else:
                break
            idx //= 2

    def pop(self):
        res = self.tree[1]
        self.cnt -= 1
        if self.cnt != 0:
            self.tree[1] = self.tree.pop()
            idx = 1
            while idx < self.cnt:
                if idx * 2 <= self.cnt and not self.cmp_func(self.tree[idx].key, self.tree[idx * 2].key):
                    self.tree[idx], self.tree[idx * 2] = swap(self.tree[idx], self.tree[idx * 2])
                elif idx * 2 + 1 <= self.cnt and not self.cmp_func(self.tree[idx].key, self.tree[idx * 2 + 1].key):
                    self.tree[idx], self.tree[idx * 2 + 1] = swap(self.tree[idx], self.tree[idx * 2 + 1])
                else:
                    break
        return res

    def top(self):
        return self.tree[1]

    def merge(self, target, k=-1):
        # step 1: reverse target
        temp_target = Heap(not target.cmp_func)
        while target.cnt != 0:
            temp_target.push(target.pop())
        if k == -1:
            k = self.cnt + target.cnt

        while self.cnt + target.cnt > k:
            if not self.cmp_func(self.top().key, target.top().key):
                self.pop()
            else:
                target.pop()


if __name__ == '__main__':
    heap = Heap(lambda a, b: a < b)
    heap.push(Node(2, 1))
    heap.push(Node(3, 1))
    heap.push(Node(2, 1))
    print(heap.pop().key)
    print(heap.pop().key)
    print(heap.pop().key)
