from typing import Protocol, Any, Generator, Generic, Optional, TypeVar, Union


T = TypeVar("T")


class Node(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
        self.next: Node[T] | None = None

    def __repr__(self) -> str:
        current = self.value
        next = None if self.next is None else self.next.value
        return f"Node(current={current}, next={next})"


class LinkedList(Generic[T]):
    def __init__(self) -> None:
        self.head: Node[T] | None = None
        self.tail: Node[T] | None = None
        self.size: int = 0

    def __iter__(self) -> Generator[Node[T], None, None]:
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def __repr__(self) -> str:
        values = []
        for node in self:
            values.append(str(node.value))
        return " -> ".join(values) if values else "Empty Linked List"

    def is_empty(self) -> bool:
        return self.size == 0

    def append_head(self, value: T) -> None:
        node = Node(value)
        if self.is_empty():
            self.head, self.tail = node, node

        elif self.head is not None:
            previous_head, new_head = self.head, node
            new_head.next = previous_head
            self.head = new_head

        self.size += 1

    def append_tail(self, value: T) -> None:
        node = Node(value)
        node.next = None
        if self.is_empty():
            self.head, self.tail = node, node

        if self.tail is not None:
            previous_tail, new_tail = self.tail, node
            previous_tail.next = new_tail
            self.tail = new_tail

        self.size += 1

    def remove_head(self) -> T | None:
        previous_head = None
        if self.is_empty():
            return previous_head

        if self.head is not None:
            previous_head = self.head
            new_head = self.head.next
            self.head = new_head
            if self.head is None:
                self.tail = None

            self.size -= 1
            return previous_head.value


class Stack(Generic[T]):
    def __init__(self) -> None:
        self.ll = LinkedList[T]()
        self.size = self.ll.size

    def __repr__(self):
        return repr(self.ll) if not self.ll.is_empty() else "Empty Stack"

    def is_empty(self) -> bool:
        return self.ll.is_empty()

    def push(self, value: T) -> None:
        self.ll.append_head(value)

    def pop(self) -> T | None:
        return self.ll.remove_head()

    def peek(self) -> T | None:
        return self.ll.head.value if self.ll.head is not None else None


class Queue(Generic[T]):
    def __init__(self) -> None:
        self.ll = LinkedList[T]()
        self.size = self.ll.size

    def __repr__(self) -> str:
        values = []
        for node in self.ll:
            values.append(str(node.value))
        return " <- ".join(values) if values else "Empty Queue"

    def is_empty(self) -> bool:
        return self.ll.is_empty()

    def enqueue(self, value: T) -> None:
        self.ll.append_tail(value)

    def dequeue(self) -> T | None:
        return self.ll.remove_head()

    def first(self) -> T | None:
        return self.ll.head.value if self.ll.head is not None else None

    def last(self) -> T | None:
        return self.ll.tail.value if self.ll.tail is not None else None


def is_balanced(input_str: str) -> bool:
    if not input_str:
        raise ValueError("Input string can't be empty")

    if len(input_str) % 2 != 0:
        return False

    stack = Stack[str]()
    for s in input_str:
        if s == "(":
            stack.push(s)
        elif s == ")" and stack.peek() == "(":
            stack.pop()
        else:
            return False

    if stack.size > 1:
        return False

    return True


class User:
    def __init__(self, id: int, username: str) -> None:
        self.id = id
        self.username = username

    def __eq__(self, other: object) -> bool:
        return isinstance(other, User) and self.id == other.id

    def __lt__(self, other: object) -> bool:
        return isinstance(other, User) and self.id < other.id

    def __gt__(self, other: object) -> bool:
        return isinstance(other, User) and self.id > other.id

    def __repr__(self) -> str:
        return f'User(id={self.id}, username="{self.username}")'


class BSTNode:
    def __init__(self, value: User | None = None) -> None:
        self.value = value
        self.left: BSTNode | None = None
        self.right: BSTNode | None = None

    def min(self) -> User | None:
        if self.value is None:
            return None

        current = self
        while current.left:
            current = current.left
        return current.value

    def max(self) -> User | None:
        if self.value is None:
            return None

        current = self
        while current.right:
            current = current.right
        return current.value

    def preorder_traversal(self, visited: list[User] | None = None) -> list[User]:
        if visited is None:
            visited = []

        if self.value:
            visited.append(self.value)

        if self.left:
            self.left.preorder_traversal(visited)

        if self.right:
            self.right.preorder_traversal(visited)

        return visited

    def postorder_traversal(self, visited: list[User] | None = None) -> list[User]:
        if visited is None:
            visited = []

        if self.left:
            self.left.postorder_traversal(visited)

        if self.right:
            self.right.postorder_traversal(visited)

        if self.value:
            visited.append(self.value)

        return visited

    def inorder_traversal(self, visited: list[User] | None = None) -> list[User]:
        if visited is None:
            visited = []

        if self.left:
            self.left.inorder_traversal(visited)

        if self.value:
            visited.append(self.value)

        if self.right:
            self.right.inorder_traversal(visited)

        return visited

    def exists(self, value: User) -> bool:
        found = False

        if value == self.value:
            found = True

        if value < self.value and self.left:
            found = self.left.exists(value)

        if value > self.value and self.right:
            found = self.right.exists(value)

        return found

    def insert(self, value: User) -> None:
        if self.value is None:
            self.value = value
            return

        if value < self.value:
            if not self.left:
                self.left = BSTNode(value)
                return
            self.left.insert(value)

        if value > self.value:
            if not self.right:
                self.right = BSTNode(value)
                return
            self.right.insert(value)

        return

    def delete(self, value: User) -> Optional["BSTNode"]:
        """https://datastructureguru.quora.com/How-to-delete-node-from-binary-search-tree"""
        # Node already deleted
        if self.value is None:
            return None

        if value < self.value:
            if self.left:
                self.left = self.left.delete(value)
            return self

        if value > self.value:
            if self.right:
                self.right = self.right.delete(value)
            return self

        # Node found with one child
        if not self.left:
            return self.right

        if not self.right:
            return self.left

        # Node found with two children:
        # - Replace the current node value with the successor node value
        # - Delete the successor node value recursively
        successor = self.right
        while successor.left:
            successor = successor.left
        assert successor.value
        self.value = successor.value
        self.right = self.right.delete(successor.value)
        return self

    def search_range(self, lower_bound: User, upper_bound: User) -> list[User]:
        visited = []

        if self.left and self.value > lower_bound:
            visited.extend(self.left.search_range(lower_bound, upper_bound))

        visited.append(self.value)

        if self.right and self.value < upper_bound:
            visited.extend(self.right.search_range(lower_bound, upper_bound))

        return visited

    def depth(self) -> int:
        if self.value is None:
            return 0

        left_height, right_height = 0, 0
        if self.left:
            left_height = self.left.depth()
        if self.right:
            right_height = self.right.depth()

        return max(left_height, right_height) + 1


class RBNode:
    def __init__(self, value: User | None) -> None:
        self.red: bool = False
        self.parent: RBNode | None = None
        self.val: User | None = value
        self.left: RBNode | None = None
        self.right: RBNode | None = None

    def __repr__(self) -> str:
        color = "R" if self.red else "B"
        return f"RBNode({self.val}, {color})"


class RBTree:
    def __init__(self) -> None:
        # The nil sentinel node represents leaves.
        self.nil: RBNode = RBNode(None)
        self.nil.red = False
        # For convenience, set both children of nil to itself.
        self.nil.left = self.nil
        self.nil.right = self.nil
        self.root = self.nil

    def insert(self, value: User) -> None:
        new_node: RBNode = RBNode(value)
        new_node.parent = None
        new_node.left = self.nil
        new_node.right = self.nil
        new_node.red = True  # new nodes are inserted red

        parent: RBNode | None = None
        current: RBNode = self.root

        while current != self.nil:
            parent = current
            # Since we never insert a nil value, it's safe to assert current.val is not None.
            if new_node.val is not None and current.val is not None:
                if new_node.val < current.val:
                    current = current.left  # type: ignore
                elif new_node.val > current.val:
                    current = current.right  # type: ignore
                else:
                    # Duplicate value; do nothing.
                    return
            else:
                break

        new_node.parent = parent
        if parent is None:
            # Tree was empty
            self.root = new_node
        elif new_node.val is not None and parent.val is not None:
            if new_node.val < parent.val:
                parent.left = new_node
            elif new_node.val > parent.val:
                parent.right = new_node

        self.fix_insert(new_node)

    def fix_insert(self, new_node: RBNode) -> None:
        while (
            new_node != self.root
            and new_node.parent is not None
            and new_node.parent.red
        ):
            parent = new_node.parent
            grandparent = parent.parent
            if grandparent is None:
                break
            if parent == grandparent.right:
                uncle: RBNode = grandparent.left  # type: ignore
                if uncle.red:
                    # Case 1: Uncle is red.
                    uncle.red = False
                    parent.red = False
                    grandparent.red = True
                    new_node = grandparent
                else:
                    # Case 2 & 3: Uncle is black.
                    if new_node == parent.left:
                        new_node = parent
                        self.rotate_right(new_node)
                    parent.red = False
                    grandparent.red = True
                    self.rotate_left(grandparent)
            else:
                # Symmetric to the above
                uncle: RBNode = grandparent.right  # type: ignore
                if uncle.red:
                    uncle.red = False
                    parent.red = False
                    grandparent.red = True
                    new_node = grandparent
                else:
                    if new_node == parent.right:
                        new_node = parent
                        self.rotate_left(new_node)
                    parent.red = False
                    grandparent.red = True
                    self.rotate_right(grandparent)
        self.root.red = False

    def exists(self, value: User) -> RBNode:
        current: RBNode = self.root
        while current != self.nil and current.val is not None and current.val != value:
            if value < current.val:
                current = current.left  # type: ignore
            else:
                current = current.right  # type: ignore
        return current

    def rotate_left(self, pivot_parent: RBNode) -> None:
        # Ensure that the right child exists (i.e. is not the nil sentinel)
        if pivot_parent == self.nil or pivot_parent.right == self.nil:
            return

        pivot: RBNode = pivot_parent.right  # type: ignore
        pivot_parent.right = pivot.left
        if pivot.left != self.nil:
            pivot.left.parent = pivot_parent  # type: ignore

        pivot.parent = pivot_parent.parent
        if pivot_parent.parent is None:
            self.root = pivot
        elif pivot_parent == pivot_parent.parent.left:
            pivot_parent.parent.left = pivot
        else:
            pivot_parent.parent.right = pivot
        pivot.left = pivot_parent
        pivot_parent.parent = pivot

    def rotate_right(self, pivot_parent: RBNode) -> None:
        if pivot_parent == self.nil or pivot_parent.left == self.nil:
            return

        pivot: RBNode = pivot_parent.left  # type: ignore
        pivot_parent.left = pivot.right
        if pivot.right != self.nil:
            pivot.right.parent = pivot_parent  # type: ignore

        pivot.parent = pivot_parent.parent
        if pivot_parent.parent is None:
            self.root = pivot
        elif pivot_parent == pivot_parent.parent.right:
            pivot_parent.parent.right = pivot
        else:
            pivot_parent.parent.left = pivot
        pivot.right = pivot_parent
        pivot_parent.parent = pivot


K = TypeVar("K", bound=str)  # keys are strings for our custom hash function
V = TypeVar("V")


class HashMap(Generic[K, V]):
    def __init__(self, size: int = 8) -> None:
        self.MULTIPLIER = 31
        self.LOAD_THRESHOLD = 0.7
        self.RESIZE_FACTOR = 2
        self.hashmap: list[tuple[K, V] | None] = [None] * size

    def key_to_index(self, key: K) -> int:
        hash_value = 0
        for c in key:
            hash_value = hash_value * self.MULTIPLIER + ord(c)

        return hash_value % len(self.hashmap)

    def current_load(self) -> float:
        filled = 0
        for item in self.hashmap:
            if item is not None:
                filled += 1

        return filled / len(self.hashmap)

    def resize(self) -> None:
        old_hashmap = self.hashmap
        new_size = len(old_hashmap) * self.RESIZE_FACTOR
        self.hashmap = [None] * new_size

        for pair in old_hashmap:
            if pair is not None:
                key, value = pair
                self._insert_no_resize(key, value)

    def _insert_no_resize(self, key: K, value: V) -> None:
        index = self.key_to_index(key)
        start_index = index

        while self.hashmap[index] is not None:
            pair = self.hashmap[index]
            assert pair is not None
            if pair[0] == key:
                self.hashmap[index] = (key, value)
                return

            index = (index + 1) % len(self.hashmap)
            if index == start_index:
                raise Exception("Hashmap is full")

        self.hashmap[index] = (key, value)

    def insert(self, key: K, value: V) -> None:
        if self.current_load() > self.LOAD_THRESHOLD:
            self.resize()

        self._insert_no_resize(key, value)

    def get(self, key: K) -> V:
        index = self.key_to_index(key)
        start_index = index

        while self.hashmap[index] is not None:
            pair = self.hashmap[index]
            assert pair is not None
            current_key, current_value = pair
            if current_key == key:
                return current_value

            index = (index + 1) % len(self.hashmap)
            if index == start_index:
                break

        raise KeyError("Key not found")

    def __repr__(self) -> str:
        return "\n".join(
            f"Index {i}: {v}" for i, v in enumerate(self.hashmap) if v is not None
        )


TNode = dict[str, Union["TNode", bool]]


class Trie:
    def __init__(self) -> None:
        self.root: TNode = {}
        self.end_symbol: str = "*"

    def add(self, word: str) -> None:
        level: TNode = self.root
        for char in word:
            if char not in level:
                level[char] = {}
            level = level[char]  # type: ignore

        level[self.end_symbol] = True

    def exists(self, word: str) -> bool:
        level: TNode = self.root
        for char in word:
            if char not in level:
                return False
            level = level[char]  # type: ignore

        return self.end_symbol in level

    def search_level(
        self, current_level: TNode, current_prefix: str, words: list[str]
    ) -> list[str]:
        if self.end_symbol in current_level:
            words.append(current_prefix)

        for letter in sorted(current_level.keys()):
            if letter != self.end_symbol:
                self.search_level(current_level[letter], current_prefix + letter, words)  # type: ignore

        return words

    def words_with_prefix(self, prefix: str) -> list[str]:
        collected_words: list[str] = []
        level: TNode = self.root
        for letter in prefix:
            if letter not in level:
                return []
            level = level[letter]  # type: ignore

        return self.search_level(level, prefix, collected_words)

    def find_matches(self, document: str) -> set[str]:
        collected_words: set[str] = set()
        document_length: int = len(document)
        for i in range(document_length):
            level: TNode = self.root
            for j in range(i, document_length):
                char = document[j]
                if char not in level:
                    break
                level = level[char]  # type: ignore
                if self.end_symbol in level:
                    collected_words.add(document[i : j + 1])

        return collected_words

    def advanced_find_matches(
        self, document: str, variations: dict[str, str]
    ) -> set[str]:
        collected_words: set[str] = set()
        document_length: int = len(document)
        for i in range(document_length):
            level: TNode = self.root
            for j in range(i, document_length):
                char = document[j]
                if char in variations:
                    char = variations[char]
                if char not in level:
                    break
                level = level[char]  # type: ignore
                if self.end_symbol in level:
                    collected_words.add(document[i : j + 1])

        return collected_words

    def longest_common_prefix(self) -> str:
        level: TNode = self.root
        prefix: str = ""
        while True:
            keys = list(level.keys())
            if self.end_symbol in keys or len(keys) != 1:
                break
            char = keys[0]
            prefix += char
            level = level[char]  # type: ignore

        return prefix


class Orderable(Protocol):
    def __lt__(self, other: Any) -> bool: ...


U = TypeVar("U", bound=Orderable)


class Graph(Generic[U]):
    def __init__(self) -> None:
        self.graph: dict[U, set[U]] = {}

    def add_node(self, u: U) -> None:
        if u not in self.graph:
            self.graph[u] = set()

    def add_edge(self, u: U, v: U) -> None:
        if neighbors := self.graph.get(u):
            neighbors.add(v)
        else:
            self.graph[u] = {v}
        if neighbors := self.graph.get(v):
            neighbors.add(u)
        else:
            self.graph[v] = {u}

    def edge_exists(self, u: U, v: U) -> bool:
        return u in self.graph and v in self.graph[u] and u in self.graph[v]

    def adjacent_nodes(self, node: U) -> list[U]:
        return sorted(self.graph.get(node, []))

    def unconnected_vertices(self) -> list[U]:
        return [node for node, neighbors in self.graph.items() if not neighbors]

    def breadth_first_search(self, start: U) -> list[U]:
        visited = []
        visited_set = {start}
        q = Queue[U]()
        q.enqueue(start)
        while not q.is_empty():
            node = q.dequeue()
            if node is None:
                continue
            visited.append(node)
            for neighbor in self.adjacent_nodes(node):
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    q.enqueue(neighbor)
        return visited

    def depth_first_search(self, start: U) -> list[U]:
        visited = []
        visited_set = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            visited.append(node)

            for neighbor in reversed(self.adjacent_nodes(node)):
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    stack.append(neighbor)
        return visited

    def bfs_path(self, start: U, end: U) -> Optional[list[U]]:
        if start == end:
            return [start]
        parent: dict[U, Optional[U]] = {start: None}
        visited_set = {start}
        q = Queue[U]()
        q.enqueue(start)
        while not q.is_empty():
            current = q.dequeue()
            if current is None:
                continue
            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            for neighbor in self.adjacent_nodes(current):
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    parent[neighbor] = current
                    q.enqueue(neighbor)
        return None


def main():
    pass


if __name__ == "__main__":
    main()
