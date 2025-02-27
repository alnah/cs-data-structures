from typing import Generator, Generic, Optional, TypeVar


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


def main():
    pass


if __name__ == "__main__":
    main()
