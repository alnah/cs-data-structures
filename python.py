from typing import Generator, Generic, TypeVar


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


def main():
    pass


if __name__ == "__main__":
    main()
