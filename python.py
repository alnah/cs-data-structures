from typing import Generator, Generic, TypeVar


T = TypeVar("T")


class Node(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
        self.next: Node[T] | None = None

    def __repr__(self) -> str:
        return str(self.value)


class LinkedList(Generic[T]):
    def __init__(self) -> None:
        self.head: Node[T] | None = None
        self.tail: Node[T] | None = None
        self.__size = 0

    def __iter__(self) -> Generator[Node[T], None, None]:
        node = self.head
        while node:
            yield node
            node = node.next

    def __repr__(self) -> str:
        return " -> ".join([str(node.value) for node in self])

    def __increment(self) -> None:
        self.__size += 1

    def __decrement(self) -> None:
        self.__size -= 1

    def is_empty(self) -> bool:
        return self.__size == 0

    def size(self) -> int:
        return self.__size

    def append_head(self, value: T) -> None:
        node = Node(value)
        if self.is_empty():
            self.head, self.tail = node, node
        elif self.head:
            node.next = self.head
            self.head = node
        self.__increment()

    def append_tail(self, value: T) -> None:
        node = Node(value)
        node.next = None
        if self.is_empty():
            self.head, self.tail = node, node
        elif self.tail:
            self.tail.next = node
            self.tail = node
        self.__increment()

    def remove_head(self) -> T | None:
        temp = None
        if self.is_empty():
            return None
        elif self.head:
            temp = self.head.value
            self.head = self.head.next
        if not self.head:
            self.tail = None
        self.__decrement()
        return temp


class Stack(Generic[T]):
    def __init__(self) -> None:
        self.ll = LinkedList()

    def __repr__(self) -> str:
        return repr(self.ll)

    def is_empty(self) -> bool:
        return self.ll.is_empty()

    def push(self, value: T) -> None:
        self.ll.append_head(value)

    def pop(self) -> T | None:
        return self.ll.remove_head()

    def peek(self) -> T | None:
        return None if not self.ll.head else self.ll.head.value

    def size(self) -> int:
        return self.ll.size()


class Queue(Generic[T]):
    def __init__(self) -> None:
        self.ll = LinkedList()

    def __repr__(self) -> str:
        return " <- ".join([node.value for node in self.ll])

    def is_empty(self) -> bool:
        return self.ll.is_empty()

    def enqueue(self, value: T) -> None:
        self.ll.append_tail(value)

    def dequeue(self) -> T | None:
        return self.ll.remove_head()

    def size(self) -> int:
        return self.ll.size()


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

    if stack.size() > 1:
        return False

    return True


def main():
    pass

if __name__ == "__main__":
    main()
