from typing import Generator, Generic, TypeVar


T = TypeVar("T")


class Node(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
        self.next: Node | None = None

    def __repr__(self) -> str:
        value, next = self.value, self.next.value if self.next else None
        return f"Node(value={value}, next={next})"


class LinkedList(Generic[T]):
    def __init__(self) -> None:
        self.head: Node | None = None
        self.tail: Node | None = None
        self.size: int = 0

    def __iter__(self) -> Generator[Node[T], None, None]:
        node = self.head
        while node:
            yield node.value
            node = node.next

    def __repr__(self) -> str:
        values = []
        for value in self:
            values.append(value)
        return " -> ".join(values) if values else "Empty Linked List"

    def is_empty(self) -> bool:
        return self.size == 0

    def append_head(self, value: T) -> None:
        node = Node(value)

        if self.is_empty():
            self.head = self.tail = node
        else:
            assert self.head is not None
            node.next = self.head
            self.head = node

        self.size += 1

    def append_tail(self, value: T) -> None:
        node = Node(value)

        if self.is_empty():
            self.head = self.tail = node
        else:
            assert self.tail is not None
            self.tail.next = node
            self.tail = node

        self.size += 1

    def remove_head(self) -> T | None:
        if self.is_empty():
            return None

        assert self.head is not None
        removed_value = self.head.value
        self.head = self.head.next
        if self.head is None:
            self.tail = None

        self.size -= 1
        return removed_value


def main():
    pass


if __name__ == "__main__":
    main()
