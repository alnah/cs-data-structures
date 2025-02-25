# What are data structures ?

Organizational tools that allow for more advanced algorithms:

- **Linked Lists**: A chain of nodes, efficient for inserts and deletes.
- **Stacks**: Last in, first out.
- **Queues**: First in, first out.
- **Binary Trees**: A tree where each node has up to two children.
- **Red Black Trees**: A self-balancing binary tree using colors.
- **Hashmaps**: A data structure that maps keys to values.
- **Tries**: A tree used for storing and searching words efficiently.
- **Graphs**: A collection of nodes connected by edges.

Data structure:

- Stores data
- Organizes data so that it can easily be accessed and modified
- Contains algorithmic functions to expose the ability to read and modify the data

# Linked List

A linked list is a linear data structure where elements are not stored next to each other in memory.
The elements in a linked list are linked using references to each other.
Why a linked list instead of an array?

- An array is O(n) time complexity, index based, and consumes less memory.
- A linked list is O(1) time complexity, not index based, consumes more memory.
- Not index based means it's the right tool to work with what's in the middle of the data structure.

Linked lists are used as a base for other data structures:

- Stack
- Queue

# Stack - Last In First Out - LIFO

A stack is a data structure that stores ordered items.
It's like a list, but its design is more restrictive.
It only allows items to be added or removed from the top of the stack.
Stacks are often used in the real world for:

- Function call management
- Undo/redo functionality
- Expression evaluation
- Browser history

# Queue - First In FIrst Out - FIFO

A queue is a data structure that stores ordered items.
It's like a list, but again, like a stack, its design is more restrictive.
A queue only allows items to be added to the tail of the queue and removed from the head of the queue.
Queues are often used for:

- Task schedulling
- Data streamling and buffers
- BFS traversal in graphs and trees to explore nodes level by level
- Traffic management
- Ticketing systems
- Email and message processing
- Caching and memory manamgent
- Real-time data processing
