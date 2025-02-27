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

# Binary Search Tree - BSTNode

Trees are data structures where each node contains a value and holds references to its child nodes. In a generic tree:

- Each node has a value and a list of children
- Every child node has exactly one parent

A binary search tree (BST) is a special type of tree where each node has at most two children, and it adheres to the following rules:

- Each node can have at most two children, typically called the left and right child
- The value of the left child must be less than its parent's value
- The value of the right child must be greater than its parent's value
- No two nodes in the BST have the same value

BST methods should have a complexity of O(log(n)) for lookups, deletions and insertions.

## Naming convention

### Predecessor

Node with the next lower value.

- If the node has a left subtree, the predecessor is the maximum value node within that left subtree.
- If there is no left subtree, the predecessor is typically found by moving up the tree to find an ancestor node whose value is less than the current node's value.

### Successor

Node with the next higher value.

- If the node has a right subtree, the successor is the minimum value node in that right subtree.
- If there is no right subtree, the successor is determined by traversing up the tree to locate an ancestor whose value is greater than the current node's value.

### Tree Traversals

Methods for visiting all the nodes in a tree in a specific order.

#### Preorder

Visit the root first, then traverse the left subtree, and finally the right subtree.

- ex. For creating a copy of the tree.
- ex. To get a prefix expression on an expression tree.

#### Postorder

Traverse the left subtree first, then the right subtree, and visit the root node last.

- ex. For deleting trees (freeing memory) or evaluating postfix expressions.

#### Inorder

Traverse the left subtree first, visit the root, and then traverse the right subtree.

- ex. For printing the tree's contents in order.
