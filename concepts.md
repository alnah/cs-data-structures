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
However, if the tree is unbalanced due to partially, or worst, totally sorted data, BST methods will have a O(n) complexity equal to their depth.

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

# Red-Black Tree - RBTree and RBNode

A red-black tree is a kind of binary search tree that solves the "balancing" problem.
It contains a bit of extra logic to ensure that as nodes are inserted and deleted, the tree remains relatively balanced.

- Each node is either red or black
- The root is black
- All nil leaf nodes are black
- If a node is red, then both its children are black
- All paths from a single node go through the same number of black nodes to reach any of its descendant nil nodes

The re-balancing of a red-black tree does not result in a perfectly balanced tree.
However, its insertion and deletion operations, along with the tree rearrangement and recoloring, are always performed in O(log(n)) time.

# Hashmap

A hash map (or hash table) is a data structure that maps keys to values, offering very fast average-case performance for lookups, insertions, and deletions—typically O(1).
Hash maps use an array to store key-value pairs.
Each key is mapped to an index in the array via a hash function.

- The hash function processes the key (e.g., by combining the Unicode values of its characters) to produce a numerical hash value. This hash value is then converted to an array index (often using the modulus operator). For example, a simple hash function that sums Unicode values might produce the same result for "ab" and "ba", hence requiring collision handling.
- Since different keys can hash to the same index, hash maps must handle collisions. If a collision occurs, the algorithm searches sequentially through the array to find an empty slot.
- When a hash map becomes too full, its performance degrades due to increased collisions. To address this, hash maps are resized—typically by allocating a larger array (often double the current size) and rehashing all existing elements. Although resizing is an O(n) operation, it happens infrequently enough that the amortized cost per operation remains O(1).
- Iterating over a hash map involves traversing the underlying array. This operation is O(n) because every bucket in the array must be examined, even if many buckets are empty.
