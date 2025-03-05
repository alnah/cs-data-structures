// main.go
package main

import (
	"fmt"
	"sort"
	"strings"
)

// ──────────────────────────────
// 1. Node and LinkedList (Generic)
// ──────────────────────────────

type Node[T any] struct {
	Value T
	Next  *Node[T]
}

func (n *Node[T]) String() string {
	var nextVal interface{}
	if n.Next != nil {
		nextVal = n.Next.Value
	} else {
		nextVal = nil
	}
	return fmt.Sprintf("Node(current=%v, next=%v)", n.Value, nextVal)
}

type LinkedList[T any] struct {
	Head *Node[T]
	Tail *Node[T]
	Size int
}

func (ll *LinkedList[T]) IsEmpty() bool {
	return ll.Size == 0
}

func (ll *LinkedList[T]) AppendHead(value T) {
	node := &Node[T]{Value: value}
	if ll.IsEmpty() {
		ll.Head = node
		ll.Tail = node
	} else {
		node.Next = ll.Head
		ll.Head = node
	}
	ll.Size++
}

func (ll *LinkedList[T]) AppendTail(value T) {
	node := &Node[T]{Value: value}
	if ll.IsEmpty() {
		ll.Head = node
		ll.Tail = node
	} else {
		ll.Tail.Next = node
		ll.Tail = node
	}
	ll.Size++
}

func (ll *LinkedList[T]) RemoveHead() (T, bool) {
	var zero T
	if ll.IsEmpty() {
		return zero, false
	}
	removed := ll.Head
	ll.Head = ll.Head.Next
	if ll.Head == nil {
		ll.Tail = nil
	}
	ll.Size--
	return removed.Value, true
}

func (ll *LinkedList[T]) String() string {
	if ll.IsEmpty() {
		return "Empty Linked List"
	}
	var values []string
	for node := ll.Head; node != nil; node = node.Next {
		values = append(values, fmt.Sprintf("%v", node.Value))
	}
	return strings.Join(values, " -> ")
}

// ──────────────────────────────
// 2. Stack (Generic)
// ──────────────────────────────

type Stack[T any] struct {
	ll LinkedList[T]
}

func (s *Stack[T]) IsEmpty() bool {
	return s.ll.IsEmpty()
}

func (s *Stack[T]) Push(value T) {
	s.ll.AppendHead(value)
}

func (s *Stack[T]) Pop() (T, bool) {
	return s.ll.RemoveHead()
}

func (s *Stack[T]) Peek() (T, bool) {
	var zero T
	if s.ll.Head != nil {
		return s.ll.Head.Value, true
	}
	return zero, false
}

func (s *Stack[T]) String() string {
	if s.IsEmpty() {
		return "Empty Stack"
	}
	return s.ll.String()
}

// ──────────────────────────────
// 3. Queue (Generic)
// ──────────────────────────────

type Queue[T any] struct {
	ll LinkedList[T]
}

func (q *Queue[T]) IsEmpty() bool {
	return q.ll.IsEmpty()
}

func (q *Queue[T]) Enqueue(value T) {
	q.ll.AppendTail(value)
}

func (q *Queue[T]) Dequeue() (T, bool) {
	return q.ll.RemoveHead()
}

func (q *Queue[T]) First() (T, bool) {
	var zero T
	if q.ll.Head != nil {
		return q.ll.Head.Value, true
	}
	return zero, false
}

func (q *Queue[T]) Last() (T, bool) {
	var zero T
	if q.ll.Tail != nil {
		return q.ll.Tail.Value, true
	}
	return zero, false
}

func (q *Queue[T]) String() string {
	if q.IsEmpty() {
		return "Empty Queue"
	}
	var values []string
	for node := q.ll.Head; node != nil; node = node.Next {
		values = append(values, fmt.Sprintf("%v", node.Value))
	}
	return strings.Join(values, " <- ")
}

// ──────────────────────────────
// 4. Parentheses Balance Checker
// ──────────────────────────────

func IsBalanced(input string) (bool, error) {
	if len(input) == 0 {
		return false, fmt.Errorf("input string can't be empty")
	}
	if len(input)%2 != 0 {
		return false, nil
	}

	var stack Stack[rune]
	for _, char := range input {
		if char == '(' {
			stack.Push(char)
		} else if char == ')' {
			peek, ok := stack.Peek()
			if !ok || peek != '(' {
				return false, nil
			}
			_, _ = stack.Pop()
		} else {
			// For any other character, we return false.
			return false, nil
		}
	}
	return stack.IsEmpty(), nil
}

// ──────────────────────────────
// 5. User and Comparison Methods
// ──────────────────────────────

type User struct {
	ID       int
	Username string
}

func (u User) Compare(other User) int {
	if u.ID < other.ID {
		return -1
	} else if u.ID > other.ID {
		return 1
	}
	return 0
}

func (u User) String() string {
	return fmt.Sprintf("User(id=%d, username=%q)", u.ID, u.Username)
}

// ──────────────────────────────
// 6. Binary Search Tree (BST) for Users
// ──────────────────────────────

type BSTNode struct {
	Value *User
	Left  *BSTNode
	Right *BSTNode
}

func (node *BSTNode) Min() *User {
	if node == nil || node.Value == nil {
		return nil
	}
	current := node
	for current.Left != nil {
		current = current.Left
	}
	return current.Value
}

func (node *BSTNode) Max() *User {
	if node == nil || node.Value == nil {
		return nil
	}
	current := node
	for current.Right != nil {
		current = current.Right
	}
	return current.Value
}

func (node *BSTNode) PreorderTraversal(visited *[]User) {
	if node == nil || node.Value == nil {
		return
	}
	*visited = append(*visited, *node.Value)
	if node.Left != nil {
		node.Left.PreorderTraversal(visited)
	}
	if node.Right != nil {
		node.Right.PreorderTraversal(visited)
	}
}

func (node *BSTNode) InorderTraversal(visited *[]User) {
	if node == nil || node.Value == nil {
		return
	}
	if node.Left != nil {
		node.Left.InorderTraversal(visited)
	}
	*visited = append(*visited, *node.Value)
	if node.Right != nil {
		node.Right.InorderTraversal(visited)
	}
}

func (node *BSTNode) PostorderTraversal(visited *[]User) {
	if node == nil || node.Value == nil {
		return
	}
	if node.Left != nil {
		node.Left.PostorderTraversal(visited)
	}
	if node.Right != nil {
		node.Right.PostorderTraversal(visited)
	}
	*visited = append(*visited, *node.Value)
}

func (node *BSTNode) Exists(value User) bool {
	if node == nil || node.Value == nil {
		return false
	}
	cmp := value.Compare(*node.Value)
	if cmp == 0 {
		return true
	} else if cmp < 0 {
		return node.Left.Exists(value)
	} else {
		return node.Right.Exists(value)
	}
}

func (node *BSTNode) Insert(value User) *BSTNode {
	if node == nil || node.Value == nil {
		return &BSTNode{Value: &value}
	}
	cmp := value.Compare(*node.Value)
	if cmp < 0 {
		if node.Left == nil {
			node.Left = &BSTNode{Value: &value}
		} else {
			node.Left = node.Left.Insert(value)
		}
	} else if cmp > 0 {
		if node.Right == nil {
			node.Right = &BSTNode{Value: &value}
		} else {
			node.Right = node.Right.Insert(value)
		}
	}
	return node
}

func (node *BSTNode) Delete(value User) *BSTNode {
	if node == nil || node.Value == nil {
		return nil
	}
	cmp := value.Compare(*node.Value)
	if cmp < 0 {
		node.Left = node.Left.Delete(value)
	} else if cmp > 0 {
		node.Right = node.Right.Delete(value)
	} else {
		// Node found
		if node.Left == nil {
			return node.Right
		}
		if node.Right == nil {
			return node.Left
		}
		// Node with two children: replace with successor.
		successor := node.Right
		for successor.Left != nil {
			successor = successor.Left
		}
		node.Value = successor.Value
		node.Right = node.Right.Delete(*successor.Value)
	}
	return node
}

func (node *BSTNode) SearchRange(lowerBound, upperBound User) []User {
	var visited []User
	if node == nil || node.Value == nil {
		return visited
	}
	if node.Left != nil && (*node.Value).Compare(lowerBound) > 0 {
		visited = append(visited, node.Left.SearchRange(lowerBound, upperBound)...)
	}
	if (*node.Value).Compare(lowerBound) >= 0 && (*node.Value).Compare(upperBound) <= 0 {
		visited = append(visited, *node.Value)
	}
	if node.Right != nil && (*node.Value).Compare(upperBound) < 0 {
		visited = append(visited, node.Right.SearchRange(lowerBound, upperBound)...)
	}
	return visited
}

func (node *BSTNode) Depth() int {
	if node == nil || node.Value == nil {
		return 0
	}
	leftDepth, rightDepth := 0, 0
	if node.Left != nil {
		leftDepth = node.Left.Depth()
	}
	if node.Right != nil {
		rightDepth = node.Right.Depth()
	}
	if leftDepth > rightDepth {
		return leftDepth + 1
	}
	return rightDepth + 1
}

// ──────────────────────────────
// 7. Red–Black Tree
// ──────────────────────────────

type RBNode struct {
	Red    bool
	Parent *RBNode
	Val    *User
	Left   *RBNode
	Right  *RBNode
}

func (n *RBNode) String() string {
	color := "B"
	if n.Red {
		color = "R"
	}
	return fmt.Sprintf("RBNode(%v, %s)", n.Val, color)
}

type RBTree struct {
	Nil  *RBNode
	Root *RBNode
}

func NewRBTree() *RBTree {
	nilNode := &RBNode{Red: false}
	// For convenience, the nil sentinel's children point to itself.
	nilNode.Left = nilNode
	nilNode.Right = nilNode
	return &RBTree{
		Nil:  nilNode,
		Root: nilNode,
	}
}

func (tree *RBTree) Insert(value User) {
	newNode := &RBNode{
		Val:    &value,
		Red:    true, // new nodes are red
		Left:   tree.Nil,
		Right:  tree.Nil,
		Parent: nil,
	}
	var parent *RBNode
	current := tree.Root
	for current != tree.Nil {
		parent = current
		if newNode.Val.Compare(*current.Val) < 0 {
			current = current.Left
		} else if newNode.Val.Compare(*current.Val) > 0 {
			current = current.Right
		} else {
			// Duplicate value; do nothing.
			return
		}
	}
	newNode.Parent = parent
	if parent == nil {
		tree.Root = newNode
	} else if newNode.Val.Compare(*parent.Val) < 0 {
		parent.Left = newNode
	} else {
		parent.Right = newNode
	}
	tree.fixInsert(newNode)
}

func (tree *RBTree) fixInsert(newNode *RBNode) {
	for newNode != tree.Root && newNode.Parent != nil && newNode.Parent.Red {
		parent := newNode.Parent
		grandparent := parent.Parent
		if grandparent == nil {
			break
		}
		if parent == grandparent.Right {
			uncle := grandparent.Left
			if uncle != nil && uncle.Red {
				// Case 1: Uncle is red.
				uncle.Red = false
				parent.Red = false
				grandparent.Red = true
				newNode = grandparent
			} else {
				// Cases 2 & 3: Uncle is black.
				if newNode == parent.Left {
					newNode = parent
					tree.rotateRight(newNode)
				}
				parent.Red = false
				grandparent.Red = true
				tree.rotateLeft(grandparent)
			}
		} else {
			uncle := grandparent.Right
			if uncle != nil && uncle.Red {
				uncle.Red = false
				parent.Red = false
				grandparent.Red = true
				newNode = grandparent
			} else {
				if newNode == parent.Right {
					newNode = parent
					tree.rotateLeft(newNode)
				}
				parent.Red = false
				grandparent.Red = true
				tree.rotateRight(grandparent)
			}
		}
	}
	tree.Root.Red = false
}

func (tree *RBTree) rotateLeft(pivotParent *RBNode) {
	if pivotParent == tree.Nil || pivotParent.Right == tree.Nil {
		return
	}
	pivot := pivotParent.Right
	pivotParent.Right = pivot.Left
	if pivot.Left != tree.Nil {
		pivot.Left.Parent = pivotParent
	}
	pivot.Parent = pivotParent.Parent
	if pivotParent.Parent == nil {
		tree.Root = pivot
	} else if pivotParent == pivotParent.Parent.Left {
		pivotParent.Parent.Left = pivot
	} else {
		pivotParent.Parent.Right = pivot
	}
	pivot.Left = pivotParent
	pivotParent.Parent = pivot
}

func (tree *RBTree) rotateRight(pivotParent *RBNode) {
	if pivotParent == tree.Nil || pivotParent.Left == tree.Nil {
		return
	}
	pivot := pivotParent.Left
	pivotParent.Left = pivot.Right
	if pivot.Right != tree.Nil {
		pivot.Right.Parent = pivotParent
	}
	pivot.Parent = pivotParent.Parent
	if pivotParent.Parent == nil {
		tree.Root = pivot
	} else if pivotParent == pivotParent.Parent.Right {
		pivotParent.Parent.Right = pivot
	} else {
		pivotParent.Parent.Left = pivot
	}
	pivot.Right = pivotParent
	pivotParent.Parent = pivot
}

func (tree *RBTree) Exists(value User) *RBNode {
	current := tree.Root
	for current != tree.Nil && current.Val != nil && value.Compare(*current.Val) != 0 {
		if value.Compare(*current.Val) < 0 {
			current = current.Left
		} else {
			current = current.Right
		}
	}
	return current
}

// ──────────────────────────────
// 8. HashMap (Custom with linear probing)
// Only supports string keys.
type Entry[V any] struct {
	Key   string
	Value V
}

type HashMap[V any] struct {
	Multiplier    int
	LoadThreshold float64
	ResizeFactor  int
	Data          []*Entry[V]
}

func NewHashMap[V any](size int) *HashMap[V] {
	return &HashMap[V]{
		Multiplier:    31,
		LoadThreshold: 0.7,
		ResizeFactor:  2,
		Data:          make([]*Entry[V], size),
	}
}

func (hm *HashMap[V]) keyToIndex(key string) int {
	hashValue := 0
	for _, c := range key {
		hashValue = hashValue*hm.Multiplier + int(c)
	}
	return hashValue % len(hm.Data)
}

func (hm *HashMap[V]) currentLoad() float64 {
	filled := 0
	for _, item := range hm.Data {
		if item != nil {
			filled++
		}
	}
	return float64(filled) / float64(len(hm.Data))
}

func (hm *HashMap[V]) resize() {
	oldData := hm.Data
	newSize := len(oldData) * hm.ResizeFactor
	hm.Data = make([]*Entry[V], newSize)
	for _, entry := range oldData {
		if entry != nil {
			hm.insertNoResize(entry.Key, entry.Value)
		}
	}
}

func (hm *HashMap[V]) insertNoResize(key string, value V) {
	index := hm.keyToIndex(key)
	startIndex := index
	for {
		if hm.Data[index] == nil {
			hm.Data[index] = &Entry[V]{Key: key, Value: value}
			return
		} else if hm.Data[index].Key == key {
			hm.Data[index].Value = value
			return
		}
		index = (index + 1) % len(hm.Data)
		if index == startIndex {
			panic("hashmap is full")
		}
	}
}

func (hm *HashMap[V]) Insert(key string, value V) {
	if hm.currentLoad() > hm.LoadThreshold {
		hm.resize()
	}
	hm.insertNoResize(key, value)
}

func (hm *HashMap[V]) Get(key string) (V, bool) {
	index := hm.keyToIndex(key)
	startIndex := index
	for {
		if hm.Data[index] == nil {
			var zero V
			return zero, false
		}
		if hm.Data[index].Key == key {
			return hm.Data[index].Value, true
		}
		index = (index + 1) % len(hm.Data)
		if index == startIndex {
			break
		}
	}
	var zero V
	return zero, false
}

func (hm *HashMap[V]) String() string {
	var sb strings.Builder
	for i, entry := range hm.Data {
		if entry != nil {
			sb.WriteString(fmt.Sprintf("Index %d: %v\n", i, entry))
		}
	}
	return sb.String()
}

// ──────────────────────────────
// 9. Trie
// ──────────────────────────────

type TrieNode struct {
	Children map[rune]*TrieNode
	IsEnd    bool
}

func NewTrieNode() *TrieNode {
	return &TrieNode{Children: make(map[rune]*TrieNode)}
}

type Trie struct {
	Root *TrieNode
}

func NewTrie() *Trie {
	return &Trie{Root: NewTrieNode()}
}

func (t *Trie) Add(word string) {
	node := t.Root
	for _, char := range word {
		if _, exists := node.Children[char]; !exists {
			node.Children[char] = NewTrieNode()
		}
		node = node.Children[char]
	}
	node.IsEnd = true
}

func (t *Trie) Exists(word string) bool {
	node := t.Root
	for _, char := range word {
		if _, exists := node.Children[char]; !exists {
			return false
		}
		node = node.Children[char]
	}
	return node.IsEnd
}

func (t *Trie) searchLevel(current *TrieNode, currentPrefix string, words *[]string) {
	if current.IsEnd {
		*words = append(*words, currentPrefix)
	}
	// Get sorted keys
	keys := make([]rune, 0, len(current.Children))
	for k := range current.Children {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	for _, letter := range keys {
		t.searchLevel(current.Children[letter], currentPrefix+string(letter), words)
	}
}

func (t *Trie) WordsWithPrefix(prefix string) []string {
	node := t.Root
	for _, char := range prefix {
		if _, exists := node.Children[char]; !exists {
			return []string{}
		}
		node = node.Children[char]
	}
	var words []string
	t.searchLevel(node, prefix, &words)
	return words
}

func (t *Trie) FindMatches(document string) map[string]struct{} {
	collected := make(map[string]struct{})
	runes := []rune(document)
	n := len(runes)
	for i := 0; i < n; i++ {
		node := t.Root
		for j := i; j < n; j++ {
			char := runes[j]
			if _, exists := node.Children[char]; !exists {
				break
			}
			node = node.Children[char]
			if node.IsEnd {
				collected[string(runes[i:j+1])] = struct{}{}
			}
		}
	}
	return collected
}

func (t *Trie) AdvancedFindMatches(document string, variations map[rune]rune) map[string]struct{} {
	collected := make(map[string]struct{})
	runes := []rune(document)
	n := len(runes)
	for i := 0; i < n; i++ {
		node := t.Root
		for j := i; j < n; j++ {
			char := runes[j]
			if v, ok := variations[char]; ok {
				char = v
			}
			if _, exists := node.Children[char]; !exists {
				break
			}
			node = node.Children[char]
			if node.IsEnd {
				collected[string(runes[i:j+1])] = struct{}{}
			}
		}
	}
	return collected
}

func (t *Trie) LongestCommonPrefix() string {
	prefix := ""
	node := t.Root
	for {
		if node.IsEnd || len(node.Children) != 1 {
			break
		}
		var next rune
		for k := range node.Children {
			next = k
			break
		}
		prefix += string(next)
		node = node.Children[next]
	}
	return prefix
}

// ──────────────────────────────
// 10. Graph (Generic, for ordered types)
// ──────────────────────────────

type Ordered interface {
	comparable
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64 | ~string
}

type Graph[T Ordered] struct {
	Graph map[T]map[T]bool
}

func NewGraph[T Ordered]() *Graph[T] {
	return &Graph[T]{Graph: make(map[T]map[T]bool)}
}

func (g *Graph[T]) AddNode(u T) {
	if _, exists := g.Graph[u]; !exists {
		g.Graph[u] = make(map[T]bool)
	}
}

func (g *Graph[T]) AddEdge(u, v T) {
	if neighbors, exists := g.Graph[u]; exists {
		neighbors[v] = true
	} else {
		g.Graph[u] = map[T]bool{v: true}
	}
	if neighbors, exists := g.Graph[v]; exists {
		neighbors[u] = true
	} else {
		g.Graph[v] = map[T]bool{u: true}
	}
}

func (g *Graph[T]) EdgeExists(u, v T) bool {
	if neighbors, exists := g.Graph[u]; exists {
		if _, ok := neighbors[v]; ok {
			if neighbors2, exists2 := g.Graph[v]; exists2 {
				_, ok2 := neighbors2[u]
				return ok2
			}
		}
	}
	return false
}

func (g *Graph[T]) AdjacentNodes(node T) []T {
	var nodes []T
	if neighbors, exists := g.Graph[node]; exists {
		for neighbor := range neighbors {
			nodes = append(nodes, neighbor)
		}
	}
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i] < nodes[j]
	})
	return nodes
}

func (g *Graph[T]) UnconnectedVertices() []T {
	var vertices []T
	for node, neighbors := range g.Graph {
		if len(neighbors) == 0 {
			vertices = append(vertices, node)
		}
	}
	return vertices
}

func (g *Graph[T]) BreadthFirstSearch(start T) []T {
	var visited []T
	visitedSet := make(map[T]bool)
	var q Queue[T]
	visitedSet[start] = true
	q.Enqueue(start)
	for !q.IsEmpty() {
		node, ok := q.Dequeue()
		if !ok {
			continue
		}
		visited = append(visited, node)
		for _, neighbor := range g.AdjacentNodes(node) {
			if !visitedSet[neighbor] {
				visitedSet[neighbor] = true
				q.Enqueue(neighbor)
			}
		}
	}
	return visited
}

func (g *Graph[T]) DepthFirstSearch(start T) []T {
	var visited []T
	visitedSet := make(map[T]bool)
	stack := []T{start}
	visitedSet[start] = true
	for len(stack) > 0 {
		// Pop from stack
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		visited = append(visited, node)
		neighbors := g.AdjacentNodes(node)
		// Iterate in reverse order to mimic recursive DFS
		for i := len(neighbors) - 1; i >= 0; i-- {
			neighbor := neighbors[i]
			if !visitedSet[neighbor] {
				visitedSet[neighbor] = true
				stack = append(stack, neighbor)
			}
		}
	}
	return visited
}

func (g *Graph[T]) BFSPath(start, end T) []T {
	if start == end {
		return []T{start}
	}
	parent := make(map[T]T)
	visitedSet := make(map[T]bool)
	var q Queue[T]
	visitedSet[start] = true
	q.Enqueue(start)
	found := false
	var current T
	for !q.IsEmpty() && !found {
		current, _ = q.Dequeue()
		if current == end {
			found = true
			break
		}
		for _, neighbor := range g.AdjacentNodes(current) {
			if !visitedSet[neighbor] {
				visitedSet[neighbor] = true
				parent[neighbor] = current
				q.Enqueue(neighbor)
			}
		}
	}
	if !found {
		return nil
	}
	var path []T
	for node := end; ; {
		path = append([]T{node}, path...)
		p, ok := parent[node]
		if !ok {
			break
		}
		node = p
	}
	return path
}
