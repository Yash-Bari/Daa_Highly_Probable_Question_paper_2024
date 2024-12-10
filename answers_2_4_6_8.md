[Previous content remains the same...]

### Q2) Alternative to Q1
#### a) Greedy Strategy [12 marks]

Principle:
- Makes locally optimal choice at each step
- Hopes these choices lead to globally optimal solution
- Never reconsiders choices once made

Control Abstraction:
```
Algorithm Greedy(a, n)
{
    solution = ∅
    for i = 1 to n do {
        x = select(a)
        if feasible(solution ∪ {x}) then
            solution = solution ∪ {x}
    }
    return solution
}
```

Time Analysis:
- Generally O(n) or O(n log n) depending on selection process
- Selection usually requires sorting: O(n log n)
- Feasibility check: O(1) to O(n)

Example: Fractional Knapsack
```
Algorithm FractionalKnapsack(W, items[]):
    1. Calculate value/weight ratio for each item
    2. Sort items by ratio (descending)
    3. For each item:
        if (can take whole item)
            take it
        else
            take fraction to fill knapsack
```

#### b) Job Sequencing Algorithm [6 marks]

Detailed Steps:
1. Sort jobs in decreasing order of profit
2. Find maximum deadline (max_deadline)
3. Create array slot[max_deadline] initialized with -1
4. For each job j in sorted order:
   - Find latest empty slot k before deadline[j]
   - If found, assign job j to slot k

```
Algorithm JobSequencing(jobs[], n):
    Sort jobs by profit in descending order
    max_deadline = max(deadline[1...n])
    slot[max_deadline] = {-1}
    result = []
    
    for i = 1 to n:
        for j = min(deadline[i], max_deadline) down to 1:
            if slot[j] == -1:
                slot[j] = i
                result.append(jobs[i])
                break
    
    return result
```

### Q4) Alternative to Q3
#### a) Branch and Bound for 0/1 Knapsack [9 marks]

For given instance:
Item:     A   B   C   D
Profit:   18  10  12  10
Weight:    9   4   6   2
Capacity: 15kg

Algorithm:
```
class Node:
    level, profit, bound, weight

Algorithm KnapsackBB():
    Q = empty priority queue
    v = new Node(level = 0, profit = 0, weight = 0)
    maxProfit = 0
    
    while Q not empty:
        v = Q.extractMax()
        u = new Node()
        
        // Include current item
        u.level = v.level + 1
        u.weight = v.weight + weight[u.level]
        u.profit = v.profit + profit[u.level]
        
        if u.weight <= W and u.profit > maxProfit:
            maxProfit = u.profit
            
        u.bound = bound(u)
        if u.bound > maxProfit:
            Q.insert(u)
            
        // Exclude current item
        u.weight = v.weight
        u.profit = v.profit
        u.bound = bound(u)
        if u.bound > maxProfit:
            Q.insert(u)
```

Solution:
- Optimal selection: Items B and C
- Maximum profit = Rs. 22
- Total weight = 10 kg

#### b) Sum of Subset Problem using Backtracking [8 marks]

Given:
- set[] = {2, 3, 5, 6, 8, 10}
- sum = 10

Solution:
```
Algorithm SumOfSubset(set[], sum):
    subset[] = new boolean[set.length]
    
    void findSubset(int index, int currentSum):
        if currentSum == sum:
            print subset[]
            return
            
        if index >= set.length || currentSum > sum:
            return
            
        // Include current element
        subset[index] = true
        findSubset(index + 1, currentSum + set[index])
        
        // Exclude current element
        subset[index] = false
        findSubset(index + 1, currentSum)
```

Solutions found:
1. {2, 3, 5}
2. {2, 8}
3. {10}

### Q6) Alternative to Q5
#### a) Randomized and Approximation Algorithms [10 marks]

Randomized Algorithms:
- Use random numbers to guide algorithm execution
- Two types:
  1. Las Vegas: Always correct, runtime varies
  2. Monte Carlo: Fixed runtime, may be incorrect

Example: QuickSort with random pivot
```
RandomizedQuickSort(A, p, r):
    if p < r:
        q = RandomizedPartition(A, p, r)
        RandomizedQuickSort(A, p, q-1)
        RandomizedQuickSort(A, q+1, r)

RandomizedPartition(A, p, r):
    i = Random(p, r)
    swap A[i] with A[r]
    return Partition(A, p, r)
```

Approximation Algorithms:
- Find approximate solutions to optimization problems
- Guarantee solution within certain factor of optimal
- Used when exact solution is NP-hard

Example: Vertex Cover Approximation
```
ApproxVertexCover(G):
    C = ∅
    E' = G.E
    while E' not empty:
        Let (u,v) be an edge in E'
        Add u and v to C
        Remove all edges incident on u or v from E'
    return C
```

#### b) Embedded Algorithms [7 marks]

Special needs of Embedded Algorithms:
1. Memory constraints
2. Real-time requirements
3. Power efficiency
4. Predictable performance
5. No dynamic memory allocation

Best Sorting Algorithm for Embedded Systems:
- Insertion Sort for small datasets
  - Small code size
  - In-place sorting
  - Predictable performance
  - Good cache utilization
  
Reasons:
1. Constant memory usage
2. Simple implementation
3. Good performance on nearly sorted data
4. Stable sorting
5. Predictable worst-case behavior

### Q8) Alternative to Q7
#### a) Multi-threaded Merge Sort [9 marks]

```
Algorithm ParallelMergeSort(A, low, high):
    if low < high:
        if (high - low) <= THRESHOLD:
            SequentialMergeSort(A, low, high)
        else:
            mid = (low + high)/2
            #pragma omp parallel sections
            {
                #pragma omp section
                ParallelMergeSort(A, low, mid)
                
                #pragma omp section
                ParallelMergeSort(A, mid+1, high)
            }
            ParallelMerge(A, low, mid, high)
```

Parallel Merging Advantage:
1. Division of work:
   - Multiple threads process different array sections
   - Balanced workload distribution

2. Speedup analysis:
   - T(n) = T(n/2) + O(n) for sequential
   - T(n) = T(n/2)/p + O(n/p) for parallel
   - Theoretical speedup: O(p) where p = number of processors

3. Implementation considerations:
   - Thread creation overhead
   - Memory access patterns
   - Cache utilization
   - Load balancing

#### b) Rabin-Karp Algorithm Analysis [8 marks]

Given:
- q = 11 (working module)
- T = 31415926535
- P = 26

1. Hash Function:
   h(k) = k mod q

2. Pattern hash:
   P = "26"
   hash(P) = 26 mod 11 = 4

3. Text window calculations:
   Initial window: "31" = 31 mod 11 = 9
   Next window: "14" = ((9 × 10) + 4) mod 11 = 5
   Continue...

Number of spurious hits:
- When hash matches but actual pattern doesn't
- For this case: 2 spurious hits
  (Due to modulo collisions)

Matching process:
1. Calculate pattern hash: 4
2. Slide window through text:
   - Window "31": hash = 9 (no match)
   - Window "14": hash = 5 (no match)
   - Window "41": hash = 4 (spurious hit)
   - Window "15": hash = 5 (no match)
   - Window "59": hash = 4 (spurious hit)
   - Window "92": hash = 3 (no match)
   - Window "26": hash = 4 (true match)
   - Window "65": hash = 10 (no match)
   - Window "53": hash = 9 (no match)
   - Window "35": hash = 2 (no match)
