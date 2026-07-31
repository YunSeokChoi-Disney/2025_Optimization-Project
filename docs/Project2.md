# Project 2 — Combinatorial Optimization: The Knapsack Problem

**Course:** Introduction to Optimization (최적화개론), Hanyang University
**Focus:** Dynamic Programming and Branch & Bound

> Solve the **0/1 Knapsack Problem** — a classic combinatorial optimization problem — with several
> algorithmic strategies and compare their optimality and efficiency.

---

## Objective

Find the item selection that maximizes value under a weight budget, then compare an **exact** solution
against **heuristic** and **relaxation** approaches in terms of accuracy and time complexity.

## Key Algorithms

- **Dynamic Programming (DP)** — computes the exact optimal cost and the specific item selection.
- **Branch and Bound (B&B)** — tree-search that prunes branches to improve efficiency over exhaustive
  search.
- **Greedy Algorithm** — value/weight heuristic, compared against the exact optimum.
- **LP Relaxation** — relaxes the 0/1 constraint to a linear program; a round-down of the fractional
  solution gives a feasible integer bound.

## Outcome

Verified the optimal knapsack strategy and ran a **cost comparison across algorithms** to analyze the
trade-off between solution quality and running time.

---

## Files

> These live on the `Project2` branch.

| File | Description |
|------|-------------|
| `DP_YUNSEOK.py` | Dynamic Programming solution (exact) |
| `Branch and Bound_YUN SEOK.py` | Branch & Bound solution (Python) |
| `Greedy_Knapsack_YUNSEOK.py` | Greedy heuristic |
| `LP_Relaxation_YUNSEOK.py` | LP relaxation |
| `LP_Round_Down_Value_YUN SEOK.py` | Round-down of the LP-relaxed solution |
| `branch_bound_matlab.m`, `Branch&Bound_CHOI_YUN_SEOK.m` | Branch & Bound (MATLAB) |
| `knapsack_linprog_YUNSEOK.m` | Knapsack via `linprog` (MATLAB) |
| `Project_2_Opt_Undergrad-1 (2).pdf` | Assignment slides |
| `그룹_5_발표자료.pdf` | Presentation slides |
| `그룹_5_답안.pdf` | Written solution |

---

> See [Project 1](Project1.md) and [Project 3](Project3.md) for the other projects in this repository.
