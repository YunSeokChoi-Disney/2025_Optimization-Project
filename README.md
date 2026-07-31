# 2025 Introduction to Optimization — Projects

Coursework for **Introduction to Optimization (최적화개론)**, Hanyang University.

This repository collects two projects, kept side by side. Click through for the full write-up of each.

| Project | Topic | Details |
|---------|-------|---------|
| **Project 1** | Numerical optimization algorithms — Gradient Descent, Newton's Method, LP (Simplex), QP | [📄 docs/Project1.md](docs/Project1.md) |
| **Project 3** | Reinforcement learning in a 6×6 grid maze — Q-Learning, SARSA, DQN | [📄 docs/Project3.md](docs/Project3.md) |

---

## Repository layout

```
.
├── README.md                     ← you are here (index)
├── docs/
│   ├── Project1.md               ← Project 1 write-up
│   └── Project3.md               ← Project 3 write-up (with flowcharts)
│
├── PJ1_Problem1_(4).py           ┐
├── PJ1_Problem1_(5).py           ├─ Project 1 code
├── PJ1_Problem1_(6).py           ┘
│
├── problem1(maze_environment)…py ┐
├── problem2(q_learning,_sarsa)…py├─ Project 3 code
├── problem3(dqn)…py               ┘
│
├── Qlearing_maze.mp4             ┐
├── SARSA_maze.mp4                ├─ Project 3 result videos
├── DQN_maze.mp4                  ┘
│
├── Project_3_2025_Opt_Undergrad.pdf                    ← assignment slides
├── [최적화개론 Project3] 발표자료_3조_….pdf              ← team presentation
└── [최적화개론 Project3] 솔루션_3조_….pdf                ← written solution
```

---

## Projects at a glance

### Project 1 — Numerical Optimization Algorithms
Implement fundamental optimization algorithms **from scratch** (no pre-built solvers): unconstrained
optimization (Gradient Descent, Newton's Method), Linear Programming (Simplex), and Quadratic
Programming. → [Read more](docs/Project1.md)

### Project 3 — Reinforcement Learning in a Grid Maze
Train an agent to find the **shortest path** through a 6×6 [MiniGrid](https://minigrid.farama.org/)
maze with tabular RL (Q-Learning, SARSA) and deep RL (DQN with CNN, replay buffer, and target
network). Includes flowcharts of the agent loop and the DQN training loop.
→ [Read more](docs/Project3.md)

*Project 3 team: 3조 — 최윤석, 김영민*
