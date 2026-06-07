# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指导。

## 项目概述

本项目实现了 **iLQR**（迭代线性二次型调节器）和 **AL-iLQR**（增广拉格朗日 iLQR），用于类车机器人的轨迹优化。它以粗略的 Reeds-Shepp 参考路径为输入，将其平滑为满足控制约束（转向角和加速度限制）的运动学可行轨迹。演示场景为侧方停车。

## 常用命令

- **运行演示**：`python ilqr_test.py`
- 依赖（无 requirements.txt，需手动安装）：`pip install jax jaxlib numpy matplotlib`

## 架构

三层设计，所有文件均在仓库根目录：

1. **路径生成** — `ReedsSheppPathPlanning.py`：生成从起点到目标位姿的运动学可行参考路径（涵盖 48 种标准 Reeds-Shepp 曲线类型）。也可独立运行（自带 `main()`）。

2. **轨迹优化** — 两个递进式求解器：
   - `ilqr.py`：无约束 iLQR，通过 DDP 风格的后向传递（Riccati 扫描）和前向传递（带线搜索）实现。使用 JAX 的 `grad` / `jacfwd` 对动力学和代价函数进行自动微分。
   - `al_ilqr.py`：带约束的轨迹优化。在外层增广拉格朗日循环中封装 iLQR，用于处理不等式约束（控制量限制）。每次外层迭代更新拉格朗日乘子 `mu` 和惩罚因子 `sigma`，内层循环通过 iLQR 求解无约束的 AL 子问题。

3. **演示脚本** — `ilqr_test.py`：定义自行车模型（Euler 和 RK3 积分器）、二次轨迹/终端跟踪代价、控制不等式约束，运行两种求解器，并用 matplotlib 绘制路径/曲率/状态图。

## 关键细节

- 两个求解器均使用 JAX（`jax.numpy`、`jax.grad`、`jax.jacfwd`）——所有代价和动力学函数必须可被 JAX 微分。
- 自行车模型参数：轴距 = 2.84 m，最大转向角 = 0.5 rad，最大加速度 = 1.0 m/s²。
- 可配置的积分方法（Euler 或 RK3）和参考路径离散化分辨率（如 0.5 m、0.1 m）。
- 两个 `Solve` 方法中的线搜索存在潜在死循环问题：当 `alpha < 1e-6` 时，会打印警告但不会跳出 while 循环。
