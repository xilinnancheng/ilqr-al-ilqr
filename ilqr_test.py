from turtle import color
import ReedsSheppPathPlanning as rs
from ilqr import ILQRSolver, ILQRSolverParameter
import jax.numpy as np
import math
from matplotlib import pyplot as plt
from functools import partial


def CalculateCurvature(xi_minus_1, xi, xi_plus_1, yi_minus_1, yi, yi_plus_1):
    # Menger curvature
    tri_area = 0.5 * ((xi - xi_minus_1) * (yi_plus_1 - yi_minus_1) -
                      (yi - yi_minus_1) * (xi_plus_1 - xi_minus_1))
    if abs(tri_area) < 1e-8:
        return 0.0

    kappa = (4.0 * tri_area) / (math.hypot(xi_minus_1 - xi, yi_minus_1 - yi)
                                * math.hypot(xi - xi_plus_1, yi - yi_plus_1)
                                * math.hypot(xi_plus_1 - xi_minus_1, yi_plus_1 - yi_minus_1))
    return kappa


def CalculatePathCurvature(x, y):
    path_curvature = []
    for i in range(1, len(x) - 1):
        path_curvature.append(CalculateCurvature(
            x[i-1], x[i], x[i+1], y[i-1], y[i], y[i+1]))
    return path_curvature


def GetRsPathCost(path):
    path_cost = 0.0
    for length in path.lengths:
        if length >= 0.0:
            path_cost += length
        else:
            path_cost += abs(length)
    return path_cost


def StateCostFunction(x, u, index, ref_path_x, ref_path_y, ref_path_yaw, w_ref, w_delta, w_acc):
    angle_diff = x[2] - ref_path_yaw[index]
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi

    stage_cost = w_ref * pow(x[0] - ref_path_x[index], 2) + \
        w_ref * pow(x[1] - ref_path_y[index], 2) + \
        w_ref * pow(angle_diff, 2) + \
        w_delta * pow(u[0], 2) + w_acc * pow(u[1], 2)
    return stage_cost


def TerminalCostFunction(x, ref_path_x, ref_path_y, ref_path_yaw, w_terminal):
    angle_diff = x[2] - ref_path_yaw[-1]
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    terminal_cost = w_terminal * pow(x[0] - ref_path_x[-1], 2) + \
        w_terminal * pow(x[1] - ref_path_y[-1], 2) + \
        w_terminal * pow(angle_diff, 2)
    return terminal_cost


def BicycleModel(x_curr, u_curr, delta_t):
    wheel_base = 2.84
    x_next = []

    x_next.append(x_curr[0] + x_curr[3] * np.cos(x_curr[2]) * delta_t)
    x_next.append(x_curr[1] + x_curr[3] * np.sin(x_curr[2]) * delta_t)
    x_next.append(x_curr[2] + x_curr[3] /
                  wheel_base * np.tan(u_curr[0]) * delta_t)
    x_next.append(x_curr[3] + u_curr[1] * delta_t)
    return np.array(x_next)


def BicycleModeldx(x_curr, u_curr, delta_t):
    wheel_base = 2.84
    theta_k = x_curr[2]
    v_k = x_curr[3]

    delta_k = u_curr[0]
    return np.array([[1, 0, -v_k * np.sin(theta_k) * delta_t, np.cos(theta_k) * delta_t],
                     [0, 1, v_k * np.cos(theta_k) * delta_t,
                      np.sin(theta_k) * delta_t],
                     [0, 0, 1, np.tan(delta_k) / wheel_base * delta_t],
                     [0, 0, 0, 1]])


def BicycleModeldu(x_curr, u_curr, delta_t):
    wheel_base = 2.84
    v_k = x_curr[3]
    delta_k = u_curr[0]
    return np.array([[0, 0],
                     [0, 0],
                     [0, v_k / (wheel_base * pow(np.cos(delta_k), 2))
                      * delta_t],
                     [0, delta_t]])


def main():
    print("ILQR Smoother Demo")
    start_pose = [30, 10, np.deg2rad(0.0)]
    goal_pose = [40, 7, np.deg2rad(0.0)]
    wheel_base = 2.84
    max_steer = 0.5
    step_size = 0.5
    delta_t = 1
    only_show_opt_result = True

    max_curvature = math.tan(max_steer) / wheel_base
    rs_paths = rs.calc_paths(start_pose[0], start_pose[1], start_pose[2], goal_pose[0],
                             goal_pose[1], goal_pose[2], max_curvature, step_size)

    best_rs_path = None
    best_rs_cost = None
    for path in rs_paths:
        cost = GetRsPathCost(path)
        if not best_rs_cost or cost < best_rs_cost:
            best_rs_cost = cost
            best_rs_path = path

    plt.figure(num=1)
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.arrow(x=start_pose[0], y=start_pose[1], dx=2.0 * math.cos(
        start_pose[2]), dy=2.0 * math.sin(start_pose[2]), width=.08, color='b')
    plt.arrow(x=goal_pose[0], y=goal_pose[1], dx=2.0 * math.cos(
        goal_pose[2]), dy=2.0 * math.sin(goal_pose[2]), width=.08, color='r')
    plt.title("Side Parking Scenario")

    w_ref = 1
    w_terminal = 100
    w_delta = 1
    w_acc = 1
    state_cost_func = partial(
        StateCostFunction, ref_path_x=best_rs_path.x, ref_path_y=best_rs_path.y, ref_path_yaw=best_rs_path.yaw, w_ref=w_ref, w_delta=w_delta, w_acc=w_acc)
    terminal_cost_func = partial(
        TerminalCostFunction, ref_path_x=best_rs_path.x, ref_path_y=best_rs_path.y, ref_path_yaw=best_rs_path.yaw, w_terminal=w_terminal)

    ilqr_solver_param = ILQRSolverParameter(1e-3, 50, 1e-4, 10, 0.5, 1e-4)
    ilqr_solver = ILQRSolver(BicycleModel, state_cost_func, terminal_cost_func, len(
        best_rs_path.x) - 1, delta_t, ilqr_solver_param)

    x0 = np.array([30, 10, np.deg2rad(0.0), step_size/delta_t])
    u0 = [np.array([0.0, 0.0])] * (len(best_rs_path.x) - 1)
    res = ilqr_solver.Solve(x0, u0)

    for i in range(len(best_rs_path.x)):
        plt.figure(num=1)
        plt.plot(best_rs_path.x[i],
                 best_rs_path.y[i], marker="o", color='g')
        plt.grid()

    plt.figure(num=2)
    plt.plot(CalculatePathCurvature(
        best_rs_path.x, best_rs_path.y))
    plt.title("Path Curvature")

    for i in range(len(res['x_hist'])):
        x = res['x_hist'][i]
        u = res['u_hist'][i]
        x_list = []
        y_list = []
        yaw_list = []
        v_list = []
        delta_list = []
        acc_list = []
        for j in range(len(x)):
            x_list.append(x[j][0])
            y_list.append(x[j][1])
            yaw_list.append(x[j][2])
            v_list.append(x[j][3])

        for j in range(len(u)):
            delta_list.append(u[j][0])
            acc_list.append(u[j][1])

        if (only_show_opt_result and i == len(res['x_hist']) - 1) or not only_show_opt_result:
            plt.figure(num=1)
            plt.plot(x_list,
                     y_list, marker="o")
            plt.figure(num=2)
            plt.plot(CalculatePathCurvature(
                x_list, y_list))
            plt.grid()

            plt.figure(num=3)
            plt.subplot(411)
            plt.plot(yaw_list)
            plt.grid()
            plt.title("Yaw")

            plt.subplot(412)
            plt.plot(v_list)
            plt.grid()
            plt.title("Velocity")

            plt.subplot(413)
            plt.plot(delta_list)
            plt.grid()
            plt.title("Delta")

            plt.subplot(414)
            plt.plot(acc_list)
            plt.grid()
            plt.title("Acceleration")
    plt.show()


if __name__ == "__main__":
    main()
