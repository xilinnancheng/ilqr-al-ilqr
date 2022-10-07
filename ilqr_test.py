import ReedsSheppPathPlanning as rs
import jax.numpy as np
import math
from matplotlib import pyplot as plt
from functools import partial
from ilqr import ILQRSolver, ILQRSolverParameter
from al_ilqr import ALILQRSolver, ALILQRSolverParameter


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


def InequalityConstraintCostFunction(u, mu, sigma, max_delta, max_acc):
    return 0.5 * sigma * (pow(max(mu[0]/sigma + (u[0] - max_delta), 0), 2) - pow(mu[0]/sigma, 2) +
                          pow(max(mu[1]/sigma + (-u[0] - max_delta), 0), 2) - pow(mu[1]/sigma, 2) +
                          pow(max(mu[2]/sigma + (u[1] - max_acc), 0), 2) - pow(mu[2]/sigma, 2) +
                          pow(max(mu[3]/sigma + (-u[1] - max_acc), 0), 2) - pow(mu[3]/sigma, 2))


def InequalityConstraintFunction(u, mu, sigma, max_delta, max_acc):
    return np.array([max(u[0] - max_delta + mu[0]/sigma, 0.0), max(-u[0] - max_delta + mu[1]/sigma, 0.0), max(u[1] - max_acc + mu[2]/sigma, 0.0), max(-u[1] - max_acc + mu[3]/sigma, 0.0)])


def BicycleModel(x_curr, u_curr):
    wheel_base = 2.84
    x_dot = []
    x_dot.append(x_curr[3] * np.cos(x_curr[2]))
    x_dot.append(x_curr[3] * np.sin(x_curr[2]))
    x_dot.append(x_curr[3] /
                 wheel_base * np.tan(u_curr[0]))
    x_dot.append(u_curr[1])
    return np.array(x_dot)


def BicycleModelEulerMethod(x_curr, u_curr, delta_t):
    wheel_base = 2.84
    x_next = []

    x_next.append(x_curr[0] + x_curr[3] * np.cos(x_curr[2]) * delta_t)
    x_next.append(x_curr[1] + x_curr[3] * np.sin(x_curr[2]) * delta_t)
    x_next.append(x_curr[2] + x_curr[3] /
                  wheel_base * np.tan(u_curr[0]) * delta_t)
    x_next.append(x_curr[3] + u_curr[1] * delta_t)
    return np.array(x_next)


def BicycleModelRK3(x_curr, u_curr, delta_t):
    k1 = BicycleModel(x_curr, u_curr) * delta_t
    k2 = BicycleModel(x_curr + 0.5 * k1, u_curr) * delta_t
    k3 = BicycleModel(x_curr - k1 + 2 * k2, u_curr) * delta_t
    return x_curr + (k1 + 4 * k2 + k3)/6


def main():
    print("ILQR Smoother Demo")
    start_pose = [30, 10, np.deg2rad(0.0)]
    goal_pose = [40, 7, np.deg2rad(0.0)]
    wheel_base = 2.84
    max_steer = 0.5
    max_acc = 1.0
    step_size = 0.5
    delta_t = 1
    max_curvature = math.tan(max_steer) / wheel_base

    # Rs path
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

    # Viz rs path
    plt.figure(num=1)
    plt.plot(best_rs_path.x,
             best_rs_path.y, marker="o", color='g')
    plt.grid()

    plt.figure(num=2)
    plt.plot(CalculatePathCurvature(
        best_rs_path.x, best_rs_path.y))
    plt.title("Path Curvature")

    w_ref = 1
    w_terminal = 100
    w_delta = 1
    w_acc = 1
    state_cost_func = partial(
        StateCostFunction, ref_path_x=best_rs_path.x, ref_path_y=best_rs_path.y, ref_path_yaw=best_rs_path.yaw, w_ref=w_ref, w_delta=w_delta, w_acc=w_acc)
    terminal_cost_func = partial(
        TerminalCostFunction, ref_path_x=best_rs_path.x, ref_path_y=best_rs_path.y, ref_path_yaw=best_rs_path.yaw, w_terminal=w_terminal)
    in_con_func = partial(InequalityConstraintFunction,
                          max_delta=max_steer, max_acc=max_acc)
    in_con_cost_func = partial(
        InequalityConstraintCostFunction, max_delta=max_steer, max_acc=max_acc)

    # Init position and control input
    x0 = np.array([30, 10, np.deg2rad(0.0), 0.0])
    u0 = [np.array([0.0, 0.0])] * (len(best_rs_path.x) - 1)

    # ilqr
    ilqr_solver_param = ILQRSolverParameter(1e-3, 50, -1e-4, 10, 0.5, 1e-4)
    ilqr_solver = ILQRSolver(BicycleModelRK3, state_cost_func, terminal_cost_func, len(
        best_rs_path.x) - 1, delta_t, ilqr_solver_param)
    res_ilqr = ilqr_solver.Solve(x0, u0)

    # al-ilqr
    al_ilqr_solver_param = ALILQRSolverParameter(
        1e-3, 50, 50, -1e-4, 10, 0.5, 1e-5, 10, 1e6, 1e-4)
    al_ilqr_solver = ALILQRSolver(BicycleModelRK3, state_cost_func, terminal_cost_func, in_con_cost_func, in_con_func, len(
        best_rs_path.x) - 1, delta_t, al_ilqr_solver_param)
    res_al_ilqr = al_ilqr_solver.Solve(x0, u0)

    # Viz ilqr and al_ilqr path
    x_ilqr = res_ilqr['x_hist'][-1]
    u_ilqr = res_ilqr['u_hist'][-1]
    x_ilqr_list = []
    y_ilqr_list = []
    yaw_ilqr_list = []
    v_ilqr_list = []
    delta_ilqr_list = []
    acc_ilqr_list = []
    for j in range(len(x_ilqr)):
        x_ilqr_list.append(x_ilqr[j][0])
        y_ilqr_list.append(x_ilqr[j][1])
        yaw_ilqr_list.append(x_ilqr[j][2])
        v_ilqr_list.append(x_ilqr[j][3])

    for j in range(len(u_ilqr)):
        delta_ilqr_list.append(u_ilqr[j][0])
        acc_ilqr_list.append(u_ilqr[j][1])

    x_al_ilqr = res_al_ilqr['x_hist'][-1]
    u_al_ilqr = res_al_ilqr['u_hist'][-1]
    x_al_ilqr_list = []
    y_al_ilqr_list = []
    yaw_al_ilqr_list = []
    v_al_ilqr_list = []
    delta_al_ilqr_list = []
    acc_al_ilqr_list = []
    for j in range(len(x_al_ilqr)):
        x_al_ilqr_list.append(x_al_ilqr[j][0])
        y_al_ilqr_list.append(x_al_ilqr[j][1])
        yaw_al_ilqr_list.append(x_al_ilqr[j][2])
        v_al_ilqr_list.append(x_al_ilqr[j][3])

    for j in range(len(u_al_ilqr)):
        delta_al_ilqr_list.append(u_al_ilqr[j][0])
        acc_al_ilqr_list.append(u_al_ilqr[j][1])

    plt.figure(num=1)
    plt.plot(x_ilqr_list,
             y_ilqr_list, marker="o")
    plt.plot(x_al_ilqr_list,
             y_al_ilqr_list, marker="o")
    plt.legend(["Reeds-sheps", "ILQR", "AL-ILQR"])
    plt.figure(num=2)
    plt.plot(CalculatePathCurvature(
        x_ilqr_list, y_ilqr_list))
    plt.plot(CalculatePathCurvature(
        x_al_ilqr_list, y_al_ilqr_list))
    plt.legend(["Reeds-sheps", "ILQR", "AL-ILQR"])
    plt.grid()

    plt.figure(num=3)
    plt.subplot(411)
    plt.plot(yaw_ilqr_list)
    plt.plot(yaw_al_ilqr_list)
    plt.legend(["ILQR", "AL-ILQR"])
    plt.grid()
    plt.title("Yaw")

    plt.subplot(412)
    plt.plot(v_ilqr_list)
    plt.plot(v_al_ilqr_list)
    plt.legend(["ILQR", "AL-ILQR"])
    plt.grid()
    plt.title("Velocity")

    plt.subplot(413)
    plt.plot(delta_ilqr_list)
    plt.plot(delta_al_ilqr_list)
    plt.plot([max_steer] * len(delta_ilqr_list), 'r')
    plt.plot([-max_steer] * len(delta_ilqr_list), 'r')
    plt.legend(["ILQR", "AL-ILQR", "Max Delta", "Min Delta"])
    plt.grid()
    plt.title("Delta")

    plt.subplot(414)
    plt.plot(acc_ilqr_list)
    plt.plot(acc_al_ilqr_list)
    plt.plot([max_acc] * len(acc_ilqr_list), 'r')
    plt.plot([-max_acc] * len(acc_ilqr_list), 'r')
    plt.legend(["ILQR", "AL-ILQR", "Max Acc", "Min Acc"])
    plt.grid()
    plt.title("Acceleration")

    plt.show()


if __name__ == "__main__":
    main()
