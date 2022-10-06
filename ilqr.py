
from re import I
import jax.numpy as np
from jax import grad, jacfwd
from numpy import diag


class ILQRSolverParameter:
    def __init__(self, reg, max_iter, line_search_beta_1, line_search_beta_2, line_search_gamma, J_tolerance, enable_auto_grad=True) -> None:
        self.enable_auto_grad = enable_auto_grad
        self.reg = reg
        self.max_iter = max_iter
        self.line_search_beta_1 = line_search_beta_1
        self.line_search_beta_2 = line_search_beta_2
        self.line_search_gamma = line_search_gamma
        self.J_tolerance = J_tolerance


class ILQRSolver:
    def __init__(self, system_dynamic, state_cost_func, terminal_cost_func, step, delta_t, param) -> None:
        self.system_dynamic = system_dynamic
        self.state_cost_func = state_cost_func
        self.terminal_cost_func = terminal_cost_func
        self.step = step
        self.delta_t = delta_t
        self.param = param

        self.system_dynamic_dx = None
        self.system_dynamic_du = None

        self.state_cost_func_dx = None
        self.state_cost_func_du = None
        self.state_cost_func_dxdx = None
        self.state_cost_func_dudu = None
        self.state_cost_func_dudx = None

        self.terminal_cost_func_dx = None
        self.terminal_cost_func_dxdx = None

        if(self.param.enable_auto_grad):
            self.system_dynamic_dx = jacfwd(self.system_dynamic, 0)
            self.system_dynamic_du = jacfwd(self.system_dynamic, 1)

            self.state_cost_func_dx = grad(self.state_cost_func, 0)
            self.state_cost_func_du = grad(self.state_cost_func, 1)
            self.state_cost_func_dxdx = jacfwd(self.state_cost_func_dx, 0)
            self.state_cost_func_dudu = jacfwd(self.state_cost_func_du, 1)
            self.state_cost_func_dudx = jacfwd(self.state_cost_func_du, 0)
            self.terminal_cost_func_dx = grad(self.terminal_cost_func, 0)
            self.terminal_cost_func_dxdx = jacfwd(
                self.terminal_cost_func_dx, 0)
        return

    def EvaluateTrajectoryCost(self, x, u):
        J = 0.0
        # stage cost
        for i in range(self.step):
            J += self.state_cost_func(x[i], u[i], i)

        # terminal state cost
        J += self.terminal_cost_func(x[-1])
        return J

    def ForwardPass(self, x, u, k, d, alpha, Qu_list, Quu_list):
        x_new = []
        u_new = []
        x_new.append(x[0])
        delta_J = 0.0
        for i in range(self.step):
            u_new.append(u[i] + k[i].dot(x_new[i] - x[i]) + alpha * d[i])
            x_new.append(self.system_dynamic(x_new[i], u_new[i], self.delta_t))
            delta_J += alpha * (d[i].T.dot(Qu_list[i]) +
                                0.5 * alpha * d[i].T.dot(Quu_list[i]).dot(d[i]))

        J = self.EvaluateTrajectoryCost(x_new, u_new)
        return x_new, u_new, J, delta_J

    def BackwardPass(self, x, u):
        Vx = self.terminal_cost_func_dx(x[-1])
        Vxx = self.terminal_cost_func_dxdx(x[-1])

        fb = [None] * self.step
        ff = [None] * self.step
        Qu_list = [None] * self.step
        Quu_list = [None] * self.step

        for i in reversed(range(self.step)):
            dfdx = self.system_dynamic_dx(x[i], u[i], self.delta_t)
            dfdu = self.system_dynamic_du(x[i], u[i], self.delta_t)

            dldx = self.state_cost_func_dx(x[i], u[i], i)
            dldu = self.state_cost_func_du(x[i], u[i], i)
            dldxdx = self.state_cost_func_dxdx(x[i], u[i], i)
            dldudu = self.state_cost_func_dudu(x[i], u[i], i)
            dldudx = self.state_cost_func_dudx(x[i], u[i], i)

            Qxx = dldxdx + dfdx.T.dot(Vxx).dot(dfdx)
            Quu = dldudu + dfdu.T.dot(Vxx).dot(dfdu)
            Qux = dldudx + dfdu.T.dot(Vxx).dot(dfdx)
            Qx = dldx + dfdx.T.dot(Vx)
            Qu = dldu + dfdu.T.dot(Vx)
            Qu_list[i] = Qu

            inversed_Quu = self.regularized_persudo_inverse(
                Quu, self.param.reg)
            Quu_list[i] = Quu

            fb[i] = -inversed_Quu.dot(Qux)
            ff[i] = -inversed_Quu.dot(Qu)

            Vxx = Qxx + fb[i].T.dot(Quu).dot(fb[i]) + \
                fb[i].T.dot(Qux) + Qux.T.dot(fb[i])
            Vx = Qx + fb[i].T.dot(Quu).dot(ff[i]) + \
                fb[i].T.dot(Qu) + Qux.T.dot(ff[i])

        return fb, ff, Qu_list, Quu_list

    def regularized_persudo_inverse(self, mat, reg=1e-5):
        u, s, v = np.linalg.svd(mat)
        for i in range(len(s)):
            if s[i] < 0:
                s.at[i].set(0.0)
                print("Warning: inverse operator singularity{0}".format(i))
        diag_s_inv = np.diag(1./(s + reg))
        return v.dot(diag_s_inv).dot(u.T)

    def InitTrajectory(self, x0, u):
        x_init = []
        x_init.append(x0)

        for i in range(self.step):
            x_init.append(self.system_dynamic(x_init[-1], u[i], self.delta_t))
        return x_init

    def Solve(self, x0, u_init, verbose=True):
        print("============== ILQR starts ==============")
        # Init
        u = u_init
        x = self.InitTrajectory(x0, u)
        J_opt = self.EvaluateTrajectoryCost(x, u)
        J_hist = [J_opt]
        x_hist = [x]
        u_hist = [u]

        # Main loop
        iter = 0
        converged = False
        while not converged:
            print("ILQR: New iteration {0} starts ...".format(iter))
            if iter >= self.param.max_iter:
                print("ILQR: Reach the maximum iteration number")
                break

            # Backward pass
            K, k, Qu_list, Quu_list = self.BackwardPass(x, u)

            # Line search
            alpha = 1.0
            J_new = 0.0
            accept = False
            while not accept:
                if alpha < 1e-6:
                    print("ILQR:Line search fail to decrease cost function")
                # Forward pass
                x_new, u_new, J_new, delta_J = self.ForwardPass(
                    x, u, K, k, alpha, Qu_list, Quu_list)
                z = (J_opt - J_new) / -delta_J
                print("ILQR: J_opt:{0} J_new:{1} delta_J:{2} z:{3}".format(
                    J_opt, J_new, delta_J, z))
                if z > self.param.line_search_beta_1 and z < self.param.line_search_beta_2:
                    x = x_new
                    u = u_new
                    accept = True
                alpha *= self.param.line_search_gamma

            iter += 1
            J_hist.append(J_opt)
            x_hist.append(x)
            u_hist.append(u)

            if accept:
                if abs(J_opt - J_new)/J_opt < self.param.J_tolerance:
                    converged = True
                    if verbose:
                        print(
                            'ILQR:Converged at iteration {0}; J={1}; reg={2}'.format(iter, J_opt, self.param.reg))
                J_opt = J_new

        res_dict = {'x_hist': x_hist, 'u_hist': u_hist, 'J_hist': J_hist}
        print("============== ILQR ends ==============")
        return res_dict
