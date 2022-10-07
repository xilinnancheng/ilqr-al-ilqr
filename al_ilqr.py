
from re import I
import jax.numpy as np
from jax import grad, jacfwd
from numpy import diag


class ALILQRSolverParameter:
    def __init__(self, reg, al_ilqr_max_iter, ilqr_max_iter, line_search_beta_1, line_search_beta_2, line_search_gamma, J_tolerance, sigma_factor, sigma_max, constraint_tolerance, enable_auto_grad=True) -> None:
        self.enable_auto_grad = enable_auto_grad
        self.reg = reg
        self.al_ilqr_max_iter = al_ilqr_max_iter
        self.ilqr_max_iter = ilqr_max_iter
        self.line_search_beta_1 = line_search_beta_1
        self.line_search_beta_2 = line_search_beta_2
        self.line_search_gamma = line_search_gamma
        self.J_tolerance = J_tolerance
        self.sigma_factor = sigma_factor
        self.sigma_max = sigma_max
        self.constraint_tolerance = constraint_tolerance


class ALILQRSolver:
    def __init__(self, system_dynamic, state_cost_func, terminal_cost_func, in_con_cost_func, in_con_func, step, delta_t, param) -> None:
        self.system_dynamic = system_dynamic
        self.state_cost_func = state_cost_func
        self.terminal_cost_func = terminal_cost_func
        self.in_con_cost_func = in_con_cost_func
        self.in_con_func = in_con_func
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

        self.in_con_cost_func_du = None
        self.in_con_cost_func_dudu = None

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

            self.in_con_cost_func_du = grad(self.in_con_cost_func, 0)
            self.in_con_cost_func_dudu = jacfwd(self.in_con_cost_func_du, 0)
        return

    def EvaluateTrajectoryCost(self, x, u, mu, sigma):
        J = 0.0
        # stage cost
        for i in range(self.step):
            J += self.state_cost_func(x[i], u[i], i) + \
                self.in_con_cost_func(u[i], mu[i], sigma)

        # terminal state cost
        J += self.terminal_cost_func(x[-1])
        return J

    def BuildErrorStateLQRSystem(self, x, u, mu, sigma):
        dfdx = [None] * self.step
        dfdu = [None] * self.step
        dldx = [None] * (self.step + 1)
        dldu = [None] * self.step
        dldxdx = [None] * (self.step + 1)
        dldudu = [None] * self.step
        dldudx = [None] * self.step
        didu = [None] * self.step
        didudu = [None] * self.step

        # terminal state
        dldxdx[-1] = self.terminal_cost_func_dxdx(x[-1])
        dldx[-1] = self.terminal_cost_func_dx(x[-1])

        # stage state
        for i in reversed(range(self.step)):
            dfdx[i] = self.system_dynamic_dx(x[i], u[i], self.delta_t)
            dfdu[i] = self.system_dynamic_du(x[i], u[i], self.delta_t)

            dldx[i] = self.state_cost_func_dx(x[i], u[i], i)
            dldu[i] = self.state_cost_func_du(x[i], u[i], i)
            dldxdx[i] = self.state_cost_func_dxdx(x[i], u[i], i)
            dldudu[i] = self.state_cost_func_dudu(x[i], u[i], i)
            dldudx[i] = self.state_cost_func_dudx(x[i], u[i], i)

            didu[i] = self.in_con_cost_func_du(u[i], mu[i], sigma)
            didudu[i] = self.in_con_cost_func_dudu(u[i], mu[i], sigma)

        es_lqr_system = {'dfdx': dfdx, 'dfdu': dfdu, 'dldx': dldx,
                         'dldu': dldu, 'dldxdx': dldxdx, 'dldudu': dldudu, 'dldudx': dldudx,
                         'didu': didu, 'didudu': didudu}
        return es_lqr_system

    def ForwardPass(self, x, u, k, d, alpha, Qu_list, Quu_list, mu, sigma):
        x_new = []
        u_new = []
        x_new.append(x[0])
        delta_J = 0.0
        for i in range(self.step):
            u_new.append(u[i] + k[i].dot(x_new[i] - x[i]) + alpha * d[i])
            x_new.append(self.system_dynamic(x_new[i], u_new[i], self.delta_t))
            delta_J += alpha * (d[i].T.dot(Qu_list[i]) +
                                0.5 * alpha * d[i].T.dot(Quu_list[i]).dot(d[i]))
        delta_x_final = x_new[-1] - x[-1]
        delta_J += 0.5 * \
            delta_x_final.T.dot(
                self.terminal_cost_func_dxdx(x[-1])).dot(delta_x_final)
        + self.terminal_cost_func_dx(x[-1]).T.dot(delta_x_final)

        J = self.EvaluateTrajectoryCost(x_new, u_new, mu, sigma)
        return x_new, u_new, J, delta_J

    def BackwardPass(self, x, u, mu, sigma):
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

            didudu = self.in_con_cost_func_dudu(u[i], mu[i], sigma)
            didu = self.in_con_cost_func_du(u[i], mu[i], sigma)

            Qxx = dldxdx + dfdx.T.dot(Vxx).dot(dfdx)
            Quu = dldudu + dfdu.T.dot(Vxx).dot(dfdu) + didudu
            Qux = dldudx + dfdu.T.dot(Vxx).dot(dfdx)
            Qx = dldx + dfdx.T.dot(Vx)
            Qu = dldu + dfdu.T.dot(Vx) + didu
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

    def UpdateMu(self, mu, sigma, u):
        mu_new = [None] * len(mu)
        for i in range(len(mu_new)):
            mu_new[i] = mu[i] + sigma * \
                self.in_con_func(u[i], [0.0, 0.0, 0.0, 0.0], 1.0)
            for j in range(len(mu_new[i])):
                mu_new[i].at[j].set(max(0.0, mu_new[i][j]))
        return mu_new

    def ConstraintViolationInfinitNorm(self, u, mu, sigma):
        infinity_norm = 0.0

        for i in range(len(u)):
            violation_norm = np.linalg.norm(
                self.in_con_func(u[i], mu[i], sigma))
            infinity_norm = max(violation_norm, infinity_norm)

        return infinity_norm

    def Solve(self, x0, u_init, verbose=True):
        print("============== AL-ILQR starts ==============")
        # Init trajectory and control input
        u = u_init
        x = self.InitTrajectory(x0, u)

        x_hist = []
        u_hist = []
        sigma = 1.0
        mu = [np.array([0.0] * (2 * len(u[0])))] * self.step

        al_ilqr_iter = 0
        # al-ilqr main loop
        while True:
            print(
                "ALILQR: New al-ilqr iteration {0} starts ...".format(al_ilqr_iter))
            if al_ilqr_iter >= self.param.al_ilqr_max_iter:
                print("ALILQR: Reach ilqr maximum iteration number")
                break

            J_opt = self.EvaluateTrajectoryCost(x, u, mu, sigma)

            # ilqr Main loop
            ilqr_iter = 0
            converged = False
            while not converged:
                print(
                    "ALILQR: New ilqr iteration {0} starts ...".format(ilqr_iter))
                if ilqr_iter >= self.param.ilqr_max_iter:
                    print("ALILQR: Reach ilqr maximum iteration number")
                    break

                # Backward pass
                K, k, Qu_list, Quu_list = self.BackwardPass(x, u, mu, sigma)

                # Line search
                alpha = 1.0
                J_new = 0.0
                accept = False
                while not accept:
                    if alpha < 1e-6:
                        print("ALILQR: Line search fail to decrease cost function")
                    # Forward pass
                    x_new, u_new, J_new, delta_J = self.ForwardPass(
                        x, u, K, k, alpha, Qu_list, Quu_list, mu, sigma)
                    z = (J_opt - J_new) / -delta_J
                    print("ALILQR: J_opt:{0} J_new:{1} delta_J:{2} z:{3}".format(
                        J_opt, J_new, delta_J, z))
                    if ((J_opt - J_new)/J_opt < self.param.J_tolerance and (J_opt - J_new)/J_opt > 0.0) or z > self.param.line_search_beta_1 and z < self.param.line_search_beta_2:
                        x = x_new
                        u = u_new
                        accept = True
                    alpha *= self.param.line_search_gamma

                if accept:
                    if abs(J_opt - J_new)/J_opt < self.param.J_tolerance:
                        converged = True
                        if verbose:
                            print(
                                'ALILQR: Converged at iteration {0}; J={1}; reg={2}'.format(ilqr_iter, J_opt, self.param.reg))
                    J_opt = J_new
                    x_hist.append(x)
                    u_hist.append(u)

                ilqr_iter += 1

            constraint_violation_infinity_norm = self.ConstraintViolationInfinitNorm(
                u, mu, sigma)
            print("ALILQR: New al-ilqr iteration {0} ends ... constraint violation: {1}".format(
                al_ilqr_iter, constraint_violation_infinity_norm))
            if constraint_violation_infinity_norm < self.param.constraint_tolerance:
                break

            mu = self.UpdateMu(mu, sigma, u)
            sigma = min(sigma * self.param.sigma_factor, self.param.sigma_max)
            al_ilqr_iter += 1

        res_dict = {'x_hist': x_hist, 'u_hist': u_hist}
        print("============== AL-ILQR ends ==============")
        return res_dict
