import numpy as np
from casadi import *
import math
import time
# from data_module import data
import matplotlib.pyplot as plt
L, W = 4.8, 2.0

class Decision(object):
    def __init__(self):
        self.Processor = Processor()
        self.MPC = ModelPredictiveControl()

    def update(self, init_states, ref_path):
        # self.Processor.nav_data = self.data.get_navigation(id=0)
        # self.Processor.sensor_data = dict(ego_data=self.data.get_navigation(id=0),
        #                                   sur_data=self.data.get_traffic(id=0))
        # self.Processor.nav_data = nav_data
        # self.Processor.sensor_data = sensor_data
        # trajs = self.Processor.get_trajs()
        # init_states = self.Processor.get_nearest_vehs()
        X0 = np.zeros(86).tolist()
        return self.MPC.mpc_solver(init_states, X0, ref_path)

        # info = {'index': {'action': action, 'cost': cost, 'dd': deaddistance}}


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, x, u, tau):
        v_x, v_y, r, x, y, phi = x[0], x[1], x[2], x[3], x[4], x[5]
        phi = phi * np.pi / 180.
        steer, a_x = u[0], u[1]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * power(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (power(a, 2) * C_f + power(b, 2) * C_r) - I_z * v_x),
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return next_state


class Dynamics(object):
    def __init__(self, x_init, tau, per_veh_info_dim=4):
        self.tau = tau
        self.per_veh_info_dim = per_veh_info_dim
        self.vd = VehicleDynamics()
        self.vehs = x_init[6:]
        self.x_init = x_init

    def vehs_pred(self):
        vehs_pred = []
        for vehs_index in range(4):
            vehs_pred += \
                self.sur_veh_predict(
                    self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim])
        self.vehs = vehs_pred

    def sur_veh_predict(self, vehs):
        veh_x, veh_y, veh_v, veh_phi = vehs[0], vehs[1], vehs[2], vehs[3]
        veh_phis_rad = veh_phi * np.pi / 180.
        veh_x_delta = veh_v * self.tau * math.cos(veh_phis_rad)
        veh_y_delta = veh_v * self.tau * math.sin(veh_phis_rad)
        veh_phi_rad_delta = 0
        # veh_phi_rad_delta = veh_r * self.tau           # TODO: 确认是否可以获得角速度来进行估计

        next_veh_x, next_veh_y, next_veh_v, next_veh_phi_rad = \
            veh_x + veh_x_delta, veh_y + veh_y_delta, veh_v, veh_phis_rad + veh_phi_rad_delta
        next_veh_phi = next_veh_phi_rad * 180 / np.pi
        next_veh_phi = deal_with_phi(next_veh_phi)

        return [next_veh_x, next_veh_y, next_veh_v, next_veh_phi]

    def ego_veh_predict(self, x, u):
        next_ego = self.vd.f_xu(x, u, self.tau)           # Unit of heading angle is degree
        return next_ego

    def construct_sur_constraints(self, x):
        ego_x, ego_y, ego_phi = x[3], x[4], x[5]
        g_list = []
        ego_lws = (L - W) / 2.              # TODO：自车长宽的设定由自车信息引入，考虑在x_init引入
        ego_front_points = ego_x + ego_lws * cos(ego_phi * np.pi / 180.), \
                           ego_y + ego_lws * sin(ego_phi * np.pi / 180.)
        ego_rear_points = ego_x - ego_lws * cos(ego_phi * np.pi / 180.), \
                          ego_y - ego_lws * sin(ego_phi * np.pi / 180.)
        for vehs_index in range(4):
            veh = self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim]
            veh_x, veh_y, veh_phi = veh[0], veh[1], veh[3]
            veh_lws = (L - W) / 2.                      # TODO：周车的长宽引入变量，L = veh[4], W = veh[5]， 同时在x_init引入
            veh_front_points = veh_x + veh_lws * math.cos(veh_phi * np.pi / 180.), \
                               veh_y + veh_lws * math.sin(veh_phi * np.pi / 180.)
            veh_rear_points = veh_x - veh_lws * math.cos(veh_phi * np.pi / 180.), \
                              veh_y - veh_lws * math.sin(veh_phi * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = sqrt(power(ego_point[0] - veh_point[0], 2) + power(ego_point[1] - veh_point[1], 2)) - 3.5
                    g_list.append(veh2veh_dist)
        return g_list


class ModelPredictiveControl(object):
    def __init__(self, horizon=10):
        self.horizon = horizon
        self.base_frequency = 10.
        self.exp_v = 10.
        self.STATE_DIM = 6               # ego_info
        self.ACTION_DIM = 2
        self.dynamics = None
        self._sol_dic = {'ipopt.print_level': 0,
                         'ipopt.sb': 'yes',
                         'print_time': 0}

    def mpc_solver(self, x_init, XO, ref_path):
        self.dynamics = Dynamics(x_init, 1 / self.base_frequency)
        x = SX.sym('x', self.STATE_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        f = vertcat(*self.dynamics.ego_veh_predict(x, u))
        F = Function("F", [x, u], [f])          # ego predict model


        # Create empty NLP
        w = []                   # variables to optimize
        lbw = []                 # lower bound for state and action constraints
        ubw = []                 # upper bound for state and action constraints
        lbg = []                 # lower bound for distance constraint
        ubg = []                 # upper bound for distance constraint
        G = []                   # dynamic constraints ( reduce the nonlinear of the original NLP)
        J = 0                    # accumulated cost

        # Initial conditions
        Xk = MX.sym('X0', self.STATE_DIM)
        w += [Xk]
        lbw += x_init[:6]
        ubw += x_init[:6]                 # force the X0 = x_init
        g = vertcat(*self.dynamics.construct_sur_constraints(x))
        G_f = Function('Gf', [x], [g])

        for k in range(1, self.horizon + 1):
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += [-0.4, -4.]                    # todo: action constraints
            ubw += [0.4, 2.]

            Fk = F(Xk, Uk)
            Gk = G_f(Xk)
            self.dynamics.vehs_pred()
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.STATE_DIM)

            # Dynamic Constraints
            G += [Fk - Xk]                                         # ego vehicle dynamic constraints
            lbg += [0.0] * self.STATE_DIM
            ubg += [0.0] * self.STATE_DIM
            G += [Gk]                                              # surrounding vehicle constraints
            lbg += [0.0] * (4 * 4)
            ubg += [inf] * (4 * 4)
            w += [Xk]
            lbw += [0.] + [-inf] * (self.STATE_DIM - 1)         # speed constraints
            ubw += [8.] + [inf] * (self.STATE_DIM - 1)


            # Cost function
            F_cost = Function('F_cost', [x, u], [0.05 * power(x[0]-ref_path[3][k], 2)
                                                 + 0.8 * power(x[3] - ref_path[0][k], 2)
                                                 + 0.8 * power(x[4] - ref_path[1][k], 2)
                                                 + 30 * power((x[5] - ref_path[2][k]) * np.pi / 180., 2)
                                                 + 0.02 * power(x[2], 2)
                                                 + 5 * power(u[0], 2)
                                                 + 0.05 * power(u[1], 2)
                                                 ])                         # TODO: cost的形式没有改正确，重点是自车状态和ref没有确定

            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)



        # load constraints and solve NLP
        r = S(lbx=vertcat(*lbw), ubx=vertcat(*ubw), x0=XO, lbg=vertcat(*lbg), ubg=vertcat(*ubg))

        state_all = np.array(r['x'])
        g_all = np.array(r['g'])
        state = np.zeros([self.horizon, self.STATE_DIM])
        control = np.zeros([self.horizon, self.ACTION_DIM])
        nt = self.STATE_DIM + self.ACTION_DIM  # total variable per step
        cost = np.array(r['f']).squeeze(0)

        # save trajectories
        for i in range(self.horizon):
            state[i] = state_all[nt * i: nt * (i + 1) - self.ACTION_DIM].reshape(-1)
            control[i] = state_all[nt * (i + 1) - self.ACTION_DIM: nt * (i + 1)].reshape(-1)
        return state, control, state_all, g_all, cost


class Processor(object):
    def __init__(self):
        # self.ego_xy = cur_position
        # self.nav_data = nav    # nav is a dict
        # self.sensor_data = sensor   # ?
        self.horizon = 20
        # self.exp_v = 10  # m/s
        self.tau = 0.1  # s
        self.out_20_point = None

    def get_trajs(self):
        trajs = [{},{},{}]
        ref_points = []
        self.nav_data['ref_points'] = ref_points
        for i in ref_points:
            for j in ref_points[i]:
                trajs[i]['path'] = ref_points[i][j][:,0:42:2]
                trajs[i]['dead_dist'] = self.nav_data['dead_dist']
        return trajs

    def get_nearest_vehs(self):

        '''
        sensor_data is a dict
        input: sensor_data-->{'ego_data':np.array([x,y,v_x,v_y,phi,r]),'sur_data':np.array([[x,y,v_y,phi,r],[]....])}
        output:init_states-->np.array([...]) 36dim

        '''
        ego_data = []
        sur_data = []
        init_states = []
        tep = []
        self.sensor_data['ego_data'] = ego_data
        self.sensor_data['sur_data'] = sur_data
        if self.sensor_data['sur_data'].shape[0] > 6:
            for i in range(self.sensor_data['sur_data'].shape[0]):
                dis = np.sum(np.square(ego_data[:2]-sur_data[i][:2]))
                tep.append([dis, i])

            tep.sort()

            for i in range(6):
                init_states = ego_data
                init_states.append(sur_data[tep[i][1]])
        else:
            init_states = ego_data
            init_states.append(sur_data)

        return np.array(init_states)


def deal_with_phi(phi):
    return if_else(phi > 180, phi - 360, if_else(phi < -180, phi + 360, phi))


if __name__ == '__main__':
    Decision = Decision()
    ego_data = np.array([3, 0, 0, 0, 0, 0])
    # sur_data = np.array([[5, 0, 10, 0], [10, 5, 10, 0], [5, 0, 10, 0], [2, -4, 10, 0],
    #                      [0, 10, 10, 0], [0, 20, 10, 0]])
    sur_data = np.array([[10, 0, 0, 8], [10, 0, 0, 10], [10, 0, 0, 10], [10, 0, 0, 10]])#, [10, 0, 0, 10], [10, 0, 0, 10]])    # x,y,v,phi
    init_state = np.concatenate((ego_data, sur_data.flatten()), axis=0).tolist()
    ref_path = np.array([[0.3*i, 0, 0, 3] for i in range(21)]).T    # x, y, phi, v
    ref_path = ref_path.tolist()
    # state, control, state_all, g_all, cost = Decision.update(init_state, ref_path)
    t1 = time.time()
    state, control, state_all, g_all, cost = Decision.update(init_state, ref_path)
    t2 = time.time()
    print(t2-t1)


