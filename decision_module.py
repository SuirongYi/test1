import numpy as np
from casadi import *
import math
from data_module import data
import matplotlib.pyplot as plt
L, W = 4.8, 2.0

class Decision(object):

    def __init__(self, agentID:str):
        self.agentID = agentID
        self.MPC = ModelPredictiveControl()

    def sur_data(self):
        """
        从data模块获取离自车最近的4俩周车的信息
        """
        sur_data = data.get_sur_data(self.agentID).values()
        ego_x, ego_y = data.get_ego_position(self.agentID)[3:5]
        vehs_sort = sorted(sur_data, key=lambda a: (a.x - ego_x) ** 2 + (a.y - ego_y) ** 2)
        if len(vehs_sort) < 5:
            return [[m.x, m.y, m.v, m.phi] for m in vehs_sort]
        else:
            return [[m.x, m.y, m.v, m.phi] for m in vehs_sort[:4]]

    def ego_data(self):
        """从data模块获取自车信息"""
        return list(data.get_ego_position(self.agentID))#list[longitudinalvelocity,lateralvelocity,yaw_rate,x,y,phi]

    def ref_path(self):
        """从data模块获取参考点"""
        ref_path = data.get_navigation(self.agentID)
        for arr1 in range(len(ref_path)): #若参考轨迹中某两条轨迹的最后两个参考点包涵的参数相等，则选取其中一条为参考轨迹点，否则默认第一条轨迹
            for arr2 in range(arr1 + 1, len(ref_path)):
                if (ref_path[arr1][:3][-2:].tolist() == ref_path[arr2][:3][-2:].tolist()):
                    return ref_path[arr1].tolist()
                else:
                    continue
        return ref_path[0].tolist()

    def init_states(self):
        """返回初始状态值，其中包涵自车6个状态与4*周车四个状态"""
        # ego_data = self.ego_data()
        # sur_data = np.array(self.sur_data()).flatten().tolist()
        # return [ego_data() + sur_data][0]
        return [self.ego_data() + np.array(self.sur_data()).flatten().tolist()][0]

    def update(self):
        """决策更新函数"""
        X0 = np.zeros(86).tolist()
        init_states = self.init_states()  # [ego *6, ov * 4 ]
        ref_path = self.ref_path()   # [4 * n]
        output, control, state_all, g_all, cost = self.MPC.mpc_solver(init_states, X0, ref_path)
        newdata = (output[:, 3], output[:, 4], output[:, 5], output[:, 0])
        data.set_planned_traj(self.agentID, newdata)#set the output newdata[tuple(x,y,phi,v)] to data


class VehicleDynamics(object):
    """车辆动力学模型类"""
    def __init__(self, x_init, tau, per_veh_info_dim=4):
        """
        Args:
            x_init: 初始状态，共22维，前6维为自车状态，后16维为4辆周车状态
            tau: 离散时间间隔
            per_veh_info_dim: 周车状态个数
        """
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
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)#分别计算前后轴的垂向载荷
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        self.tau = tau
        self.per_veh_info_dim = per_veh_info_dim
        # self.vd = VehicleDynamics()
        self.vehs = x_init[6:]
        self.x_init = x_init

    def f_xu(self, x, u, tau):
        """车辆状态转移方程
        Args:
            x: 状态量
            u: 控制量
            tau: 离散时间间隔
        """
        v_x, v_y, r, x, y, phi = x[0], x[1], x[2], x[3], x[4], x[5]
        steer, a_x = u[0], u[1] #车轮转角，纵向加速度
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        next_state = [v_x + tau * (a_x + v_y * r), #利用车辆动力学模型计算下一时刻车辆状态
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * power(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (power(a, 2) * C_f + power(b, 2) * C_r) - I_z * v_x),
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      (phi + tau * r)]

        return next_state

    def sur_veh_predict(self, vehs):
        """周车状态预测
        Args:
            vehs: 周车信息
        """
        veh_x, veh_y, veh_v, veh_phi = vehs[0], vehs[1], vehs[2], vehs[3]
        # veh_phis_rad = veh_phi * np.pi / 180.
        veh_x_delta = veh_v * self.tau * math.cos(veh_phi)#递推下一时刻周车横向位移
        veh_y_delta = veh_v * self.tau * math.sin(veh_phi)
        veh_phi_rad_delta = 0#默认周车航向角不变
        # veh_phi_rad_delta = veh_r * self.tau           # TODO: 确认是否可以获得角速度来进行估计

        next_veh_x, next_veh_y, next_veh_v, next_veh_phi = \
            veh_x + veh_x_delta, veh_y + veh_y_delta, veh_v, veh_phi + veh_phi_rad_delta
        # next_veh_phi = next_veh_phi_rad
        next_veh_phi = deal_with_phi(next_veh_phi)

        return [next_veh_x, next_veh_y, next_veh_v, next_veh_phi]

    def vehs_pred(self):
        """更新四辆周车下一时刻状态"""
        vehs_pred = []
        for vehs_index in range(4):
            vehs_pred += \
                self.sur_veh_predict(
                    self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim])
        self.vehs = vehs_pred

    def ego_veh_predict(self, x, u):
        """自车状态预测"""
        next_ego = self.f_xu(x, u, self.tau)           # Unit of heading angle is degree
        return next_ego

    def construct_sur_constraints(self, x):
        """构建自车与周车距离约束"""
        ego_x, ego_y, ego_phi = x[3], x[4], x[5]
        g_list = []
        ego_lws = (L - W) / 2.              # TODO：自车长宽的设定由自车信息引入，考虑在x_init引入
        #自车前后轴的横纵坐标
        ego_front_points = ego_x + ego_lws * cos(ego_phi), \
                           ego_y + ego_lws * sin(ego_phi)
        ego_rear_points = ego_x - ego_lws * cos(ego_phi), \
                          ego_y - ego_lws * sin(ego_phi)
        for vehs_index in range(4): #遍历周车，添加约束，使之不于自车碰撞
            veh = self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim]
            veh_x, veh_y, veh_phi = veh[0], veh[1], veh[3]
            veh_lws = (L - W) / 2.                      # TODO：周车的长宽引入变量，L = veh[4], W = veh[5]， 同时在x_init引入
            #周车前后轴的横纵坐标
            veh_front_points = veh_x + veh_lws * math.cos(veh_phi), \
                               veh_y + veh_lws * math.sin(veh_phi)
            veh_rear_points = veh_x - veh_lws * math.cos(veh_phi), \
                              veh_y - veh_lws * math.sin(veh_phi)
            for ego_point in [ego_front_points, ego_rear_points]:#遍历4辆周车与自车的距离，安全距离为3.5m
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = sqrt(power(ego_point[0] - veh_point[0], 2) + power(ego_point[1] - veh_point[1], 2)) - 3.5
                    g_list.append(veh2veh_dist)#添加约束
        return g_list


class ModelPredictiveControl(object):
    """模型预测控制类"""
    def __init__(self, horizon=10):
        """
        Args:
            horizon: 预测时域
        """
        self.horizon = horizon
        self.base_frequency = 10.
        self.exp_v = 10.
        self.STATE_DIM = 6               # ego_info
        self.ACTION_DIM = 2
        self.vd = None
        self._sol_dic = {'ipopt.print_level': 0,
                         'ipopt.sb': 'yes',
                         'print_time': 0}

    def mpc_solver(self, x_init, XO, ref_path):
        """
        Args:
            x_init: 初始状态
            XO: 列表元素为0
            ref_path:参考轨迹点

        Returns:mpc求解结果

        """
        self.vd = VehicleDynamics(x_init, 1 / self.base_frequency)
        x = SX.sym('x', self.STATE_DIM) #casadi框架下，声明变量x，u
        u = SX.sym('u', self.ACTION_DIM)

        f = vertcat(*self.vd.ego_veh_predict(x, u))
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
        w += [Xk]#添加优化目标，车辆初始状态
        lbw += x_init[:6]
        ubw += x_init[:6]                 # force the X0 = x_init
        g = vertcat(*self.vd.construct_sur_constraints(x_init))
        G_f = Function('Gf', [x], [g])

        for k in range(1, self.horizon + 1):#遍历预测步长，添加优化目标和约束条件
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += [-0.4, -4.]                    # todo: action constraints
            ubw += [0.4, 2.]

            Fk = F(Xk, Uk) #自车下一步预测
            Gk = G_f(Xk)#周车下一步预测
            self.vd.vehs_pred()
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
            #
            # print("x:  ",x)
            # print("u  ",u)
            # print("ref_path  ",ref_path)
            # print("k  ", k)
            # Cost function
            F_cost = Function('F_cost', [x, u], [0.05 * power(x[0]-ref_path[3][k], 2) \
                                                 + 0.8 * power(x[3] - ref_path[0][k], 2) \
                                                 + 0.8 * power(x[4] - ref_path[1][k], 2) \
                                                 + 30 * power((x[5] - ref_path[2][k]), 2) \
                                                 + 0.02 * power(x[2], 2) \
                                                 + 5 * power(u[0], 2) \
                                                 + 0.05 * power(u[1], 2) \
                                                 ])                         # TODO: cost的形式没有改正确，重点是自车状态和ref没有确定

            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        #构造casadi框架下的优化问题：优化对象为f = J, 约束条件为g = G, 求解/优化结果为x = w
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)



        # load constraints and solve NLP
        r = S(lbx=vertcat(*lbw), ubx=vertcat(*ubw), x0=XO, lbg=vertcat(*lbg), ubg=vertcat(*ubg))

        state_all = np.array(r['x'])#求解结果为自车状态
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
        # x, y, phi, v_long, v_lat, phi_a,

def deal_with_phi(phi):
    """将航向角约束在[-180°，180°]内"""
    return if_else(phi > pi, phi - 2*pi, if_else(phi < -pi, phi + 2*pi, phi))

def create_decision(agentID: str):
    return Decision(agentID)

