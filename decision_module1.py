import numpy as np
from casadi import *
import math
import time
# from data_module import data
import matplotlib.pyplot as plt

# 车的长与宽
L, W = 4.8, 2.0


# 决策模块：
class Decision(object):
    def __init__(self):
        # 调用 Processor()类
        self.Processor = Processor()
        # 调用 ModelPredictiveControl()类
        self.MPC = ModelPredictiveControl()

    # 输入初始状态以及参考路径，输出MPC解
    def update(self, init_states, ref_path):

        # self.Processor.nav_data = self.data.get_navigation(id=0)
        # self.Processor.sensor_data = dict(ego_data=self.data.get_navigation(id=0),
        #                                   sur_data=self.data.get_traffic(id=0))
        # self.Processor.nav_data = nav_data
        # self.Processor.sensor_data = sensor_data
        # trajs = self.Processor.get_trajs()
        # init_states = self.Processor.get_nearest_vehs()

        # 生成大小为(1，86)的列表,元素都是0 ：[0 , 0, ...,0]
        X0 = np.zeros(86).tolist()
        # 调用类中的求解函数得到结果
        return self.MPC.mpc_solver(init_states, X0, ref_path)

        # info = {'index': {'action': action, 'cost': cost, 'dd': deaddistance}}


# 车辆模型类：存储车辆参数 + 离散系统下的车辆状态转移方程
class VehicleDynamics(object):
    def __init__(self, ):

        # 生成一个字典：用于调用车辆参数
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )

        # 在vehicle_params字典中，按照 key 获取 value，物理意义依次是： 质心到前轴距离、质心到后轴距离、整车质量、重力加速度
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']

        # 分别计算前后轴的垂向载荷
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)

        # 将前后轴载荷计算结果添加到vehicle_params字典中
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    # 离散系统下，车辆的状态转移方程，简而言之：x(t+1) = f(x_t,u_t,dt)
    def f_xu(self, x, u, tau):

        # 车辆状态，依次是：[纵向速度、横向速度、横摆角速度、 纵向坐标、横向坐标、航向角]
        v_x, v_y, r, x, y, phi = x[0], x[1], x[2], x[3], x[4], x[5]
        # 角度转弧度: °--->rad
        phi = phi * np.pi / 180.
        # 控制量依次是：[车轮转角，纵向加速度]
        steer, a_x = u[0], u[1]
        # 从vehicle_params字典中提取参数，依次是：[前轮侧偏刚度、后轮侧偏刚度、质心到前轴距离、质心到后轴距离、整车质量、绕z轴的转动惯量、胎-陆摩擦系数、重力加速度]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']
        # 利用模型计算下一时刻车辆状态：x(t+1) = f(x_t,u_t,dt)
        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * power(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (power(a, 2) * C_f + power(b, 2) * C_r) - I_z * v_x),
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      (phi + tau * r) * 180 / np.pi]
        # 返回下一时刻车辆状态
        return next_state



# 该类功能：周车状态预测 + 构造自车与周车约束
class Dynamics(object):

    # 输入初始状态、离散时间间隔、单车的状态维数
    def __init__(self, x_init, tau, per_veh_info_dim=4):

        # 初始化：将输入依次存储
        self.x_init = x_init
        self.tau = tau
        self.per_veh_info_dim = per_veh_info_dim
        # 调用车辆模型类：VehicleDynamics()
        self.vd = VehicleDynamics()
        # 存储x_init[6:]  todo： 需要明确 x_init[6:]是什么，猜测是四辆周车的信息，每辆车状态是四维
        self.vehs = x_init[6:]

    # 递推四辆周车状态并更新self.vehs
    def vehs_pred(self):
        vehs_pred = []
        # 遍历，调用sur_veh_predict 函数递推四辆周车状态并更新
        for vehs_index in range(4):
            vehs_pred += \
                self.sur_veh_predict(
                    self.vehs[vehs_index * self.per_veh_info_dim : (vehs_index + 1) * self.per_veh_info_dim])

        # 递推四辆周车状态并更新self.vehs
        self.vehs = vehs_pred

    # 周车的状态预测
    def sur_veh_predict(self, vehs):
        # 大地坐标系下，周车 横向位置、纵向位置、速度、航向角
        veh_x, veh_y, veh_v, veh_phi = vehs[0], vehs[1], vehs[2], vehs[3]
        # ° ---> rad
        veh_phis_rad = veh_phi * np.pi / 180.
        # 离散时间内，大地坐标系下周车横纵向位移
        veh_x_delta = veh_v * self.tau * math.cos(veh_phis_rad)
        veh_y_delta = veh_v * self.tau * math.sin(veh_phis_rad)
        # 默认预测期间，周车航向角不变
        veh_phi_rad_delta = 0
        # veh_phi_rad_delta = veh_r * self.tau           # TODO: 确认是否可以获得角速度来进行估计
        # 计算周车下一时刻状态
        next_veh_x, next_veh_y, next_veh_v, next_veh_phi_rad = \
            veh_x + veh_x_delta, veh_y + veh_y_delta, veh_v, veh_phis_rad + veh_phi_rad_delta
        # rad ---> °
        next_veh_phi = next_veh_phi_rad * 180 / np.pi
        # 航向角约束在[-180°，+180°]
        next_veh_phi = deal_with_phi(next_veh_phi)
        # 返回周车一步预测状态 [横向位置、纵向位置、速度、航向角]
        return [next_veh_x, next_veh_y, next_veh_v, next_veh_phi]

    # 调用车辆模型类 VehicleDynamics：完成一步自车状态预测
    def ego_veh_predict(self, x, u):
        next_ego = self.vd.f_xu(x, u, self.tau)           # Unit of heading angle is degree
        return next_ego

    # 构造自车与周车约束
    def construct_sur_constraints(self, x):
        # 自车 横向坐标、纵向坐标、航向角
        ego_x, ego_y, ego_phi = x[3], x[4], x[5]
        # 约束集合
        g_list = []
        # 自车 (长-宽)/2
        ego_lws = (L - W) / 2.              # TODO：自车长宽的设定由自车信息引入，考虑在x_init引入
        # 自车前轴的横纵坐标
        ego_front_points = ego_x + ego_lws * cos(ego_phi * np.pi / 180.), \
                           ego_y + ego_lws * sin(ego_phi * np.pi / 180.)
        # 自车后轴的横纵坐标
        ego_rear_points = ego_x - ego_lws * cos(ego_phi * np.pi / 180.), \
                          ego_y - ego_lws * sin(ego_phi * np.pi / 180.)

        # 实质是添加约束：遍历周车，使之不与自车碰撞
        for vehs_index in range(4):
            # 遍历周车
            veh = self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim]
            # 获取周车横纵向位置与航向角
            veh_x, veh_y, veh_phi = veh[0], veh[1], veh[3]
            # 本意应该是周车的 (长-宽)/2
            veh_lws = (L - W) / 2.                      # TODO：周车的长宽引入变量，L = veh[4], W = veh[5]， 同时在x_init引入
            # 周车前轴的横纵坐标
            veh_front_points = veh_x + veh_lws * math.cos(veh_phi * np.pi / 180.), \
                               veh_y + veh_lws * math.sin(veh_phi * np.pi / 180.)
            # 周车后轴的横纵坐标
            veh_rear_points = veh_x - veh_lws * math.cos(veh_phi * np.pi / 180.), \
                              veh_y - veh_lws * math.sin(veh_phi * np.pi / 180.)
            # 计算自车前轴坐标点与每辆周车前后轴坐标点的距离，自车后轴坐标点与每辆周车前后轴坐标点的距离：共 2*2*4 =16 个约束，注意约束的添加顺序是一辆一辆算周车的
            # 遍历自车前后轴坐标点
            for ego_point in [ego_front_points, ego_rear_points]:
                # 遍历周车前后轴坐标点
                for veh_point in [veh_front_points, veh_rear_points]:
                    # 绝对距离计算 - 3.5 ： 猜测3.5是安全距离，单位m
                    veh2veh_dist = sqrt(power(ego_point[0] - veh_point[0], 2) + power(ego_point[1] - veh_point[1], 2)) - 3.5
                    # 添加约束
                    g_list.append(veh2veh_dist)
        # 返回自车与周车约束
        return g_list



# 模型预测控制，需要调用casadi实现
class ModelPredictiveControl(object):
    # 初始化：预测时域、频率(倒数为离散时间)、期望速度、自车状态维数、动作维数、None、求解器字典
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

    # mpc求解器：输出初始状态、XO、参考路径
    def mpc_solver(self, x_init, XO, ref_path):
        # 调用 Dynamics 类
        self.dynamics = Dynamics(x_init, 1 / self.base_frequency)
        # casadi框架下: 声明变量x，u
        x = SX.sym('x', self.STATE_DIM)
        u = SX.sym('u', self.ACTION_DIM)
        # casadi框架下声明向量f: 将"x" "u"作为输入，调用ego_veh_predict函数，得到输出, 将输出用casadi框架下的vertcat表示
        f = vertcat(*self.dynamics.ego_veh_predict(x, u))
        # casadi框架下创建一个函数，命名为F ： 描述casadi框架下变量 “x” “u” 与 “f” 的关系（通过x，u构造f）
        F = Function("F", [x, u], [f])          # ego predict model

        # 创建用于优化非线性问题的空列表
        # Create empty NLP
        # 优化目标/对象
        w = []                   # variables to optimize
        # 状态与动作约束条件的下界
        lbw = []                 # lower bound for state and action constraints
        # 状态与动作约束条件的上界
        ubw = []                 # upper bound for state and action constraints
        # 距离约束条件的下界
        lbg = []                 # lower bound for distance constraint
        # 距离约束条件的上界
        ubg = []                 # upper bound for distance constraint
        # 动力学约束条件
        G = []                   # dynamic constraints ( reduce the nonlinear of the original NLP)
        # 目标函数/代价函数
        J = 0                    # accumulated cost

        # Initial conditions
        # casadi框架下初始化状态,取名X0
        Xk = MX.sym('X0', self.STATE_DIM)
        # 添加优化目标：车辆初始状态
        w += [Xk]
        # 添加初始状态的约束条件(初始状态是自约束，上下界都是本身)
        lbw += x_init[:6]
        ubw += x_init[:6]                 # force the X0 = x_init

        # 得到casadi框架下的变量g
        g = vertcat(*self.dynamics.construct_sur_constraints(x))
        # casadi框架下创建一个函数G_f，命名为"Gf" ：描述casadi框架下变量 “x” 与 “g” 的关系（通过x构造g）
        G_f = Function('Gf', [x], [g])

        # 遍历预测步长，添加优化目标和约束条件：
        for k in range(1, self.horizon + 1):
            # Local control
            # casadi框架下: 声明控制变量U0~Un
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            # 添加优化目标/对象(自车动作)
            w += [Uk]
            # 添加动作约束的上下界
            lbw += [-0.4, -4.]                    # todo: action constraints
            ubw += [0.4, 2.]

            # 调用casadi框架下声明的函数F，实质是实现自车状态的一步预测
            Fk = F(Xk, Uk)
            # 调用casadi框架下声明的函数G_f，实质是实现周车状态的一步预测
            Gk = G_f(Xk)
            # 周车递推一步，改变周车状态
            self.dynamics.vehs_pred()
            # casadi框架下: 声明自车状态变量X0~Xn
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.STATE_DIM)

            # Dynamic Constraints
            # 自车状态递推一步后的自约束：上下界都是0 （通过规定上下界都是0，使状态最优解唯一：解为模型递推结果本身）
            G += [Fk - Xk]                                         # ego vehicle dynamic constraints
            lbg += [0.0] * self.STATE_DIM
            ubg += [0.0] * self.STATE_DIM
            # 自车与周车距离的上下界约束：为达到安全距离，给一个非负范围
            G += [Gk]                                              # surrounding vehicle constraints
            lbg += [0.0] * (4 * 4)
            ubg += [inf] * (4 * 4)
            # 添加优化目标/对象(自车状态) ： 这里第一维状态范围是[0~8],其他是正负无穷
            w += [Xk]
            lbw += [0.] + [-inf] * (self.STATE_DIM - 1)         # speed constraints
            ubw += [8.] + [inf] * (self.STATE_DIM - 1)

            # casadi框架下创建一个函数F_cost，命名为"F_cost" ：描述casadi框架下变量 “x”  “u” 与 “x^TQX + u^TRu”的关系
            # todo:1. 如果x，u维数固定，最好写成 x^TQX + u^TRu； 2.

            # Cost function
            F_cost = Function('F_cost', [x, u], [0.05 * power(x[0]-ref_path[3][k], 2)
                                                 + 0.8 * power(x[3] - ref_path[0][k], 2)
                                                 + 0.8 * power(x[4] - ref_path[1][k], 2)
                                                 + 30 * power((x[5] - ref_path[2][k]) * np.pi / 180., 2)
                                                 + 0.02 * power(x[2], 2)
                                                 + 5 * power(u[0], 2)
                                                 + 0.05 * power(u[1], 2)
                                                 ])                         # TODO: cost的形式没有改正确，重点是自车状态和ref没有确定
            # 叠加每步的“代价”，组成代价函数
            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        # 构造casadi框架下的优化问题：优化对象为f = J, 约束条件为g = G, 求解/优化结果为x = w
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        # 构造casadi框架下的求解器： 取名'S', 内置求解器‘ipopt’, 其他配置：self._sol_dic
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # load constraints and solve NLP
        # 为casadi框架下的求解器添加约束条件，得到求解/优化结果
        r = S(lbx=vertcat(*lbw), ubx=vertcat(*ubw), x0=XO, lbg=vertcat(*lbg), ubg=vertcat(*ubg))

        # 提取求解结果，state_all为求解得到的自车状态和动作序列
        state_all = np.array(r['x'])
        # 提取求解结果，g_all为求解得到的周车信息序列
        g_all = np.array(r['g'])
        # 得到大小为(预测步数，状态维数)的数组
        state = np.zeros([self.horizon, self.STATE_DIM])
        # 得到大小为(预测步数，动作维数)的数组
        control = np.zeros([self.horizon, self.ACTION_DIM])
        # 得到大小为(预测步数，状态维数+动作维数)的数组
        nt = self.STATE_DIM + self.ACTION_DIM  # total variable per step
        # 提取求解结果，cost为求解得到“代价”序列
        cost = np.array(r['f']).squeeze(0)

        # 将得到的序列信息存起来
        # save trajectories
        for i in range(self.horizon):
            state[i] = state_all[nt * i: nt * (i + 1) - self.ACTION_DIM].reshape(-1)
            control[i] = state_all[nt * (i + 1) - self.ACTION_DIM: nt * (i + 1)].reshape(-1)
        return state, control, state_all, g_all, cost



# 类
class Processor(object):

    # 初始化
    def __init__(self):
        # self.ego_xy = cur_position
        # self.nav_data = nav    # nav is a dict
        # self.sensor_data = sensor   # ?
        # 预测 时域/步数
        self.horizon = 20
        # self.exp_v = 10  # m/s
        # s 离散时间间隔
        self.tau = 0.1
        self.out_20_point = None

    # 获取参考路径轨迹
    def get_trajs(self):
        trajs = [{},{},{}]
        ref_points = []
        self.nav_data['ref_points'] = ref_points
        for i in ref_points:
            for j in ref_points[i]:
                trajs[i]['path'] = ref_points[i][j][:,0:42:2]
                trajs[i]['dead_dist'] = self.nav_data['dead_dist']
        return trajs

    # 找到最近的车辆
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



# 将航向角约束在[-180°,180°]
def deal_with_phi(phi):
    # casadi函数： if_else(DM cond, DM if_true, DM if_false, bool short_circuit) -> DM
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


