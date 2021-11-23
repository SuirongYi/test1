import pickle
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import copy
import os
import bezier
import os, sys
import numba
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary
import traci
dirname = os.path.dirname(__file__)
SUMOCFG_DIR = dirname + "/map/Map4_Tsinghua_Intersection/configuration.sumocfg"
SUMO_BINARY = checkBinary('sumo-gui')
import warnings
from navigation_dev.navigation_module import Navigation, get_update_info, scale_phi
warnings.filterwarnings("ignore")

dis_interval = 0.5
num_ref_points = 10
step_time = 0.1
DeadDistance_base = 20

def get_sur_veh(vehID):
    vehs = []
    veh_infos = traci.vehicle.getSubscriptionResults(str(vehID))
    veh_info_dict = copy.deepcopy(veh_infos)
    for i, veh in enumerate(veh_info_dict):
        if veh != '0':
            # type = veh_info_dict[veh][traci.constants.VAR_TYPE]
            x, y = veh_info_dict[veh][traci.constants.VAR_POSITION]
            phi = veh_info_dict[veh][traci.constants.VAR_ANGLE]
            length = veh_info_dict[veh][traci.constants.VAR_LENGTH]
            width = veh_info_dict[veh][traci.constants.VAR_WIDTH]
            v = veh_info_dict[veh][traci.constants.VAR_SPEED]
            vehs.append(dict(x=x, y=y, v=v, phi=phi, l=length, w=width))
    return vehs

def main():
    vehID = 0
    navi = Navigation(vehID)
    num = 1200
    import time
    time_list = []
    traci.vehicle.add(vehID=str(vehID), routeID='self_route', departLane=2, typeID="car_4")
    traci.simulationStep()
    x, y = traci.vehicle.getPosition(str(vehID))
    traci.vehicle.subscribeContext('0',
                                    traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                    200.,
                                    [traci.constants.VAR_TYPE,
                                     traci.constants.VAR_POSITION,
                                     traci.constants.VAR_SPEED,
                                     traci.constants.VAR_SPEED_LAT,
                                     traci.constants.VAR_ACCELERATION,
                                     traci.constants.VAR_LENGTH,
                                     traci.constants.VAR_WIDTH,
                                     traci.constants.VAR_ANGLE,  # 67
                                     traci.constants.VAR_LANE_INDEX,  # 82
                                     traci.constants.VAR_ROAD_ID  # 80
                                     ],
                                    0, 10000000)

    for i in range(int(num)):

        # sensor
        vehs = get_sur_veh(vehID)
        sur_veh = np.array(vehs)[:, :4]

        # navi
        edgeID, laneIndex, ahead_lane_length = get_update_info(x, y)
        start = time.perf_counter_ns()
        out_dict = navi.update(x, y, edgeID, laneIndex, ahead_lane_length)
        end = time.perf_counter_ns()
        time_list.append(float(end - start) / 10 ** 6)
        points = out_dict
        ref =

        # decision


        x = points[0][0][0][4]
        y = points[0][0][1][4]
        phi = points[0][0][2][4]
        angle_in_sumo = scale_phi(-phi + np.pi / 2) * 180 / np.pi
        if edgeID == navi.edge_list[-1] and ahead_lane_length < num_ref_points * dis_interval + DeadDistance_base:
            print('finished')
            break
        traci.vehicle.moveToXY(vehID=str(vehID), edgeID=edgeID, lane=laneIndex, x=x, y=y, angle=angle_in_sumo)
        traci.simulationStep()
    traci.close()

if __name__ == "__main__":
    main()