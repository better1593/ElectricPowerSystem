import pickle

from PIL.ImageChops import offset
from Risk_Evaluate.pcnumber import *
from Risk_Evaluate.edge_image import *
from Risk_Evaluate.lighting_parameters_distribution import *
from Risk_Evaluate.Lightning_stroke_location import *
from Risk_Evaluate.current_waveform_generator import *
from Utils.Math import distance
import numpy as np
import json
import pandas as pd
from Model.Lightning import Stroke,Lightning,Channel
import math
import copy
# network = pickle.load(open('../Data/output/network.pkl', 'rb'))
# print(network.dt)

def run_MC(network,load_dict):

    Line = {}
    Line["Node_all"] = np.empty((0, 5))
    Line["Edge_all"] = np.empty((0, 3))
    Line["SWnumber"] = []
    Line["Pcnumber"] = []
    Line["segments"] = []
    Line["Suppose_OHLP"] = []
    Line["Span"] = []
    Line["Tower"] = []
    max_length = 50
    def find_height(id):
        for tower in network.towers:
            if tower.info.ID == id:
                return tower.info.Pole_Height
    for tower in network.towers:
        new_row = np.array([[tower.info.ID,tower.info.position[0],tower.info.position[1],tower.info.position[2]
                          ,tower.info.Pole_Height]]).astype(float)
        # 将新行添加到ndarray中
        Line["Node_all"] = np.vstack([Line["Node_all"], new_row])
        Line["Tower"].append(int(tower.info.ID))
    for ohl in network.OHLs:
        Line["Span"].append(int(ohl.info.ID))
        new_row = np.array([[ohl.info.ID, ohl.info.HeadTower_id,ohl.info.TailTower_id]]).astype(float)
        Line["Edge_all"] = np.vstack([Line["Edge_all"] , new_row])

        s = np.empty((0, 9))
        span = np.array([[ohl.info.HeadTower_pos[0],ohl.info.HeadTower_pos[1],ohl.info.HeadTower_pos[2],
                         ohl.info.TailTower_pos[0],ohl.info.TailTower_pos[1],ohl.info.TailTower_pos[2]
                          ,0,0,0]]).astype(float)
        s = np.vstack([s, span])

        sw = 0
        pc = 0
        for wire in ohl.wires.air_wires:
            if wire.name[5] == "S":
                sw += 1
                new_wire = np.array([[wire.start_node.x, wire.start_node.y, wire.start_node.z,
                                  wire.end_node.x, wire.end_node.y, wire.end_node.z, wire.offset,
                                  wire.name[1:5],0]]).astype(float)
                s = np.vstack([s, new_wire])
            pc +=1
            new_wire = np.array([[wire.start_node.x, wire.start_node.y, wire.start_node.z,
                                  wire.end_node.x, wire.end_node.y, wire.end_node.z, wire.offset,
                                  wire.name[1:5], pc]]).astype(float)
            s = np.vstack([s, new_wire])

        Line["Suppose_OHLP"].append(s)
        Line["SWnumber"].append(sw)
        Line["Pcnumber"].append(ohl.phase_num)
        length = distance(ohl.info.HeadTower_pos, ohl.info.TailTower_pos)
        segment_num = int(np.ceil(length / network.max_length))
        Line["segments"].append(segment_num)


    Line["SWnumber"] = np.array(Line["SWnumber"]).reshape(1, len(network.OHLs))
    Line["Pcnumber"] = np.array(Line["Pcnumber"]).reshape(1, len(network.OHLs))
    Line["segments"] = np.array(Line["segments"]).reshape(1, len(network.OHLs))

    Line['Slope_angle'] = np.array(
        [[10, 20], [20, 30], [30, 40]])  # 每条span的斜坡角度[上（左），下（右）各一个角度（度数）]，斜坡起点是最初的落雷点位置到最近的pole-pole线的垂足点
    # 多个building
    Line['buildings'] = [
        {
            'building_XY': np.array([[50, 0], [70, 0], [50, 20], [70, 20]]),  # building四个点的XY坐标（顺序依次是左下角，右下角，左上角，右上角）
            'building_height': 8.5  # building的高度
        },
        {
            'building_XY': np.array([[30, 10], [40, 10], [30, 25], [40, 25]]),  # 第二个建筑物
            'building_height': 10.0
        }
    ]

    Line['OHLPf'] = pcnumber(Line)  # 判断phase conductor有几相-OHLPf

    # 函数
    Line['Node'] = Line['Node_all'][:, 1: 3]
    Line['Edges'] = Line['Edge_all'][:, 1: 3]

    json_file_path = "Data/input/MC.json"
    # 0. read json file
    with open(json_file_path, 'r', encoding="utf-8") as j:
        load_dict = json.load(j)


    DSave = load_dict["MC"]["DSave"]
    MC_lgtn = load_dict["MC"]["MC_lgtn"]
    AR = load_dict["MC"]["AR"]
    surrounding_distance = 100
    foldname = "Data/output"
    Wave_Model = 1
    # 1. 画范围框
    [resultedge, XY_need3] = edge_image(Line, DSave, surrounding_distance, foldname)


    # 2. number of flashes and number of strokes
    [resultcur, parameterst, flash_stroke, points_need] = lighting_parameters_distribution(MC_lgtn, DSave, Line,
                                                                                           resultedge, foldname)
    # 3. 判断落雷点位置
    [resultstro, dataSTS, stroke_result] = Lightning_stroke_location(Line, resultcur, DSave, foldname,
                                                                     resultedge, AR)

    # 4. 描述电流的波形
    light_final = current_waveform_generator(Wave_Model, DSave, resultstro, resultcur, foldname)

    DIND = stroke_result[0]
    DINDs = np.array([1 if x == 'Direct' else 0 for x in DIND])
    DIND2 = DINDs[resultcur['siteone']]
    flash_stroke = np.hstack((flash_stroke, DIND2.reshape(-1, 1)))
    df27 = pd.DataFrame(flash_stroke, columns=['flash', 'stroke', 'Direct1_Indirect2'])
    return df27,parameterst,stroke_result
