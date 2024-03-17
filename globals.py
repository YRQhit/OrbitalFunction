import math
import numpy as np
GM_Earth = 398600.4415#地球引力参数
GM_Sun = 132712440017.9870#太阳引力参数
GM_Moon = 4902.79#月亮引力参数

r_E = 6378.1363#地球半径
J2 = 1082.6355e-6#非球形参数

deg = 21# 默认非球形摄动阶数取20（HPOP)

# Load Cnm, Snm, lon_lan, SpaceWeather, and DE405Coeff data here

CD = 2.2
s_m = 0.02#默认大气摄动参数

# Load DE405Coeff data here

rad2deg = 180 / math.pi
deg2rad = math.pi / 180

orbitModel = 'LPOP'

E = np.array([[  -0.179248254063455  , 0.983803811524152  , 0.000351917253008],
  [-0.983802197036787 , -0.179248597532891  , 0.001782522954484],
  [ 0.001816733550800  ,-0.000026702839265  , 0.999998349381720]])

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [list(map(float, line.split())) for line in lines]
    return data

import os
current_file_path = os.getcwd()#获取当前路径
parent_directory_path = os.path.abspath(os.path.join(current_file_path, '..'))

PC_path = os.path.abspath(os.path.join(current_file_path, 'PC.txt'))
PC = load_txt_file(PC_path)

Cnm_path = os.path.abspath(os.path.join(current_file_path, 'Cnm.txt'))
Cnm = load_txt_file(Cnm_path)
C = Cnm[:deg + 1][:]
Snm_path = os.path.abspath(os.path.join(current_file_path, 'Snm.txt'))
Snm = load_txt_file(Snm_path)
S = Snm[:deg + 1][:]
spaceWheather_path = os.path.abspath(os.path.join(current_file_path, 'spaceWheather.txt'))
spaceWheather = load_txt_file(spaceWheather_path)

DE405Coeff_path = os.path.abspath(os.path.join(current_file_path, 'DE405Coeff.txt'))
DE405Coeff = load_txt_file(DE405Coeff_path)

