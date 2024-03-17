import math
import numpy as np
from OrbitCore import *
from sofa.Mjday import Mjday
from sofa.ICRS2ITRS import ICRS2ITRS
import globals
from scipy.integrate import odeint
from scipy.optimize import minimize
from skyfield.api import Topos, load
import numpy as np
# 计算星下点
from ToolFunction import *
from OrbitPredict import *

class ToolFunction:
    def fuelCost(self,dv, m, I=None):
        if I is None:
            I = 285

        g0 = 9.8
        dv = abs(dv) * 1000
        fuel = m * (1 - math.exp(-dv / I / g0))
        return fuel
    def getSunVector(self,JD):
        Pi = 3.141592653589793238462643383279502884197169399375105
        T = (JD - 2451545.0) / 36525.
        lambdaM = 280.460 + 36000.771 * T
        M = 357.5277233 + 35999.05034 * T
        lambda_sun = lambdaM + 1.91466647 * np.sin(M * Pi / 180.) + 0.019994643 * np.sin(2 * M * Pi / 180.)
        r = 1.000140612 - 0.016708617 * np.cos(M * Pi / 180.) + 0.000139589 * np.cos(2 * M * Pi / 180.)
        e = 23.439291 - 0.0130042 * T

        sun_Vector_J2000 = np.zeros((3, 1))
        sun_Vector_J2000[0, 0] = r * np.cos(lambda_sun * Pi / 180.)
        sun_Vector_J2000[1, 0] = r * np.cos(e * Pi / 180) * np.sin(lambda_sun * Pi / 180.)
        sun_Vector_J2000[2, 0] = r * np.sin(e * Pi / 180.) * np.sin(lambda_sun * Pi / 180.)

        return sun_Vector_J2000
    
    #计算太阳光照角（太阳指向目标星的矢量在目标星轨道平面的投影与追踪星到目标星的矢量夹角）
    # 输入：x_c - 追踪星位置速度                    x_t - 目标星位置速度
    #      JD_startTime - 儒略日或时间行向量        second - 偏移时间（可以不给定）
    # 输出：idealDegree - 理想太阳光照角（默认两星对地心夹角为0）     actualDegree - 实际太阳光照角
    def cross_product(self,a, b):
        return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

    def IlluminationAngle(self,x_c, x_t, JD_startTime, second=None):
        if len(JD_startTime) == 6:
            JD = Mjday(JD_startTime[0], JD_startTime[1], JD_startTime[2], JD_startTime[3], JD_startTime[4], JD_startTime[5]) + 2400000.5
        else:
            JD = JD_startTime

        if second is not None:
            JD = JD + second / 86400
        Core = OrbitCore()
        t_norm_vector = self.cross_product(x_t[:3], x_t[3:]) / np.linalg.norm(self.cross_product(x_t[:3], x_t[3:]))
        sun_earth_vector = -Core.CalAstroBodyPos(JD)  # 太阳指向地球矢量
        sun_target_vector = x_t[:3] + sun_earth_vector  # 太阳指向目标星矢量

        idealDegree = np.degrees(np.arccos(np.dot(x_t[:3], sun_target_vector) / (np.linalg.norm(x_t[:3]) * np.linalg.norm(sun_target_vector))))

        chase_target_vector = x_t[:3] - x_c[:3]  # 追踪星指向目标星矢量
        actualDegree = np.degrees(np.arccos(np.dot(chase_target_vector, sun_target_vector) / (np.linalg.norm(chase_target_vector) * np.linalg.norm(sun_target_vector))))

        return idealDegree, actualDegree
    
    def Inertial2Orbit(self,RV):
        R = RV[0:3]
        V = RV[3:6]

        if np.linalg.norm(R) <= 0:
            raise ValueError('Satellite Position norm(R) = 0 in Inertial2Orbit!')

        k = -R / np.linalg.norm(R)

        H = np.cross(R, V)
        j = -H / np.linalg.norm(H)
        j = j / np.linalg.norm(j)
        i = np.cross(j, k)
        i = i / np.linalg.norm(i)

        L_oi = np.vstack((i, j, k))
        return L_oi
    def Inertial2Relative(self,R1, V1, R2, V2):
        mu = 3.986005e5
        rad_to_deg = 180 / np.pi
        deg_to_rad = np.pi / 180

        a1 = -mu * R1 / (np.linalg.norm(R1)) ** 3
        a2 = -mu * R2 / (np.linalg.norm(R2)) ** 3

        i = R1 / np.linalg.norm(R1)
        h1 = np.cross(R1, V1)
        k = h1 / np.linalg.norm(h1)
        j = np.cross(k, i)

        OMG1 = h1 / (np.linalg.norm(R1)) ** 2

        deltaR = R2 - R1
        deltaV = (V2 - V1) - np.cross(OMG1, deltaR)

        QXx = np.array([i, j, k])
        deltaR = np.dot(QXx, deltaR)
        deltaV = np.dot(QXx, deltaV)

        delta = np.hstack((deltaR, deltaV))

        return delta, QXx
    # 推力一定计算燃料消耗率 推力T
    def kCal(self,T):
        g0 = 9.8
        Isp = 3000
        k = -T / g0 / Isp
        return k
    
    def LLA2ICRS(self,Mjd, lonlan):

        f = 1 / 298.257223563  # 地球扁率，无量纲
        R_p = (1 - f) * globals.r_E  # 地球极半径
        e1 = np.sqrt(globals.r_E * globals.r_E - R_p * R_p) / globals.r_E
        lon = lonlan[0]
        lan = lonlan[1]
        alt = lonlan[2]
        temp = globals.r_E / np.sqrt(1 - e1 * e1 * np.sin(np.radians(lan)) * np.sin(np.radians(lan)))
        pos = np.zeros(3)
        pos[0] = (temp + alt) * np.cos(np.radians(lan)) * np.cos(np.radians(lon))
        pos[1] = (temp + alt) * np.cos(np.radians(lan)) * np.sin(np.radians(lon))
        pos[2] = (temp * (1 - e1 * e1) + alt) * np.sin(np.radians(lan))

        E = ICRS2ITRS(Mjd)  # You need to implement or import ICRS2ITRS function
        pos = np.dot(E.T, pos)

        return pos
    
    def lonlattablegen(self,rvdata, date):
        lonlattable = []
        Predict = OrbitPredict()
        Core = OrbitCore()
        for i in range(len(rvdata[0])):
            lonlat_record, E = self.rv2lonlat(rvdata[:3, i], Core.AddTime(date, 60 * i))
            lon, lat, _ = Predict.Geodetic(np.dot(E, rvdata[:3, i]))
            lonlat_T_record = [lon, lat, 60 * i]
            lonlattable.append(lonlat_T_record)

            if i != 0:
                if lonlattable[0][i] - lonlattable[0][i - 1] > 90:
                    lonlattable[0][i] = lonlat_record[0] - 180
                elif lonlattable[0][i] - lonlattable[0][i - 1] < -90:
                    lonlattable[0][i] = lonlat_record[0] + 180

            if lonlattable[0][i] > 180:
                lonlattable[0][i] -= 360
            elif lonlattable[0][i] < -180:
                lonlattable[0][i] += 360

        nodel = 1
        return lonlattable
    def rv2lonlat(self,rv, date):
        # Make sure to define or import Mjday and ICRS2ITRS functions
        Mjd = Mjday(date[0], date[1], date[2], date[3], date[4], date[5])
        E = ICRS2ITRS(Mjd)
        rf = np.dot(E, rv[:3])  # Earth-fixed position

        lon = np.degrees(np.arctan2(rf[1], rf[0]))
        lat = np.degrees(np.arctan2(rf[2], np.linalg.norm(rf)))

        lonlat = np.array([lon, lat])
        return lonlat, E
    #输入脉冲矢量
    #输出 Azimuth-俯仰角，Elevation-方位角
    def ThrustAngleCal2(self,vec):
        z = np.array([0, 0, 1])
        x = np.array([1, 0, 0])

        Elevation = np.degrees(np.arccos(np.dot(vec, z) / np.linalg.norm(vec)))

        if Elevation < 90:
            Elevation = 90 - Elevation
        else:
            Elevation = -Elevation + 90

        vec_z = np.array([0, 0, np.dot(vec, z)])
        vec_xy = vec - vec_z

        Azimuth = np.degrees(np.arccos(np.dot(x, vec_xy) / np.linalg.norm(vec_xy)))

        return Azimuth, Elevation
    def xingxiadian(self,rv,date):
        # 使用自定义函数 rv2latlon 计算星下点经纬度
        # 使用自定义函数 rv2lonlat 计算星下点经纬度
        lonlat_c, E2 = self.rv2lonlat(rv, date)
        Predict = OrbitPredict()
        _, lat, _ = Predict.Geodetic(np.dot(E2, rv[:3]))
        lon = lonlat_c[0]
        lat = lat
        return lon,lat
