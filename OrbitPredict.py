import numpy as np
import globals
import numpy as np
import pysat
from datetime import datetime
from nrlmsise00 import msise_model
from ToolFunction import *
import math
from OrbitCore import *
class OrbitPredict:
    #  计算大气摄动加速度
    #  输入参数：
    #  PosVel - 卫星位置速度       dens - 大气密度（不给定时采用指数模型计算）
    #  输出参数：
    #  a - 大气摄动产生的三轴加速度
    def AccelDrag(self,PosVel, dens=None):
        r = np.sqrt(PosVel[0] ** 2 + PosVel[1] ** 2 + PosVel[2] ** 2)

        if dens is None:
            dens = self.ComputeDensity(r)

        w = 0.7292e-4
        v_rel = np.array([PosVel[3] + w * PosVel[1],
                          PosVel[4] - w * PosVel[0],
                          PosVel[5]])

        a = -0.5 * globals.CD * globals.s_m * 1e3 * dens * np.linalg.norm(v_rel) * v_rel
        return a

    #  计算大气密度（指数模型）
    #  输入：r - 卫星地心距   输出：den - 大气密度
    def ComputeDensity(self,r):
        p0 = 3.6e-10
        H0 = 37.4
        r0 = 6408.4
        H = H0 + 0.12 * (r - r0)
        den = p0 * np.exp(-(r - r0) / H)
        return den

    #  输入参数：
    #  Mjd - 儒略日(UTC)         r - 地心惯性系下的坐标
    #  deg - 非球形摄动阶数      E - 惯性系到地固系的转换矩阵
    #  输出参数：
    #  a - 非球形摄动产生的三轴加速度
    def AccelHarmonic_ElasticEarth(self,r, deg):

        r_bf = np.matmul(globals.E, r)  # 将卫星位置矢量 r 由地惯系 ICRF 转换为地固系 ITRS
        lon, latgc, d = self.CalcPolarAngles(r_bf)  # 计算地心纬度、地心经度、地心距

        if lon > np.pi:
            lon = lon - 2 * np.pi

        pnm, dpnm = self.Legendre(deg, deg, latgc)
        dUdr = 0
        dUdlat = 0
        dUdlon = 0
        q1 = 0
        q2 = 0
        q3 = 0

        for n in range(2, deg + 1):
            b1 = (-globals.GM_Earth / d ** 2) * (globals.r_E / d) ** n * (n + 1)
            b2 = (globals.GM_Earth / d) * (globals.r_E / d) ** n
            b3 = (globals.GM_Earth / d) * (globals.r_E / d) ** n

            for m in range(n + 1):
                ml = m * lon
                q1 = q1 + pnm[n][m] * (globals.S[n][m] * np.sin(ml) + globals.C[n][m] * np.cos(ml))
                q2 = q2 + dpnm[n][m] * (globals.S[n][m] * np.sin(ml) + globals.C[n][m] * np.cos(ml))
                q3 = q3 + m * pnm[n][m] * (globals.S[n][m] * np.cos(ml) - globals.C[n][m] * np.sin(ml))

            dUdr = dUdr + q1 * b1  # U 对地心距 r 的偏导数
            dUdlat = dUdlat + q2 * b2  # U 对地心精度 φ 的偏导数
            dUdlon = dUdlon + q3 * b3  # U 对地心纬度 λ 的偏导数
            q1 = 0
            q2 = 0
            q3 = 0

        x = r_bf[0]
        y = r_bf[1]
        z = r_bf[2]
        xy2 = x ** 2 + y ** 2
        xyn = np.sqrt(xy2)

        R_sph2rec = np.zeros((3, 3))
        R_sph2rec[:, 0] = [x, y, z] / d
        R_sph2rec[:, 1] = [-x * z / xyn, -y * z / xyn, xyn] / d ** 2
        R_sph2rec[:, 2] = [-y, x, 0] / xy2

        a_bf = np.matmul(R_sph2rec, [dUdr, dUdlat, dUdlon])
        a = np.matmul(np.transpose(globals.E), a_bf)
        return a

    #  计算地心经纬度
    # 输入：r_bf - 地固系下的坐标
    #  输出：lon - 地心维度   latgc - 地心经度    d - 地心距

    def CalcPolarAngles(r_bf):
        rhoSqr = r_bf[0] * r_bf[0] + r_bf[1] * r_bf[1]
        d = np.sqrt(rhoSqr + r_bf[2] * r_bf[2])

        if r_bf[0] == 0 and r_bf[1] == 0:
            lon = 0
        else:
            lon = np.arctan2(r_bf[1], r_bf[0])

        if lon < 0:
            lon = lon + 2 * np.pi

        rho = np.sqrt(rhoSqr)
        if r_bf[2] == 0 and rho == 0:
            latgc = 0
        else:
            latgc = np.arctan2(r_bf[2], rho)

        return lon, latgc, d

    # 计算勒让德参数表

    def Legendre(self,n, m, fi):
        sf = np.sin(fi)
        cf = np.cos(fi)

        pnm = np.zeros((n+1, m+1))
        dpnm = np.zeros((n+1, m+1))

        pnm[0, 0] = 1
        dpnm[0, 0] = 0
        pnm[1, 1] = np.sqrt(3) * cf
        dpnm[1, 1] = -np.sqrt(3) * sf

        for i in range(2, n+1):
            pnm[i, i] = np.sqrt((2*i+1)/(2*i)) * cf * pnm[i-1, i-1]
            dpnm[i, i] = np.sqrt((2*i+1)/(2*i)) * (cf * dpnm[i-1, i-1] - sf * pnm[i-1, i-1])

        for i in range(1, n+1):
            pnm[i, i-1] = np.sqrt(2*i+1) * sf * pnm[i-1, i-1]
            dpnm[i, i-1] = np.sqrt(2*i+1) * (cf * pnm[i-1, i-1] + sf * dpnm[i-1, i-1])

        j = 0
        k = 2
        while True:
            for i in range(k, n+1):
                pnm[i, j] = np.sqrt((2*i+1)/((i-j)*(i+j))) * ((np.sqrt(2*i-1)*sf*pnm[i-1, j]) - (np.sqrt(((i+j-1)*(i-j-1))/(2*i-3))*pnm[i-2, j]))
                dpnm[i, j] = np.sqrt((2*i+1)/((i-j)*(i+j))) * ((np.sqrt(2*i-1)*sf*dpnm[i-1, j]) + (np.sqrt(2*i-1)*cf*pnm[i-1, j]) - (np.sqrt(((i+j-1)*(i-j-1))/(2*i-3))*dpnm[i-2, j]))

            j += 1
            k += 1
            if j > m:
                break

        return pnm, dpnm
    def AccelThird(self,Pos, Mjd):
        r_Earth, r_moon, r_sun,  r_Mars = self.JPL_Eph_DE405(Mjd + 66.184 / 86400)
        r_moon = r_moon / 1000
        r_sun = r_sun / 1000

        erthToSun = (r_sun[0] ** 2 + r_sun[1] ** 2 + r_sun[2] ** 2) ** 1.5
        earthToMoon = (r_moon[0] ** 2 + r_moon[1] ** 2 + r_moon[2] ** 2) ** 1.5

        R_sun = ((Pos[0] - r_sun[0]) ** 2 + (Pos[1] - r_sun[1]) ** 2 + (Pos[2] - r_sun[2]) ** 2) ** 1.5
        R_moon = ((Pos[0] - r_moon[0]) ** 2 + (Pos[1] - r_moon[1]) ** 2 + (Pos[2] - r_moon[2]) ** 2) ** 1.5

        a_sun = -globals.GM_Sun * ((Pos - r_sun) / R_sun + r_sun / erthToSun)
        a_moon = -globals.GM_Moon * ((Pos - r_moon) / R_moon + r_moon / earthToMoon)

        a = a_sun + a_moon
        return a
    
    def AddTime(self,Time, addsecond):
        if addsecond != 0:
            logic = int(addsecond / abs(addsecond))  # Sign of time forward/backward
            addSecond = round(abs(addsecond))
            addMinute = addSecond // 60
            addSecond -= addMinute * 60
            addHour = addMinute // 60
            addMinute -= addHour * 60
            addDay = addHour // 24
            addHour -= addDay * 24
            newTime = Time + logic * [0, 0, addDay, addHour, addMinute, addSecond]
            newTime = self.AmendTime(newTime)
        else:
            newTime = Time
        return newTime

    def AmendTime(self,startTime):
        dayOfMonth = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Number of days in each month (January to December)
        if startTime[1] % 4 == 0:
            dayOfMonth[2] = dayOfMonth[1] + 1

        if startTime[5] >= 60:  # Adjust seconds
            startTime[5] -= 60
            startTime[4] += 1
        elif startTime[5] < 0:
            startTime[5] = 60 + startTime[5]
            startTime[4] -= 1

        if startTime[4] >= 60:  # Adjust minutes
            startTime[4] -= 60
            startTime[3] += 1
        elif startTime[4] < 0:
            startTime[4] = 60 + startTime[4]
            startTime[3] -= 1

        if startTime[3] >= 24:  # Adjust hours
            startTime[3] -= 24
            startTime[2] += 1
        elif startTime[3] < 0:
            startTime[3] = 24 + startTime[3]
            startTime[2] -= 1

        if startTime[2] > dayOfMonth[startTime[1] + 1]:  # Adjust days
            startTime[2] -= dayOfMonth[startTime[1] + 1]
            startTime[1] += 1
        elif startTime[2] <= 0:
            startTime[2] = dayOfMonth[startTime[1]] + startTime[2]
            startTime[1] -= 1

        if startTime[1] >= 13:  # Adjust months
            startTime[1] -= 12
            startTime[0] += 1
        elif startTime[1] <= 0:
            startTime[1] = 12 + startTime[1]
            startTime[0] -= 1

        newTime = startTime
        return newTime
    def ComputeDensity_HPOP(self,PosVel, startTime, addMjd):
        lon, lat, height = self.Geodetic(np.dot(globals.E, PosVel[:3]))

        if height >= 1000000:  # Default atmospheric density is 0 above 1000 km
            dens = 0
            return dens
        else:
            startTime = self.AddTime(startTime, addMjd * 86400)

        # Calculate the day of the year
        year = startTime[0]
        month = startTime[1]
        day = startTime[2]
        hour= startTime[3]
        minute= startTime[4]
        second= startTime[5]

        start_time_indices = [i for i, row in enumerate(globals.spaceWheather) if
                              row[0] == startTime[0] and row[1] == startTime[1] and row[2] == startTime[2]]

        num = start_time_indices[0]  # Use the first matching index

        # Extract values from spaceWheather
        ap = globals.spaceWheather[num][ 3]
        f107A = globals.spaceWheather[num][ 5]
        f107 = globals.spaceWheather[num - 1][ 4]

        # Create a datetime object for the start time
        start_time = datetime(year, month, day, hour, minute, second)

        height = height * 0.001#从m转化为km
        f107A = f107A
        f107 = f107
        #花了两天，msise_model替换atmosnrlmsise00
        #测试例子
        #Matlab：[~,rho] = atmosnrlmsise00(400000, 60, -70, startTime(1), dayOfYear, second,150, 150,4)
        #python：msise_model(datetime(2009, 6, 21, 8, 3, 20), 400, 60, -70, 150, 150, 4)
        rho,_ = msise_model(start_time,height,lat,lon,f107A,f107,ap)
        dens = rho[5]*1000
        return dens
    def Geodetic(self,r):
        f = 1 / 298.257223563

        epsRequ = np.finfo(float).eps * globals.r_E * 1000  # 收敛准则
        e2 = f * (2 - f)  # Square of eccentricity

        X = r[0] * 1000  # Cartesian coordinates
        Y = r[1] * 1000
        Z = r[2] * 1000
        rho2 = X * X + Y * Y  # Square of distance from z-axis

        # Check validity of input data
        if (np.linalg.norm(r)== 0):
            print('Invalid input in Geodetic constructor')
            lon = 0
            lat = 0
            h = -globals.r_E
            return lon, lat, h

        # Iteration
        dZ = e2 * Z

        while True:
            ZdZ = Z + dZ
            Nh = math.sqrt(rho2 + ZdZ * ZdZ)
            SinPhi = ZdZ / Nh  # Sine of geodetic latitude
            N = globals.r_E * 1000 / math.sqrt(1 - e2 * SinPhi * SinPhi)
            dZ_new = N * e2 * SinPhi
            if math.isclose(abs(dZ - dZ_new), 0, abs_tol=epsRequ):
                break
            dZ = dZ_new

        # Longitude, latitude, altitude
        lon = math.atan2(Y, X) * 180 / math.pi
        lat = math.atan2(ZdZ, math.sqrt(rho2)) * 180 / math.pi
        h = Nh - N

        return lon, lat, h
    def J2Cal(self,t, RV, Thrust_f, deg):
        GM_Earth, J2, r_E = globals.GM_Earth, globals.J2, globals.r_E
        Azimuth = deg[0]
        Elevation = deg[1]

        x = RV[0]
        y = RV[1]
        z = RV[2]
        Tool = ToolFunction()
        RotMat = np.transpose(Tool.Inertial2Orbit(RV))
        vec = np.array([np.cos(np.radians(Elevation)) * np.cos(np.radians(Azimuth)),
                        np.cos(np.radians(Elevation)) * np.sin(np.radians(Azimuth)),
                        np.sin(np.radians(Elevation))])

        r = np.linalg.norm(RV[0:3])
        dx = -GM_Earth * x / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (1 - 5 * z ** 2 / r ** 2)) + Thrust_f * np.dot(RotMat[0], vec)
        dy = -GM_Earth * x / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (1 - 5 * z ** 2 / r ** 2)) * (y / x) + Thrust_f * np.dot(RotMat[1], vec)
        dz = -GM_Earth * z / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (3 - 5 * z ** 2 / r ** 2)) + Thrust_f * np.dot(RotMat[2], vec)

        drv = [RV[3], RV[4], RV[5], dx, dy, dz]
        return drv
    def J2Cal_rvm(self,t, RVm, Thrust_f, deg, k):
        GM_Earth, J2, r_E = globals.GM_Earth, globals.J2, globals.r_E
        Azimuth = deg[0]
        Elevation = deg[1]
        x = RVm[3]
        y = RVm[4]
        z = RVm[5]
        m = RVm[6]

        Tool = ToolFunction()
        # 脉冲轨道系定向，需要求解对应时刻惯性系的脉冲
        RotMat = np.transpose(Tool.Inertial2Orbit(RVm))
        vec = np.array([np.cos(np.radians(Elevation)) * np.cos(np.radians(Azimuth)),
                        np.cos(np.radians(Elevation)) * np.sin(np.radians(Azimuth)),
                        np.sin(np.radians(Elevation))])
        accel = Thrust_f / m / 1000  # 根据推力计算加速度大小
        r = np.linalg.norm(RVm[0:3])
        dx = accel * np.dot(RotMat[0], vec) - GM_Earth * RVm[0] / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (1 - 5 * RVm[2] ** 2 / r ** 2))
        dy = accel * np.dot(RotMat[1], vec) - GM_Earth * RVm[0] / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (1 - 5 * RVm[2] ** 2 / r ** 2)) * (RVm[1] / RVm[0])
        dz = accel * np.dot(RotMat[2], vec) - GM_Earth * RVm[2] / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (3 - 5 * RVm[2] ** 2 / r ** 2))

        drvm = [x, y, z, dx, dy, dz, k]
        return drvm
    def J2Orbit(self,coe0, T):
        global GM_Earth, J2, r_E, rad2deg
        GM_Earth, J2, r_E, rad2deg = globals.GM_Earth, globals.J2, globals.r_E, globals.rad2deg
        a = coe0[0]
        e = coe0[1]
        i = coe0[2]

        q = self.true2Mean(coe0[1], coe0[5])

        p = a * (1 - e ** 2)
        n = np.sqrt(GM_Earth / a ** 3)
        k1 = -1.5 * J2 * (r_E / p) ** 2 * n * np.cos(np.deg2rad(i))
        k2 = 1.5 * J2 * (r_E / p) ** 2 * n * (2 - 2.5 * np.sin(np.deg2rad(i)) ** 2)
        k3 = n + 1.5 * J2 * (r_E / p) ** 2 * n * (1 - 1.5 * np.sin(np.deg2rad(i)) ** 2) * np.sqrt(1 - e ** 2)

        coe = coe0.copy()
        coe[3] = self.amendDeg(coe0[3] + k1 * T * rad2deg, '0 - 360')
        coe[4] = self.amendDeg(coe0[4] + k2 * T * rad2deg, '0 - 360')
        coe[5] = self.amendDeg(q + k3 * T * rad2deg, '0 - 360')
        coe[5] = self.mean2True(e, coe[5])

        return coe
    
    def J2OrbitRV(self,rv0, T):
        GM_Earth = globals.GM_Earth
        Core = OrbitCore()
        coe0 = Core.State_rv_2_Orbit_Element(rv0[:3], rv0[3:])
        if coe0[1] < 0 or coe0[1] > 1:
            raise ValueError('参数不符合要求')

        coe = Core.J2Orbit(coe0, T)
        r, v = Core.Orbit_Element_2_State_rv(coe, GM_Earth)
        rv = np.concatenate((r, v))
        return rv
    
    
    def Cheb3D(self,t, N, Ta, Tb, Cx, Cy, Cz):
        # Check validity
        if t < Ta or t > Tb:
            raise ValueError('ERROR: Time out of range in cheb3d')
        # Clenshaw algorithm
        tau = (2 * t - Ta - Tb) / (Tb - Ta)

        f1 = np.zeros(3)
        f2 = np.zeros(3)

        for i in range(N, 1, -1):
            old_f1 = f1
            f1 = 2 * tau * f1 - f2 + [Cx[i - 1], Cy[i - 1], Cz[i - 1]]  # 注意Python中列表索引是从0开始的，所以索引需要减1
            f2 = old_f1

        ChebApp = tau * f1 - f2 + np.array([Cx[0], Cy[0], Cz[0]])

        return ChebApp



    def JPL_Eph_DE405(self,Mjd_UTC):


        JD = Mjd_UTC + 2400000.5

        for i in range(1147):
            if (globals.PC[i][ 0] <= JD and JD <= globals.PC[i][ 1]):
                PCtemp = globals.PC[i][ :]

        t1 = PCtemp[0] - 2400000.5  # MJD at start of interval

        dt = Mjd_UTC - t1

        temp = np.arange(231, 271, 13)
        Cx_Earth = PCtemp[temp[0] - 1 : temp[1] - 1]
        Cy_Earth = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz_Earth = PCtemp[temp[2] - 1:temp[3] - 1]
        temp = temp + 39
        Cx = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz = PCtemp[temp[2] - 1:temp[3] - 1]
        Cx_Earth = np.concatenate([Cx_Earth, Cx])
        Cy_Earth = np.concatenate([Cy_Earth, Cy])
        Cz_Earth = np.concatenate([Cz_Earth, Cz])
        if (0 <= dt and dt <= 16):
            j = 0
            Mjd0 = t1
        elif (16 < dt and dt <= 32):
            j = 1
            Mjd0 = t1 + 16 * j
        r_Earth = 1e3 * self.Cheb3D(Mjd_UTC, 13, Mjd0, Mjd0 + 16, Cx_Earth[13 * j :13 * j + 13],
                              Cy_Earth[13 * j :13 * j + 13], Cz_Earth[13 * j :13 * j + 13])
        temp = np.arange(441, 481, 13)
        Cx_Moon = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy_Moon = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz_Moon = PCtemp[temp[2] - 1:temp[3] - 1]
        for i in range(7):
            temp = temp + 39
            Cx = PCtemp[temp[0] - 1:temp[1] - 1]
            Cy = PCtemp[temp[1] - 1:temp[2] - 1]
            Cz = PCtemp[temp[2] - 1:temp[3] - 1]

            Cx_Moon = np.concatenate([Cx_Moon, Cx])
            Cy_Moon = np.concatenate([Cy_Moon, Cy])
            Cz_Moon = np.concatenate([Cz_Moon, Cz])
        if (0 <= dt and dt <= 4):
            j = 0
            Mjd0 = t1
        elif (4 < dt and dt <= 8):
            j = 1
            Mjd0 = t1 + 4 * j
        elif (8 < dt and dt <= 12):
            j = 2
            Mjd0 = t1 + 4 * j
        elif (12 < dt and dt <= 16):
            j = 3
            Mjd0 = t1 + 4 * j
        elif(16 < dt and dt <= 20):
            j = 4
            Mjd0 = t1 + 4 * j
        elif(20 < dt and dt <= 24):
            j = 5
            Mjd0 = t1 + 4 * j
        elif(24 < dt and dt <= 28):
            j = 6
            Mjd0 = t1 + 4 * j
        elif(28 < dt and dt <= 32):
            j = 7
            Mjd0 = t1 + 4 * j
        r_Moon = 1e3 * self.Cheb3D(Mjd_UTC, 13, Mjd0, Mjd0 + 4, Cx_Moon[13 * j :13 * j + 13],
                              Cy_Moon[13 * j :13 * j + 13], Cz_Moon[13 * j :13 * j + 13])

        temp = np.arange(753, 787, 11)
        Cx_Sun = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy_Sun = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz_Sun = PCtemp[temp[2] - 1:temp[3] - 1]
        temp = temp + 33
        Cx = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz = PCtemp[temp[2] - 1:temp[3] - 1]
        Cx_Sun = np.concatenate([Cx_Sun , Cx])
        Cy_Sun = np.concatenate([Cy_Sun , Cy])
        Cz_Sun = np.concatenate([Cz_Sun , Cz])

        if (0 <= dt and dt <= 16):
            j = 0
            Mjd0 = t1
        elif (16 < dt and dt <= 32):
            j = 1
            Mjd0 = t1 + 16 * j

        r_Sun = 1e3 * self.Cheb3D(Mjd_UTC, 11, Mjd0, Mjd0 + 16, Cx_Sun[11 * j :11 * j + 11],
                            Cy_Sun[11 * j :11 * j + 11], Cz_Sun[11 * j :11 * j + 11])

        temp = np.arange(3, 46, 14)
        Cx_Mercury = PCtemp[temp[0] - 1 :temp[1] - 1]
        Cy_Mercury = PCtemp[temp[1] - 1 :temp[2] - 1]
        Cz_Mercury = PCtemp[temp[2] - 1 :temp[3] - 1]
        for i in range(3):
            temp = temp + 42
            Cx = PCtemp[temp[0] - 1 :temp[1] - 1]
            Cy = PCtemp[temp[1] - 1 :temp[2] - 1]
            Cz = PCtemp[temp[2] - 1 :temp[3] - 1]
            Cx_Mercury = np.concatenate([Cx_Mercury, Cx])
            Cy_Mercury = np.concatenate([Cy_Mercury, Cy])
            Cz_Mercury = np.concatenate([Cz_Mercury, Cz])
        if (0 <= dt and dt <= 8):
            j = 0
            Mjd0 = t1
        elif (8 < dt and dt <= 16):
            j = 1
            Mjd0 = t1 + 8 * j
        elif (16 < dt and dt <= 24):
            j = 2
            Mjd0 = t1 + 8 * j
        elif (24 < dt and dt <= 32):
            j = 3
            Mjd0 = t1 + 8 * j

        r_Mercury = 1e3 * self.Cheb3D(Mjd_UTC, 14, Mjd0, Mjd0 + 8, Cx_Mercury[14 * j :14 * j + 14],
                                Cy_Mercury[14 * j :14 * j + 14], Cz_Mercury[14 * j :14 * j + 14])

        temp = np.arange(171, 202, 10)
        Cx_Venus = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy_Venus = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz_Venus = PCtemp[temp[2] - 1:temp[3] - 1]

        temp = temp + 30
        Cx = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz = PCtemp[temp[2] - 1:temp[3] - 1]
        Cx_Venus = np.concatenate([Cx_Venus, Cx])
        Cy_Venus = np.concatenate([Cy_Venus, Cy])
        Cz_Venus = np.concatenate([Cz_Venus, Cz])
        if (0 <= dt and dt <= 16):
            j = 0
            Mjd0 = t1
        elif (16 < dt and dt <= 32):
            j = 1
            Mjd0 = t1 + 16 * j

        r_Venus = 1e3 * self.Cheb3D(Mjd_UTC, 10, Mjd0, Mjd0 + 16, Cx_Venus[10 * j :10 * j + 10],
                              Cy_Venus[10 * j :10 * j + 10], Cz_Venus[10 * j :10 * j + 10])

        temp = np.arange(309, 343, 11)
        Cx_Mars = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy_Mars = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz_Mars = PCtemp[temp[2] - 1:temp[3] - 1]
        j = 0
        Mjd0 = t1
        r_Mars = 1e3 * self.Cheb3D(Mjd_UTC, 11, Mjd0, Mjd0 + 32, Cx_Mars[11 * j :11 * j + 11],
                              Cy_Mars[11 * j :11 * j + 11], Cz_Mars[11 * j :11 * j + 11])

        temp = np.arange(342, 367, 8)
        Cx_Jupiter = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy_Jupiter = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz_Jupiter = PCtemp[temp[2] - 1:temp[3] - 1]
        j = 0
        Mjd0 = t1

        r_Jupiter = 1e3 * self.Cheb3D(Mjd_UTC, 8, Mjd0, Mjd0 + 32, Cx_Jupiter[8 * j :8 * j + 8],
                                Cy_Jupiter[8 * j :8 * j + 8], Cz_Jupiter[8 * j :8 * j + 8])

        EMRAT = 81.3005600000000044
        EMRAT1 = 1 / (1 + EMRAT)
        r_Earth = r_Earth - EMRAT1 * r_Moon
        r_Sun = -r_Earth + r_Sun
        r_Mars = -r_Earth + r_Mars

        return r_Earth, r_Moon, r_Sun,  r_Mars

    # Orbit Prediction function
    def OrbitPrediction(self,x0, time, h, model=[1, 1], integrator='RK4', startTime=None):
        x = np.array(x0)
        xt = np.array(x0)
        xk = np.zeros(4)
        k4 = np.zeros((4, 6))
        k7 = np.zeros((13, 6))
        M_T = 1 / 86400
        Mjd = 0
        h = h * np.sign(time)  # Determine step size based on the direction of propagation
        param_k = np.zeros((13, 12))
        dens = 0
        if (h!=0 and time % h == 0):
            finalStep = h
            num = int(time / h)
        elif (h==0 and time == 0):
            finalStep = np.nan
            num = np.nan
        else:
            finalStep = time % h
            num = int(time / h) + 1

        if startTime is not None:
            year, mon, day, hour, minute, sec = startTime
            Mjd0 = Mjday(year, mon, day, hour, minute, sec)
            Mjd = Mjd0
            E = ICRS2ITRS(Mjd)
            dens = self.ComputeDensity_HPOP(x0, startTime, 0)

        if integrator == 'RK4':
            param_k = np.array([0, 1 / 2, 1 / 2, 1])
            param_t = np.array([1, 2, 2, 1])
        elif integrator == 'RK7':
            param_t = np.array([0.0, 2.0 / 27.0, 1.0 / 9.0, 1.0 / 6.0, 5.0 / 12.0, 1.0 / 2.0, 5.0 / 6.0,
                                1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 0.0, 1.0])
            param_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0,
                                9.0 / 280.0, 9.0 / 280.0, 0.0, 41.0 / 840.0, 41.0 / 840.0])


            param_k[0,:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            param_k[1,:] = [2.0 / 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            param_k[2,:] = [1.0 / 36.0, 1.0 / 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            param_k[3,:] = [1.0 / 24.0, 0.0, 1.0 / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            param_k[4,:] = [5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            param_k[5,:] = [1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            param_k[6,:] = [-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0, 125.0 / 54.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0]
            param_k[7,:] = [31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0, 13.0 / 900.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            param_k[8,:] = [2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0, 67.0 / 90.0, 3.0, 0.0, 0.0, 0.0, 0.0]
            param_k[9,:] = [-91.0 / 108, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0, 311.0 / 54.0, -19.0 / 60.0, 17.0 / 6.0,
                            -1.0 / 12.0, 0.0, 0.0, 0.0]
            param_k[10,:] = [2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -301.0 / 82.0, 2133.0 / 4100.0,
                            45.0 / 82.0, 45.0 / 164.0, 18.0 / 41.0, 0.0, 0.0]
            param_k[11,:] = [3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0, -3.0 / 41.0, 3.0 / 41.0,
                            6.0 / 41.0, 0.0, 0.0]
            param_k[12,:] = [-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -289.0 / 82.0, 2193.0 / 4100.0,
                            51.0 / 82.0, 33.0 / 164.0, 12.0 / 41.0, 0.0, 1.0]
        if integrator == 'RK4':
            if(np.isnan(num)==True):
                for i in range(1):
                    if i == num - 1:
                        h = finalStep

                    for j in range(4):
                        xk = np.zeros(6)
                        for k in range(6):
                            if j > 0:
                                xk[k] = param_k[j] * h * k4[j - 1, k]

                        k4[j, 0] = x[3] + xk[3]
                        k4[j, 1] = x[4] + xk[4]
                        k4[j, 2] = x[5] + xk[5]

                        a = np.array([0, 0, 0])
                        if startTime is None:  # Low-accuracy propagation
                            a = self.Accel(x, xk, model)
                        elif startTime is not None and globals.orbitModel == "LPOP":
                            a = self.Accel(x, xk, model, dens)

                        elif startTime is not None and globals.orbitModel == "HPOP":  # High-accuracy propagation
                            globals.E = ICRS2ITRS(Mjd + M_T * h * param_k[j])
                            # 二体引力
                            r = np.sqrt((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2) ** 1.5
                            a = -globals.GM_Earth / r * np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                            # 非球型摄动
                            if model[0] == 1:  # Non-spherical perturbations
                                R = np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                                a += self.AccelHarmonic_ElasticEarth(R, globals.deg)
                            # 大气摄动
                            if model[1] == 1:  # Atmospheric drag
                                dens = self.ComputeDensity_HPOP(x + xk, startTime, Mjd - Mjd0 + M_T * h * param_k[j])
                                a += self.AccelDrag(x + xk, dens)
                            # 三体摄动
                            if len(model) == 3 and model[2] == 1:  # Third-body perturbations
                                a += self.AccelThird(x[:3] + xk[:3], Mjd + M_T * h * param_k[j])

                        k4[j, 3] = a[0]
                        k4[j, 4] = a[1]
                        k4[j, 5] = a[2]

                    Mjd += M_T * h
                    x += h * np.dot(k4.T, param_t) / 6
                    xt = np.hstack((xt, x))
            else:
                for i in range(num):
                    if i == num - 1:
                        h = finalStep

                    for j in range(4):
                        xk = np.zeros(6)
                        for k in range(6):
                            if j > 0:
                                xk[k] = param_k[j] * h * k4[j - 1, k]

                        k4[j, 0] = x[3] + xk[3]
                        k4[j, 1] = x[4] + xk[4]
                        k4[j, 2] = x[5] + xk[5]

                        a = np.array([0, 0, 0])
                        if startTime is None:  # Low-accuracy propagation
                            a = self.Accel(x, xk, model)
                        elif startTime is not None and globals.orbitModel == "LPOP":
                            a = self.Accel(x, xk, model, dens)

                        elif startTime is not None and globals.orbitModel == "HPOP":  # High-accuracy propagation
                            globals.E = ICRS2ITRS(Mjd + M_T * h * param_k[j])
                            # 二体引力
                            r = np.sqrt((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2) ** 1.5
                            a = -globals.GM_Earth / r * np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                            # 非球型摄动
                            if model[0] == 1:  # Non-spherical perturbations
                                R = np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                                a += self.AccelHarmonic_ElasticEarth(R, globals.deg)
                            # 大气摄动
                            if model[1] == 1:  # Atmospheric drag
                                dens = self.ComputeDensity_HPOP(x + xk, startTime, Mjd - Mjd0 + M_T * h * param_k[j])
                                a += self.AccelDrag(x + xk, dens)
                            # 三体摄动
                            if len(model) == 3 and model[2] == 1:  # Third-body perturbations
                                a += self.AccelThird(x[:3] + xk[:3], Mjd + M_T * h * param_k[j])

                        k4[j, 3] = a[0]
                        k4[j, 4] = a[1]
                        k4[j, 5] = a[2]

                    Mjd += M_T * h
                    x += h * np.dot(k4.T, param_t) / 6
                    xt = np.hstack((xt, x))

        elif integrator == "RK7":
            for i in range(num):
                if i == num - 1:
                    h = finalStep

                for j in range(13):
                    xk = np.array([0,0,0,0,0,0])
                    xk = xk.astype('float64')
                    if j > 0:
                        for n in range(j):
                            xk += param_k[j,n]  * k7[n,:]
                    k7[j, 0] = h * (x[3] + xk[3])
                    k7[j, 1] = h * (x[4] + xk[4])
                    k7[j, 2] = h * (x[5] + xk[5])
                    a = np.array([0, 0, 0])
                    if startTime is None:  # Low-accuracy propagation
                        a = h * self.Accel(x, xk, model)
                    elif startTime is not None and globals.orbitModel=="LPOP":
                        a = h * self.Accel(x, xk, model, dens)

                    elif startTime is not None and globals.orbitModel=="HPOP":  # High-accuracy propagation
                        globals.E = ICRS2ITRS(Mjd + M_T * h * param_t[j])
                        # 二体引力
                        #这个轨道部分有点问题
                        r = ((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2)**1.5
                        # r = np.sqrt((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2)
                        a = -h * globals.GM_Earth / r * np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                        # 非球型摄动
                        if model[0] == 1:  # Non-spherical perturbations
                            R = np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                            a += h * self.AccelHarmonic_ElasticEarth(R, globals.deg)
                        # 大气摄动
                        if model[1] == 1:  # Atmospheric drag
                            dens = self.ComputeDensity_HPOP(x + xk, startTime, Mjd - Mjd0 )
                            a += h * self.AccelDrag(x + xk, dens)
                        # 三体摄动
                        if len(model) == 3 and model[2] == 1:  # Third-body perturbations
                            a += h * self.AccelThird(x[:3] + xk[:3], Mjd + M_T * h * param_t[j])

                    k7[j, 3] = a[0]
                    k7[j, 4] = a[1]
                    k7[j, 5] = a[2]

                Mjd += M_T * h
                x += k7.transpose() @ param_c
                # xt = np.tile(xt, x)
                xt = np.vstack((xt, x))
        return x,xt


    def Accel(self,x, xk, model, dens=None):

        r = np.sqrt((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2)
        a = np.zeros(3)

        if model[0] == 0:
            a = -globals.GM_Earth / r ** 3 * np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
        elif model[0] == 1:
            a[0] = -globals.GM_Earth * (x[0] + xk[0]) / r ** 3 * (
                        1 + 1.5 * globals.J2 * (globals.r_E / r) ** 2 * (1 - 5 * (x[2] + xk[2]) ** 2 / r ** 2))
            a[1] = a[0] * (x[1] + xk[1]) / (x[0] + xk[0])
            a[2] = -globals.GM_Earth * (x[2] + xk[2]) / r ** 3 * (
                        1 + 1.5 * globals.J2 * (globals.r_E / r) ** 2 * (3 - 5 * (x[2] + xk[2]) ** 2 / r ** 2))

        if model[1] == 1:
            if dens is None:
                a = a + self.AccelDrag(x + xk)
            else:
                a = a + self.AccelDrag(x + xk, dens)

        return a

    def TwoBodyCal(self,t, RV, Thrust_f, deg):
        GM_Earth = globals.GM_Earth
        Azimuth = deg[0]
        Elevation = deg[1]

        x = RV[3]
        y = RV[4]
        z = RV[5]
        Tool = ToolFunction()
        # 脉冲轨道系定向  需要求解对应时刻惯性系的脉冲
        RotMat = Tool.Inertial2Orbit(RV).T
        vec = [np.cos(np.radians(Elevation)) * np.cos(np.radians(Azimuth)),
              np.cos(np.radians(Elevation)) * np.sin(np.radians(Azimuth)),
              np.sin(np.radians(Elevation))]

        r = np.linalg.norm(RV[:3])
        dx = -GM_Earth / r ** 3 * RV[0] + Thrust_f * np.dot(RotMat[0], vec)
        dy = -GM_Earth / r ** 3 * RV[1] + Thrust_f * np.dot(RotMat[1], vec)
        dz = -GM_Earth / r ** 3 * RV[2] + Thrust_f * np.dot(RotMat[2], vec)

        drv = np.array([x, y, z, dx, dy, dz])
        return drv
    def TwoBodyCal_rvm(self,t, RVm, Thrust_f, deg, k):
        GM_Earth = globals.GM_Earth
        Azimuth = deg[0]
        Elevation = deg[1]

        x = RVm[3]
        y = RVm[4]
        z = RVm[5]
        m = RVm[6]
        Tool = ToolFunction()
        # 脉冲轨道系定向  需要求解对应时刻惯性系的脉冲
        RotMat = Tool.Inertial2Orbit(RVm).T
        vec = [np.cos(np.radians(Elevation)) * np.cos(np.radians(Azimuth)),
              np.cos(np.radians(Elevation)) * np.sin(np.radians(Azimuth)),
              np.sin(np.radians(Elevation))]
        accel = Thrust_f / m / 1000  # 根据推力计算加速度大小
        r = np.linalg.norm(RVm[:3])
        dx = -GM_Earth / r ** 3 * RVm[0] + accel * np.dot(RotMat[0], vec)
        dy = -GM_Earth / r ** 3 * RVm[1] + accel * np.dot(RotMat[1], vec)
        dz = -GM_Earth / r ** 3 * RVm[2] + accel * np.dot(RotMat[2], vec)

        drvm = np.array([x, y, z, dx, dy, dz, k])
        return drvm
    def twoBodyOrbit(self,coe0, T):
        Core = OrbitCore()
        q = Core.true2Mean(coe0[1], coe0[5])
        Tperiod = 2 * math.pi * math.sqrt(coe0[0] ** 3 / globals.GM_Earth)
        M = 2 * math.pi / Tperiod * T * globals.rad2deg
        q = Core.amendDeg(Core.mean2True(coe0[1], Core.amendDeg(q + M, '0 - 360')), '0 - 360')
        coe = list(coe0)
        coe[5] = q
        return coe
    def twoBodyOrbitRV(self,rv0, T):
        # print("调用ing")
        Core = OrbitCore()
        coe0 = Core.State_rv_2_Orbit_Element(rv0[:3], rv0[3:6])
        coe = self.twoBodyOrbit(coe0, T)
        r, v = Core.Orbit_Element_2_State_rv(coe)
        rv = np.concatenate((r, v))
        return rv
