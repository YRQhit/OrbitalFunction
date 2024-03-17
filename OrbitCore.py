import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from OrbitPredict import *
from OrbitalTransfer import *
import globals
from scipy.optimize import newton
import numpy as np
import math
from astropy.coordinates import get_body_barycentric, GCRS
from astropy.time import Time
from astropy.coordinates.matrix_utilities import rotation_matrix
from scipy.optimize import newton
import numpy as np
import math
class OrbitCore:
    def amendDeg(self,deg0, type):
        if type == '':
            type = '-180 - 360'

        if type == '0 - 180':
            range_min = 0
            range_max = 180
        elif type == '0 - 360':
            range_min = 0
            range_max = 360
        elif type == '-180 - 360':
            range_min = -180
            range_max = 360
        else:
            print('该范围有误')
            return

        if deg0 > range_max:
            deg = deg0 % range_max
        elif deg0 > range_min:
            deg = deg0
        else:
            deg = range_max - abs(deg0) % range_max

        return deg
    
    def CalJDTDB(self,JD):
        DAT = 36  # TAI-UTC, in seconds
        JDTT = JD + (DAT + 32.184) / 24 / 3600
        JDTDB = JDTT + (0.001657 * np.sin(6.24 + 0.017202 * (JDTT - 2451545.0))) / 24 / 3600
        return JDTDB

    def kepler_E(self,e, M):
        def kepler_eq(E):
            return E - e * np.sin(E) - M

        E0 = M + e / 2 if M < np.pi else M - e / 2
        E = newton(kepler_eq, E0)
        return E

    def CalOrbEle(self,JCTDB, OrbEle_0, dOrbEle_0):
        OrbEle = OrbEle_0 + dOrbEle_0 * JCTDB
        return OrbEle



    def CalPos(self,h, e, Omegaa, i, omegaa, thetaa, GM):
        rp = (h ** 2 / GM) * (1 / (1 + e * np.cos(thetaa))) * np.array([np.cos(thetaa), np.sin(thetaa), 0])

        R3_Omegaa = np.array([[np.cos(Omegaa), np.sin(Omegaa), 0],
                              [-np.sin(Omegaa), np.cos(Omegaa), 0],
                              [0, 0, 1]])

        R1_i = np.array([[1, 0, 0],
                        [0, np.cos(i), np.sin(i)],
                        [0, -np.sin(i), np.cos(i)]])

        R3_omegaa = np.array([[np.cos(omegaa), np.sin(omegaa), 0],
                              [-np.sin(omegaa), np.cos(omegaa), 0],
                              [0, 0, 1]])

        Q_pX = np.dot(R3_Omegaa.T, np.dot(R1_i.T, R3_omegaa.T))
        r = np.dot(Q_pX, rp)
        r = np.reshape(r, (3, 1))
        Pos = r

        return Pos

    def CalAstroBodyPosSCI(self,JCTDB, GM, a_0, da_0, e_0, de_0, i_0, di_0, Omegaa_0, dOmegaa_0, hat_omegaa_0, dhat_omegaa_0, L_0, dL_0):
        a = self.CalOrbEle(JCTDB, a_0, da_0)
        e = self.CalOrbEle(JCTDB, e_0, de_0)
        i = self.CalOrbEle(JCTDB, i_0, di_0)
        Omegaa = self.CalOrbEle(JCTDB, Omegaa_0, dOmegaa_0)
        hat_omegaa = self.CalOrbEle(JCTDB, hat_omegaa_0, dhat_omegaa_0)
        L = self.CalOrbEle(JCTDB, L_0, dL_0)
        h = np.sqrt(GM * a * (1 - e**2))
        omegaa = hat_omegaa_0 - Omegaa
        M = L - hat_omegaa
        E = self.kepler_E(e, M)
        thetaa = math.atan(math.tan(E / 2) * math.sqrt((1 + e) / (1 - e))) * 2
        # thetaa = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

        AstroBodyPosSCI = self.CalPos(h, e, Omegaa, i, omegaa, thetaa, GM)
        return AstroBodyPosSCI

    def get_ECI_to_SCI_transform_matrix(self,obliquity):
        cos_ecl = np.cos(obliquity)
        sin_ecl = np.sin(obliquity)
        ECI_to_SCI_transform_matrix = np.array([[1, 0, 0],
                                                [0, cos_ecl, sin_ecl],
                                                [0, -sin_ecl, cos_ecl]])

        return ECI_to_SCI_transform_matrix

    def CalAstroBodyPos(self,JD):
        d2r = np.pi / 180
        JDTDB = self.CalJDTDB(JD)
        JCTDB = (JDTDB - 2451545) / 36525

        GM_s = 1.327122000000e+11  # Sun's gravitational constant, unit: km^3/s^2
        EcliOblJ2000 = 23.43929 * d2r  # J2000.0 obliquity of the ecliptic, unit: rad

        # Calculate the transformation matrix from Earth-centered inertial (ECI) to Solar-centered inertial (SCI) coordinates
        ECI_to_SCI_transform_matrix = self.get_ECI_to_SCI_transform_matrix(EcliOblJ2000)

        AU = 1.49597871e8
        EarthPosSCI = self.CalAstroBodyPosSCI(JCTDB, GM_s, 1.00000011 * AU, -0.00000005 * AU, 0.01671022, -0.00003804, 0.00005 * d2r, -46.94 / 3600 * d2r, -11.26064 * d2r, -18228.25 / 3600 * d2r, 102.94719 * d2r, 1198.28 / 3600 * d2r, 100.46435 * d2r, 129597740.63 / 3600 * d2r)

        # Apply the transformation matrix to get the Sun's position in SCI coordinates
        SunPos = np.dot(ECI_to_SCI_transform_matrix.T, EarthPosSCI)

        return SunPos
    
    def mean2True(self,e, Me):
        Me_r = self.amendDeg(Me, '0 - 360') * globals.deg2rad
        E_r = Me_r
        if Me_r < math.pi:
            E_r += e / 2
        else:
            E_r -= e / 2

        ratio = 1
        j = 0
        eps = 1e-10
        while abs(ratio) > eps:
            ratio = (E_r - e * math.sin(E_r) - Me_r) / (1 - e * math.cos(E_r))
            E_r -= ratio
            j += 1
            if j >= 50:
                break

        trueAng = 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E_r / 2)) * globals.rad2deg
        trueAng = self.amendDeg(trueAng, '0 - 360')

        return trueAng
    
    # a：轨道半长轴
    # e：轨道偏心率
    # incl：轨道倾角(单位/deg)
    # RAAN：升交点赤经(单位/deg)
    # omegaa：近地点幅角(单位/deg)
    # TA：真近点角(单位/deg)
    def Orbit_Element_2_State_rv(self,coe, muu=3.986004415e+05):
        d_2_r = np.pi / 180
        a = coe[0]
        e = coe[1]
        incl = coe[2] * d_2_r
        RAAN = coe[3] * d_2_r
        omegaa = coe[4] * d_2_r
        TA = coe[5] * d_2_r

        h = np.sqrt(a * muu * (1 - e ** 2))

        rp = (h ** 2 / muu) * (1 / (1 + e * np.cos(TA))) * (np.cos(TA) * np.array([1, 0, 0]) + np.sin(TA) * np.array([0, 1, 0]))
        vp = (muu / h) * (-np.sin(TA) * np.array([1, 0, 0]) + (e + np.cos(TA)) * np.array([0, 1, 0]))

        R3_RAAN = np.array([[np.cos(RAAN), np.sin(RAAN), 0],
                            [-np.sin(RAAN), np.cos(RAAN), 0],
                            [0, 0, 1]])

        R1_incl = np.array([[1, 0, 0],
                            [0, np.cos(incl), np.sin(incl)],
                            [0, -np.sin(incl), np.cos(incl)]])

        R3_omegaa = np.array([[np.cos(omegaa), np.sin(omegaa), 0],
                              [-np.sin(omegaa), np.cos(omegaa), 0],
                              [0, 0, 1]])

        Q_px = np.dot(np.dot(R3_RAAN.T, R1_incl.T), R3_omegaa.T)

        r = np.dot(Q_px, rp)
        v = np.dot(Q_px, vp)

        return r, v
    #  函数功能：由航天器位置速度矢量求航天器轨道六要素
    #  输入：
    #        R：航天器位置矢量（行向量,列向量均可，单位，km）
    #        V：航天器速度矢量（行向量,列向量均可，单位km/s）。
    #        muu：地球引力常量，缺省输入为398600.4415kg^3/s^2。
    #  输出：
    #        coe：航天器轨道六要素，具体见下（列向量）。
    #  ---------coe（classical orbit elements）------------- %
    #  a：轨道半长轴（单位，km）
    #  e：轨道偏心率（无量纲）
    #  incl：轨道倾角(单位，°)
    #  RAAN：升交点赤经(单位，°)
    #  omegap：近地点幅角(单位，°)
    #  TA：真近点角(单位，°)
    def State_rv_2_Orbit_Element(self,R, V, mu=398600.4415):
        eps = 1e-6
        r2d = 180 / np.pi

        R = np.reshape(R, (3, 1))
        V = np.reshape(V, (3, 1))
        r = np.linalg.norm(R)
        v = np.linalg.norm(V)

        vr = np.dot(np.transpose(R), V) / r  # 径向速度
        H = np.cross(np.transpose(R), np.transpose(V))  # 比角动量
        H = H[0]
        h = np.linalg.norm(H)
        incl = np.arccos(H[2] / h)  # 轨道倾角

        N = np.cross(np.array([0, 0, 1]), H)  # 真近点角
        n = np.linalg.norm(N)

        if np.abs(incl) <= 1e-6:
            RA = 0
        elif n != 0:
            RA = np.arccos(N[0] / n)
            if N[1] < 0:
                RA = 2 * np.pi - RA
        else:
            RA = 0

        E = 1 / mu * ((v ** 2 - mu / r) * R - r * vr * V)
        e = np.linalg.norm(E)

        if np.abs(e) <= 1e-10:
            omegap = 0
        elif n != 0:
            if e > eps:
                omegap = np.real(np.arccos(np.dot(N, E) / (n * e))).item()
                # omegap = np.real(np.arccos(np.dot(N, E) / n / e))
                if E[2] < 0:
                    omegap = 2 * np.pi - omegap
            else:
                omegap = 0
        else:
            omegap = 0

        if e > eps:
            TA = np.real(np.arccos(np.dot(E.transpose(), R) / e / r))
            TA = float(TA)
            if vr < 0:
                TA = 2 * np.pi - TA
        else:
            TA = np.real(np.arccos(np.dot(N.transpose(), R) / n / r))
            # TA = np.real(np.arccos(np.dot(N.transpose(), R) / (n + 0.0000000001) / r))
        a = h ** 2 / mu / (1 - e ** 2)
        coe = np.array([a, e, incl * r2d, RA * r2d, omegap * r2d, TA * r2d], dtype=object)
        r, _ = self.Orbit_Element_2_State_rv(coe)
        r = np.reshape(r,(3,1))
        if np.linalg.norm(r - R) > 1:
            coe[5] = 360 - coe[5]

        if e < eps and incl <= 1e-6:
            omegap = omegap + RA
            RA = 0
            TA = TA + omegap
            omegap = 0
            v_dir = np.cross(np.array([0, 0, 1]), np.array([1, 0, 0]))
            dir_proj = np.sign(np.dot(R.transpose(), v_dir) / np.linalg.norm(v_dir))
            TA = dir_proj * np.arccos(np.dot([1, 0, 0], R) / np.linalg.norm(R))
        # coe[5] = (coe[5])[0]
        return coe
    #  真近点角转平近点角
    #  输入：e - 偏心率  trueAno - 真近点角（deg)
    #  输出：Me - 平近点角（deg)
    def true2Mean(self,e, trueAno):
        trueAno = trueAno * globals.deg2rad
        E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(trueAno / 2))
        Me = (E - e * math.sin(E)) * globals.rad2deg
        return Me