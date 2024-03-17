
import numpy as np
from scipy.linalg import expm
from OrbitCore import *
from ToolFunction import *
from OrbitPredict import *
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from OrbitalTransfer import *
import globals
from poliastro.iod import izzo
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np
class OrbitalTransfer():
    def CW2ImpulseTransfer(self,coe_c, coe_t, delta, t=None):
        GM_Earth = 398600.4418

        n = np.sqrt(GM_Earth / coe_t[0] ** 3)

        if t is None:
            t = np.pi / n

        # Replace Orbit_Element_2_State_rv, Inertial2Relative, and OrbitPrediction with your implementations
        Orbit = OrbitCore()
        r_c, v_c = Orbit.Orbit_Element_2_State_rv(coe_c, GM_Earth)
        r_t, v_t = Orbit.Orbit_Element_2_State_rv(coe_t, GM_Earth)
        Tool = ToolFunction()
        p, QXx = Tool.Inertial2Relative(r_t, v_t, r_c, v_c)

        [Qrr, Qrv, _, _] = self.CWStateTransitionMatrix(n, t)


        if np.abs(np.linalg.det(Qrv)) > 1e-3:
            v1 = np.linalg.inv(Qrv).dot(delta[:3] - Qrr.dot(p[:3]))
        else:
            v1 = np.linalg.pinv(Qrv).dot(delta[:3] - Qrr.dot(p[:3]))

        deltv1 = np.dot(QXx.T, v1 - p[3:])
        Predict = OrbitPredict()
        targetPosVel, _ = Predict.OrbitPrediction(np.concatenate((r_t, v_t)), t, 60, [0, 0], 'RK7')
        E = []

        for i in range(10):
            chasePosVel, _ = Predict.OrbitPrediction(np.concatenate((r_c, v_c + deltv1)), t, 60, [0, 0], 'RK7')
            relState, QXx2 = Tool.Inertial2Relative(targetPosVel[:3], targetPosVel[3:], chasePosVel[:3], chasePosVel[3:])
            err = relState - delta
            E.append(np.linalg.norm(err[:3]))

            if np.linalg.norm(err[:3]) < 0.0001:
                break

            if np.abs(np.linalg.det(Qrv)) > 1e-3:
                v1 = v1 - np.linalg.inv(Qrv).dot(err[:3])
            else:
                v1 = v1 - np.linalg.pinv(Qrv).dot(err[:3])

            deltv1 = np.dot(QXx.T, v1 - p[3:])

        deltv2 = np.dot(QXx2.T, delta[3:] - relState[3:])

        return deltv1, deltv2


    def CW_Transfer(self,r_LVLH, v_LVLH, n, t):
        Qrr, Qrv, Qvr, Qvv = self.CWStateTransitionMatrix(n, t)
        rt_LVLH = np.dot(Qrr, r_LVLH) + np.dot(Qrv, v_LVLH)
        vt_LVLH = np.dot(Qvr, r_LVLH) + np.dot(Qvv, v_LVLH)
        return rt_LVLH, vt_LVLH


    def lambertIteration(self,r1, r2, T, startTime=None):
        GM_Earth  = globals.GM_Earth
        origin = r1[0:3]
        target = r2
        E = []
        errPre = np.linalg.norm(np.array(origin) - np.array(target))

        if len(r1) == 6:
            v_ref = r1[3:6]
        else:
            v_ref = np.array([0, 0, 0])
        r1_v1 = np.cross(origin, v_ref) / np.linalg.norm(np.cross(origin, v_ref))
        Predict = OrbitPredict()
        for i in range(10):
            v1, v2 = self.lamberthigh(origin, target, T)

            norm_diff_1 = np.linalg.norm(np.array(v1) - np.array(v_ref))
            norm_diff_2 = np.linalg.norm(np.array(v2) - np.array(v_ref))

            if norm_diff_1 <= norm_diff_2:
                V1 = v1
            else:
                V1 = v2

            initState = np.hstack((origin, V1))
            if startTime is None:
                finalState, _ = Predict.OrbitPrediction(initState, T, 60, [1, 1], 'RK7')
            else:
                finalState, _ = Predict.OrbitPrediction(initState, T, 60, [1, 1, 1], 'RK7', startTime)

            err = finalState[0:3] - r2
            e_n = np.linalg.norm(err)

            if errPre > e_n:
                V1Min = V1
                V2Min = finalState[3:6]
                errPre = e_n

            target = target - err
            E.append(e_n)
            if e_n < 0.01:
                break

        return V1Min, V2Min, E


    def DeadZone(self,r1_v1, r2):
        angle = np.dot(r2, r1_v1)
        r = r2 - angle * r1_v1
        return r
    #输入：n - 轨道角速度             t - 时间
    def CWStateTransitionMatrix(self,n, t):
        Qrr = np.array([[4 - 3 * np.cos(n * t), 0, 0],
                        [6 * (np.sin(n * t) - n * t), 1, 0],
                        [0, 0, np.cos(n * t)]])

        Qrv = np.array([[np.sin(n * t) / n, 2 * (1 - np.cos(n * t)) / n, 0],
                        [2 * (np.cos(n * t) - 1) / n, (4 * np.sin(n * t) - 3 * n * t) / n, 0],
                        [0, 0, np.sin(n * t) / n]])

        Qvr = np.array([[3 * n * np.sin(n * t), 0, 0],
                        [6 * n * (np.cos(n * t) - 1), 0, 0],
                        [0, 0, -n * np.sin(n * t)]])

        Qvv = np.array([[np.cos(n * t), 2 * np.sin(n * t), 0],
                        [-2 * np.sin(n * t), 4 * np.cos(n * t) - 3, 0],
                        [0, 0, np.cos(n * t)]])

        return Qrr, Qrv, Qvr, Qvv



    def lamberthigh(self,r0, r, tf):
        # 计算 tof
        k = Earth.k
        tof = tf / 60 * u.min

        # 使用 poliastro 的 izzo.lambert 计算 v0
        r0 = r0 * u.km
        r = r * u.km
        v0, v = izzo.lambert(k, r0, r, tof)

        return v0.value, v.value