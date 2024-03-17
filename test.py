
'''
OrbitalTransfer函数测试
'''

# from OrbitalTransfer import *
# Trans = OrbitalTransfer()
# coe_c = np.array([6885,0.001,97.5,10,20,5])
# coe_t = np.array([6885,0.0012,97.5,10,20,5.3])
# delta = np.array([10,0,0,0,0,0])
# [deltv1, deltv2] = Trans.CW2ImpulseTransfer(coe_c, coe_t, delta, 5000)
# print(deltv1, deltv2)

# from OrbitalTransfer import *
# Trans = OrbitalTransfer()
# r_LVLH = np.array([1, 2, 3])
# v_LVLH = np.array([0.1, 0.2, 0.3])
# n = 0.1
# t = 2.0
# rt_LVLH, vt_LVLH = Trans.CW_Transfer(r_LVLH, v_LVLH, n, t)
# print("rt_LVLH:", rt_LVLH)
# print("vt_LVLH:", vt_LVLH)


# from OrbitalTransfer import *
# Trans = OrbitalTransfer()
# start_time = [2022, 9, 9, 0, 0, 0]
# r2 = [-2.719328240311940e+04, 3.257002397590526e+04, 0.568397574969637, -2.360342780120658, -1.940086718474888, -3.386452149363278e-05]
# r1 = [3.925028471108628e+04, 1.428593531913334e+04, 0.249336608071141, -1.051625445598732, 2.920064636113321, 5.096474227640248e-05]
# T = 25000
# V1Min, V2Min, E = Trans.lambertIteration(r1, r2[0:3], T, start_time)
# print(V1Min, V2Min, E)
#
# from OrbitalTransfer import *
# Trans = OrbitalTransfer()
# red_rv = [4.117995088156707e+04, 8.753079979601422e+03, -72.362023354757740]
# blue_rv = [4.124450852427926e+04, 8.766802147740464e+03, -72.475465006572800]
# tf = 100
# v0, v = Trans.lamberthigh(red_rv, blue_rv, tf)
# print(v0,v)

'''
OrbitCorer函数测试
'''

# Example usage:
# from OrbitCore import *
# JD = 2459396.5  # Julian Date
# Orbit = OrbitCore()
# SunPos = Orbit.CalAstroBodyPos(JD)
# print("Sun Position (km):", SunPos)


'''
OrbitPredict函数测试
'''

'''
Toolfuncition函数测试
'''
# from ToolFunction import *
# Tool = ToolFunction()
# JD = 2459396.5  # 日期对应的Julian Date
# sun_Vector_J2000 = Tool.getSunVector(JD)
# print("Sun Vector (J2000):")
# print(sun_Vector_J2000)


# from ToolFunction import *
# Tool = ToolFunction()
# JD_startTime = [2023, 8, 2, 12, 30, 0]  # 日期对应的年月日时分秒
# x_c = np.array([0, 0, 0, 0, 0, 0])  # 追踪星的状态向量
# x_t = np.array([1, 0, 0, 0, 1, 0])  # 目标星的状态向量
# idealDegree, actualDegree = Tool.IlluminationAngle(x_c, x_t, JD_startTime)
# print("Ideal Illumination Angle:", idealDegree)
# print("Actual Illumination Angle:", actualDegree)


# from ToolFunction import *
# Tool = ToolFunction()
# R1 = np.array([8000, 10000, 15000])  # km
# V1 = np.array([7, 7.5, 6])           # km/s
# R2 = np.array([8500, 9500, 14500])  # km
# V2 = np.array([7.2, 7.7, 6.2])       # km/s
#
# delta, QXx = Tool.Inertial2Relative(R1, V1, R2, V2)
#
# print("Relative Delta:")
# print(delta)
# print("Transformation Matrix QXx:")
# print(QXx)

# from ToolFunction import *
# Tool = ToolFunction()
# Mjd = 58650.0  # Example Modified Julian Date (MJD)
# lonlan = [120.0, 30.0, 1000.0]  # Example lon, lat, alt in degrees and meters
#
# # Call the LLA2ICRS function
# pos_icrs = Tool.LLA2ICRS(Mjd, lonlan)
#
# # Print the resulting ICRS position
# print("ICRS Position:", pos_icrs)

#
# from ToolFunction import *
# Tool = ToolFunction()
# rv = [ 6.20489388e+03 , 7.08819884e+02 , 2.88195974e+03 ,-3.01287254e+00 , -1.44618334e+00 , 6.84402314e+00]
# date = [2018,1,1,0,0,0]
#
# print(Tool.xingxiadian(rv,date))