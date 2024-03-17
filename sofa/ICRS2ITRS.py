# 地心惯性系(ICRS)到地心固连系(ITRS)的坐标转化矩阵
import numpy as np
import erfa

def ICRS2ITRS(Mjd):
    # 转换MJD为UTC（协调世界时）
    UTC = Mjd
    # 计算TT（地球时）
    TT = UTC + 66.184 / 86400

    # 计算NPB矩阵（PRECESSION-NUTATION-BIAS）
    NPB = erfa.pnm06a(2400000.5, TT)

    # 计算GAST（格林尼治视恒星时）
    gast = erfa.gst06(2400000.5, UTC, 2400000.5, TT,NPB)
    # 计算旋转矩阵Theta
    Theta = erfa.rz(gast, np.eye(3))

    # 计算SP矩阵（PRECESSION）
    sp = erfa.sp00(2400000.5, TT)

    # 计算Pi矩阵（光行差）
    Pi = erfa.pom00(0, 0, sp)

    # 计算最终的坐标转换矩阵E
    E = np.matmul(np.matmul(Pi, Theta), NPB)

    return E
