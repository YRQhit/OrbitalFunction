
def Mjday(year, month, day, hour=0, min=0, sec=0):
    if month <= 2:
        year -= 1
        month += 12

    if year < 0:
        c = -0.75
    else:
        c = 0

    if year < 1582:
        # null
        pass
    elif year > 1582:
        a = year // 100
        b = 2 - a + a // 4
    elif month < 10:
        # null
        pass
    elif month > 10:
        a = year // 100
        b = 2 - a + a // 4
    elif day <= 4:
        # null
        pass
    elif day > 14:
        a = year // 100
        b = 2 - a + a // 4
    else:
        print("\n\n  This is an invalid calendar date!!\n")
        return

    jd = int(365.25 * year + c) + int(30.6001 * (month + 1))
    jd = jd + day + b + 1720994.5
    jd = jd + (hour + min / 60 + sec / 3600) / 24
    Mjd = jd - 2400000.5
    return Mjd
