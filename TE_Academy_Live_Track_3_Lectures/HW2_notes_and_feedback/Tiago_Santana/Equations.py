import math

def f(xplot, INVARIANT):
    return INVARIANT / xplot

def dydx(xplot, yplot):
    return -yplot / xplot

def dxdy(xplot, yplot):
    return -xplot / yplot

def f_concentrated1(xplot, INVARIANT, ASQUARE):
    return (INVARIANT * ASQUARE) / xplot

def dyvdxv(xvplot, INVARIANT, ASQUARE):
    return - (ASQUARE * INVARIANT) / (xvplot ** 2)

def dxvdyv(yvplot, INVARIANT, ASQUARE):
    return - (ASQUARE * INVARIANT) / (yvplot ** 2)

def f_concentrated2(Xplot, INVARIANT, ASQUARE, X0, Y0):
    return (ASQUARE * INVARIANT) / (Xplot + X0 * (math.sqrt(ASQUARE) - 1)) - Y0 * (math.sqrt(ASQUARE) - 1)

def dYdX(Xplot, INVARIANT, ASQUARE, X0):
    return - (ASQUARE * INVARIANT) / (Xplot + X0 * (math.sqrt(ASQUARE) - 1)) ** 2
