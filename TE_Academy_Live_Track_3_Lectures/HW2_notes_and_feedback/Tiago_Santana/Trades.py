from scipy.stats import norm
from Constants import *
from Equations import *

# from Plots import *
def initializeAMM():
    global x, y, yprice_curve, xprice_curve, xtrade, ytrade, xv, yv, yvprice_curve, xvprice_curve, xvtrade, yvtrade, X, Y, Yprice_curve, Xprice_curve, Xtrade, Ytrade
    
    # First function
    x = X0
    y = Y0
    
    xtrade = []
    ytrade = []

    xprice_curve = []
    yprice_curve = []
    
    # Second function
    xv = x * math.sqrt(ASQUARE)
    yv = (INVARIANT * ASQUARE) / xv
    
    xvtrade = []
    yvtrade = []
    
    yvprice_curve = []
    xvprice_curve = []
    
    # Third function
    X = X0
    Y = Y0
    
    Xtrade = []
    Ytrade = []

    Xprice_curve = []
    Yprice_curve = []


def trade():
    global x, y, yprice_curve, xprice_curve, xtrade, ytrade, xv, yv, yvprice_curve, xvprice_curve, xvtrade, yvtrade, X, Y, Yprice_curve, Xprice_curve, Xtrade, Ytrade
    nextx = nexty = 0
    nextxv = nextyv = 1 # JUST A RANDOM NUMBER ABOVE 0
    nextX = nextY = 0
    deltaxxv = norm.rvs(loc=0, scale=80, size=1)
    
    while nextx <= 0 or nexty <= 0 or nextxv <= 0 or nextyv <= 0 or XV0-nextxv>=X0 or YV0-nextyv>=Y0 or nextX <= 0 or nextY <= 0 : # MAKING SURE THE TRADES TAKE PLACE WITHIN THE PRICE RANGES
        nextx = x + deltaxxv[0]
        nexty = INVARIANT / nextx if nextx != 0 else 0
        
        nextxv = xv + deltaxxv[0]
        nextyv = (ASQUARE * INVARIANT) / nextxv if nextxv != 0 else 0
        
        nextX = X + deltaxxv[0]
        nextY = (ASQUARE*INVARIANT)/(nextX+X0*(math.sqrt(ASQUARE)-1))-Y0*(math.sqrt(ASQUARE)-1)
        
    # FUNCTION 1    
    xtrade.append([x, nextx])
    ytrade.append([y, nexty])
    xprice_curve.append([-x/y, -nextx/nexty])
    yprice_curve.append([-y/x, -nexty/nextx])
    x, y = nextx, nexty
        
    # FUNCTION 2  
    xvtrade.append([xv, nextxv])
    yvtrade.append([yv, nextyv])
    xvprice_curve.append([-ASQUARE*INVARIANT/yv**2, -ASQUARE*INVARIANT/nextyv**2]) # xvprice_curve.append([-xv/yv, -nextxv/nextyv])
    yvprice_curve.append([-ASQUARE*INVARIANT/xv**2, -ASQUARE*INVARIANT/nextxv**2]) # yvprice_curve.append([-yv/xv, -nextyv/nextxv])
    
    xv, yv = nextxv, nextyv
    
    # FUNCTION 3  
    Xtrade.append([X, nextX])
    Ytrade.append([Y, nextY])
    Xprice_curve.append([1/dYdX(X, INVARIANT, ASQUARE, X0) , 1/dYdX(nextX, INVARIANT, ASQUARE, X0)])
    Yprice_curve.append([dYdX(X, INVARIANT, ASQUARE, X0) , dYdX(nextX, INVARIANT, ASQUARE, X0)])
    
    X, Y = nextX, nextY
    
    return xtrade,ytrade, xprice_curve, yprice_curve, xvtrade, yvtrade, xvprice_curve, yvprice_curve, Xtrade, Ytrade, Xprice_curve, Yprice_curve
