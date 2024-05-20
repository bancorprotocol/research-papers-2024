from Constants import *
from Equations import *
import numpy as np
from Trades import *


class Plots:
    def __init__(self, text1Text=None,text1XCoords = 0.5, text1YCoords = 0.5, text2XCoords=0.5, text2YCoords=0.4, text2Text=None, xplot=None, yplot=None, color = None, scatterPointsX=None, scatterPointsY=None,scatterColors=None, scatterEdgeColors=None, xLabel=None, yLabel=None, arrow1=None, arrow2=None, arrow3=None, integralX=0, integralY=0,legendXtraLine1=None,legendXtraLine2=None,legendXtraLine3=None,legendXtraLine4=None,legendXtraLine5=None):
        self.text1Text = text1Text
        self.text1XCoords = text1XCoords
        self.text1YCoords = text1YCoords
        self.text2XCoords = text2XCoords
        self.text2YCoords = text2YCoords
        self.text2Text = text2Text
        self.xplot = xplot
        self.yplot = yplot
        self.color = color
        self.scatterPointsX = scatterPointsX
        self.scatterPointsY = scatterPointsY
        self.scatterColors = scatterColors
        self.scatterEdgeColors = scatterEdgeColors
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.arrow1 = arrow1
        self.arrow2 = arrow2
        self.arrow3 = arrow3
        self.integralX = integralX
        self.integralY = integralY
        self.legendXtraLine1 = legendXtraLine1
        self.legendXtraLine2 = legendXtraLine2
        self.legendXtraLine3 = legendXtraLine3
        self.legendXtraLine4 = legendXtraLine4
        self.legendXtraLine5 = legendXtraLine5
        
    def get_scatter_data(self, n,i,j, xtrade,ytrade, xprice_curve, yprice_curve, xvtrade, yvtrade, xvprice_curve, yvprice_curve, Xtrade, Ytrade, Xprice_curve, Yprice_curve):
        if i == 1:
            self.scatterColors = ['blue', 'red']
            self.scatterEdgeColors = ['blue', 'red']
            if  j == 1:
                self.scatterPointsX = [xtrade[n][0], xtrade[n][1]]
                self.scatterPointsY = [ytrade[n][0], ytrade[n][1]]
                self.arrow1 = ((xtrade[n][0], ytrade[n][0]), (xtrade[n][0], ytrade[n][1]))
                self.arrow2 = ((xtrade[n][0], ytrade[n][1]), (xtrade[n][1], ytrade[n][1]))
                self.arrow3 = ((xtrade[n][0], ytrade[n][0]), (xtrade[n][1], ytrade[n][1]))
                self.legendXtraLine1 = {'label':f'ΔX = {xtrade[n][1] - xtrade[n][0]:.2f}'}
                self.legendXtraLine2 = {'label':f'ΔY = {ytrade[n][1] - ytrade[n][0]:.2f}'}
                self.legendXtraLine3 = {'label':f'DY/DX = {- abs((ytrade[n][1] - ytrade[n][0]) / (xtrade[n][1] - xtrade[n][0])):.2f}'}
            elif j == 2:
                self.scatterPointsX = [xtrade[n][0], xtrade[n][1]] 
                self.scatterPointsY = [yprice_curve[n][0], yprice_curve[n][1]]
                self.arrow1 = ((xtrade[n][0], yprice_curve[n][0]), (xtrade[n][0], yprice_curve[n][1]))
                self.arrow2 = ((xtrade[n][0], yprice_curve[n][1]), (xtrade[n][1], yprice_curve[n][1]))
                self.arrow3 = ((xtrade[n][0], yprice_curve[n][0]), (xtrade[n][1], yprice_curve[n][1]))
                self.integralX = np.linspace(xtrade[n][0], xtrade[n][1], 10000)
                self.integralY = -f(np.linspace(xtrade[n][0], xtrade[n][1], 10000), INVARIANT)/np.linspace(xtrade[n][0], xtrade[n][1], 10000)
                self.legendXtraLine1 = {'label':f'|ΔY| = {abs(ytrade[n][1] - ytrade[n][0]):.2f}'}
            elif j == 3:
                self.scatterPointsX = [ytrade[n][0], ytrade[n][1]] 
                self.scatterPointsY = [xprice_curve[n][0], xprice_curve[n][1]]
                self.arrow1 = ((ytrade[n][0], xprice_curve[n][0]), (ytrade[n][0], xprice_curve[n][1]))
                self.arrow2 = ((ytrade[n][0], xprice_curve[n][1]), (ytrade[n][1], xprice_curve[n][1]))
                self.arrow3 = ((ytrade[n][0], xprice_curve[n][0]), (ytrade[n][1], xprice_curve[n][1]))
                self.integralX = np.linspace(ytrade[n][0], ytrade[n][1], 10000)
                self.integralY = -f(np.linspace(ytrade[n][0], ytrade[n][1], 10000), INVARIANT)/np.linspace(ytrade[n][0], ytrade[n][1], 10000)
                self.legendXtraLine1 = {'label':f'|ΔX| = {abs(xtrade[n][1] - xtrade[n][0]):.2f}'}
        elif i == 2:
            if  j == 1:
                self.scatterPointsX = [PHIGH_POINT[0], PLOW_POINT[0],xvtrade[n][0], xvtrade[n][1]] 
                self.scatterPointsY = [PHIGH_POINT[1], PLOW_POINT[1],yvtrade[n][0], yvtrade[n][1]]  
                self.scatterColors=['white',  'white', 'blue','red'] 
                self.scatterEdgeColors=['blue', 'red', 'blue', 'red']
                self.arrow1 = ((xvtrade[n][0], yvtrade[n][0]), (xvtrade[n][0], yvtrade[n][1]))
                self.arrow2 = ((xvtrade[n][0], yvtrade[n][1]), (xvtrade[n][1], yvtrade[n][1]))
                self.arrow3 = ((xvtrade[n][0], yvtrade[n][0]), (xvtrade[n][1], yvtrade[n][1]))
                self.legendXtraLine1 = {'label':'vP_Bound1', 'markerfacecolor':'white', 'markeredgecolor':'red'}
                self.legendXtraLine2 = {'label':'vP_Bound2', 'markerfacecolor':'white', 'markeredgecolor':'blue'}
                self.legendXtraLine3 = {'label':f'ΔX={xvtrade[n][1] - xvtrade[n][0]:.2f}'}
                self.legendXtraLine4 = {'label':f'ΔY={yvtrade[n][1] - yvtrade[n][0]:.2f}'}
                self.legendXtraLine5 = {'label':f'DY/DX={(yvtrade[n][1]-yvtrade[n][0]) / (xvtrade[n][1] - xvtrade[n][0]):.2f}'}  
            elif j == 2:
                self.scatterPointsX = [xvtrade[n][0], xvtrade[n][1], np.linspace(PLOW_POINT[0], PHIGH_POINT[0], 500)[0], np.linspace(PLOW_POINT[0], PHIGH_POINT[0])[-1]] 
                self.scatterPointsY = [yvprice_curve[n][0], yvprice_curve[n][1], dyvdxv(np.linspace(PLOW_POINT[0], PHIGH_POINT[0], 500),INVARIANT, ASQUARE)[0], dyvdxv(np.linspace(PLOW_POINT[0], PHIGH_POINT[0], 500),INVARIANT, ASQUARE)[-1]]
                self.scatterColors=['blue', 'red', 'orange', 'orange']
                self.scatterEdgeColors=['blue', 'red', 'red', 'blue']
                self.arrow1 = ((xvtrade[n][0], yvprice_curve[n][0]), (xvtrade[n][0], yvprice_curve[n][1]))
                self.arrow2 = ((xvtrade[n][0], yvprice_curve[n][1]), (xvtrade[n][1], yvprice_curve[n][1]))
                self.arrow3 = ((xvtrade[n][0], yvprice_curve[n][0]), (xvtrade[n][1], yvprice_curve[n][1])) 
                self.integralX = np.linspace(xvtrade[n][0], xvtrade[n][1], 10000)
                self.integralY = dyvdxv(np.linspace(xvtrade[n][0], xvtrade[n][1], 10000), INVARIANT, ASQUARE)
                self.legendXtraLine1 = {'label':f'vP_Bound1 ({round(PLOW_POINT[0],2)} , -{round(PHIGH,2)})', 'markerfacecolor':'orange', 'markeredgecolor':'red'}
                self.legendXtraLine2 = {'label':f'vP_Bound2 ({round(PLOW_POINT[1],2)} , -{round(PLOW,2)})', 'markerfacecolor':'orange', 'markeredgecolor':'blue'}
                self.legendXtraLine3 = {'label':f'|ΔY|={abs(yvtrade[n][1] - yvtrade[n][0]):.2f}'} 
            elif  j == 3:
                self.scatterPointsX=[yvtrade[n][0], yvtrade[n][1], np.linspace(PHIGH_POINT[0], PLOW_POINT[0], 500)[0], np.linspace(PHIGH_POINT[0], PLOW_POINT[0], 500)[-1]] 
                self.scatterPointsY=[xvprice_curve[n][0], xvprice_curve[n][1], dxvdyv(np.linspace(PHIGH_POINT[0], PLOW_POINT[0], 500), INVARIANT, ASQUARE)[0], dxvdyv(np.linspace(PHIGH_POINT[0], PLOW_POINT[0], 500), INVARIANT, ASQUARE)[-1]]  
                self.scatterColors=['blue', 'red', 'orange', 'orange'] 
                self.scatterEdgeColors=['blue', 'red', 'blue', 'red']
                self.arrow1 = ((yvtrade[n][0], xvprice_curve[n][0]), (yvtrade[n][0], xvprice_curve[n][1]))
                self.arrow2 = ((yvtrade[n][0], xvprice_curve[n][1]), (yvtrade[n][1], xvprice_curve[n][1]))
                self.arrow3 = ((yvtrade[n][0], xvprice_curve[n][0]), (yvtrade[n][1], xvprice_curve[n][1]))
                self.integralX = np.linspace(yvtrade[n][0], yvtrade[n][1], 10000)
                self.integralY = dxvdyv(np.linspace(yvtrade[n][0], yvtrade[n][1], 10000), INVARIANT, ASQUARE)
                self.legendXtraLine1 = {'label':f'vP_Bound1 ({round(PLOW_POINT[0],2)} , -{round(PHIGH,2)})', 'markerfacecolor':'orange', 'markeredgecolor':'red'}
                self.legendXtraLine2 = {'label':f'vP_Bound2 ({round(PLOW_POINT[1],2)} , -{round(PLOW,2)})', 'markerfacecolor':'orange', 'markeredgecolor':'blue'}
                self.legendXtraLine3 = {'label':f'|ΔX|={abs(xvtrade[n][1] - xvtrade[n][0]):.2f}'} 
        elif i == 3:
            if  j == 1:
                self.scatterPointsX = [(X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1), 0, Xtrade[n][0], Xtrade[n][1]]
                self.scatterPointsY = [0, (Y0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1), Ytrade[n][0], Ytrade[n][1]]  
                self.scatterColors=['white',  'white', 'blue','red']
                self.scatterEdgeColors=['blue', 'red', 'blue', 'red']
                self.arrow1 = ((Xtrade[n][0], Ytrade[n][0]), (Xtrade[n][0], Ytrade[n][1]))
                self.arrow2 = ((Xtrade[n][0], Ytrade[n][1]), (Xtrade[n][1], Ytrade[n][1]))
                self.arrow3 = ((Xtrade[n][0], Ytrade[n][0]), (Xtrade[n][1], Ytrade[n][1]))
                self.legendXtraLine1 = {'label':f'vP_Bound1 (0 , {round((Y0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1),2)})', 'markerfacecolor':'white', 'markeredgecolor':'red'}
                self.legendXtraLine2 = {'label':f'vP_Bound2 ({round((X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1),2)} , 0)', 'markerfacecolor':'white', 'markeredgecolor':'blue'}
                self.legendXtraLine3 = {'label':f'ΔX={Xtrade[n][1] - Xtrade[n][0]:.2f}'} 
                self.legendXtraLine4 = {'label':f'ΔY={Ytrade[n][1] - Ytrade[n][0]:.2f}'} 
                self.legendXtraLine5 = {'label':f'DY/DX={- abs((Ytrade[n][1] - Ytrade[n][0]) / (Xtrade[n][1] - Xtrade[n][0])):.2f}'} 
            elif  j == 2:
                self.scatterPointsX = [Xtrade[n][0], Xtrade[n][1], 0, (X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1)] 
                self.scatterPointsY = [Yprice_curve[n][0], Yprice_curve[n][1], -PHIGH, -PLOW]  
                self.scatterColors=['blue', 'red', 'orange', 'orange']
                self.scatterEdgeColors=['blue', 'red', 'red', 'blue']
                self.arrow1 = ((Xtrade[n][0], Yprice_curve[n][0]), (Xtrade[n][0], Yprice_curve[n][1]))
                self.arrow2 = ((Xtrade[n][0], Yprice_curve[n][1]), (Xtrade[n][1], Yprice_curve[n][1]))
                self.arrow3 = ((Xtrade[n][0], Yprice_curve[n][0]), (Xtrade[n][1], Yprice_curve[n][1]))
                self.integralX = np.linspace(Xtrade[n][0], Xtrade[n][1], 10000)
                self.integralY = dYdX(np.linspace(Xtrade[n][0], Xtrade[n][1], 10000), INVARIANT, ASQUARE, X0)
                self.legendXtraLine1 = {'label':f'vP_Bound1 (0 , {-round(PHIGH,2)})', 'markerfacecolor':'orange', 'markeredgecolor':'red'}
                self.legendXtraLine2 = {'label':f'vP_Bound2 ({round((Y0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1),2)} , {-round(PLOW,2)})', 'markerfacecolor':'orange', 'markeredgecolor':'blue'}
                self.legendXtraLine3 = {'label':f'|ΔY|={abs(Ytrade[n][1] - Ytrade[n][0]):.2f}'} 
            elif  j == 3:
                self.scatterPointsX=[Ytrade[n][0], Ytrade[n][1], 0, round((Y0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1),2)]
                self.scatterPointsY=[Xprice_curve[n][0], Xprice_curve[n][1], -PHIGH, -PLOW] 
                self.scatterColors=['blue', 'red', 'green', 'green']
                self.scatterEdgeColors=['blue', 'red', 'red', 'blue']
                self.arrow1=((Ytrade[n][0], Xprice_curve[n][0]), (Ytrade[n][0], Xprice_curve[n][1]))
                self.arrow2=((Ytrade[n][0], Xprice_curve[n][1]), (Ytrade[n][1], Xprice_curve[n][1]))
                self.arrow3=((Ytrade[n][0], Xprice_curve[n][0]), (Ytrade[n][1], Xprice_curve[n][1]))
                self.integralX = np.linspace(Ytrade[n][0], Ytrade[n][1], 10000)
                self.integralY = dYdX(np.linspace(Ytrade[n][0], Ytrade[n][1], 10000), INVARIANT, ASQUARE, X0)
                self.legendXtraLine1 = {'label':f'vP_Bound1 (0 , {-round(PHIGH,2)})', 'markerfacecolor':'green', 'markeredgecolor':'red'}
                self.legendXtraLine2 = {'label':f'vP_Bound2 ({round((Y0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1),2)} , {-round(PLOW,2)})', 'markerfacecolor':'green', 'markeredgecolor':'blue'}
                self.legendXtraLine3 = {'label':f'|ΔX|={abs(Xtrade[n][1]-Xtrade[n][0]):.2f}'} 
        return [self.scatterPointsX, self.scatterPointsY, self.scatterColors, self.scatterEdgeColors]      
       
plots = [
    [ # First Line
        Plots(# 0 0
            text1XCoords = 0.5,
            text1YCoords = -1,
            text1Text = f"\nX0={X0} , Y0={Y0}\nX0*Y0=Invariant={INVARIANT}\nA^2={ASQUARE}\nP_Low={round(PLOW,4)}\nP_High={round(PHIGH,4)}\nvPrice Bound 1=({round(PLOW_POINT[0],2)} , {round(PLOW_POINT[1],2)})\nvPrice Bound 2=({round(PHIGH_POINT[0],2)} , {round(PHIGH_POINT[1],2)})", 
            # ha='left', va='center', fontsize=8,fontweight='light',fontstyle='italic"
        ) ,
        Plots( # 0 1
            text1Text = 'Bonding Curve\n\n', 
            #ha='left', va='center', fontsize=8,fontweight='light',fontstyle='italic"
            text2Text = f"\n\nY={INVARIANT}/X\nYv={INVARIANT*ASQUARE}/Xv\nY=[{ASQUARE*INVARIANT}/(X+{round(X0*(math.sqrt(ASQUARE)-1)**2,2)})]-{round(Y0*(math.sqrt(ASQUARE)-1),2)}", 
            #ha='center', va='center', fontsize=9, fontstyle='italic'
        ) ,
        Plots( # 0 2
            text1Text = 'Price Curve dY/dX\n\n',
            #ha='left', va='center', fontsize=8,fontweight='light',fontstyle='italic"
            text2Text = f'\n\ndY/dX=-Y/X\ndYv/dXv=-{ASQUARE*INVARIANT}/(Xv^2)\ndY/dX=-{ASQUARE*INVARIANT}/[(X+{round(X0*(math.sqrt(ASQUARE)-1),2)})^2]',
            # ha='center', va='center', fontsize=9, fontstyle='italic'
        ) ,
        Plots( # 0 3
            text1Text = 'Price Curve dX/dY\n\n',
            #ha='left', va='center', fontsize=8,fontweight='light',fontstyle='italic"
            text2Text = f'\n\ndX/dY=-X/Y\ndXv/dYv=-{ASQUARE*INVARIANT}/(Yv^2)\ndX/dY=-[(X+{round(X0*(math.sqrt(ASQUARE)-1),2)})^2]/{ASQUARE*INVARIANT}',
            # ha='center', va='center', fontsize=9, fontstyle='italic'
        ) ,
    ],
    [
        Plots(# 1 0
            text1Text = 'X*Y\n=\nX0*Y0', 
            # ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
            text2YCoords=0.3, 
            text2Text='The Reference Bonding Curve',
            # ha='center', va='center', fontsize=9, fontstyle='italic'
        ) ,
        Plots(# 1 1
            xplot = np.linspace(20, 200, 10000), #xplot
            yplot = f(np.linspace(20, 200, 10000),INVARIANT),
            xLabel = 'X',
            yLabel = 'Y',
        ) ,
        Plots(# 1 2
            xplot = np.linspace(20, 200, 10000),
            yplot= dydx(np.linspace(20, 200, 10000), f(np.linspace(20, 200, 10000), INVARIANT)),
            color = 'orange',
            xLabel = 'X',
            yLabel = 'dY/dX',
        ) ,
        
        Plots(# 1 3
            xplot = f(np.linspace(20, 200, 10000), INVARIANT),
            yplot= dxdy(np.linspace(20, 200, 10000), f(np.linspace(20, 200, 10000), INVARIANT)),
            color = 'green',
            xLabel = 'Y',
            yLabel = 'dX/dY',
        ) ,
        
    ],
    [
        Plots(# 2 0
            text1Text = 'Xv*Yv\n=\nA^2*INVARIANT', 
            # ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
            text2YCoords=0.3, 
            text2Text='The Bancor v2 Virtual Curve'
            # ha='center', va='center', fontsize=9, fontstyle='italic'
        ) ,
        Plots(# 2 1
            xplot = np.linspace(PHIGH_POINT[0], PLOW_POINT[0], 500), #xplot
            yplot= f_concentrated1(np.linspace(PHIGH_POINT[0], PLOW_POINT[0], 500), INVARIANT, ASQUARE),
            xLabel = 'Xv',
            yLabel = 'Yv',
        ) ,
        Plots(# 2 2
            xplot = np.linspace(PLOW_POINT[0], PHIGH_POINT[0], 500),
            yplot= dyvdxv(np.linspace(PLOW_POINT[0], PHIGH_POINT[0], 500), INVARIANT, ASQUARE),
            color = 'orange',
            xLabel = 'Xv',
            yLabel = "dYv/dXv",
        ) ,
        Plots(# 2 3
            xplot = np.linspace(PHIGH_POINT[0], PLOW_POINT[0], 500),
            yplot= dxvdyv(np.linspace(PHIGH_POINT[0], PLOW_POINT[0], 500), INVARIANT, ASQUARE),
            color = 'green',
            xLabel = 'Yv',
            yLabel = "dXv/dYv",
        ) ,
    ],
    [
        Plots(# 3 0
            text1Text = '(X+X0*(A-1))\n*\n(Y+Y0*(A-1))\n=\nA^2*Invariant', 
            # ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
            text2YCoords=0.2, 
            text2Text='The Bancor v2 Real Curve',
            # ha='center', va='center', fontsize=9, fontstyle='italic'
        ) ,
        Plots(# 3 1
            xplot = np.linspace(0, (X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1), 10000),
            yplot= f_concentrated2(np.linspace(0, (X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1), 10000), INVARIANT, ASQUARE, X0, Y0),
            xLabel = 'X',
            yLabel = 'Y',
        ) ,
        Plots(# 3 2
            xplot = np.linspace(0, (X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1), 500),
            yplot= dYdX(np.linspace(0, (X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1), 500),INVARIANT, ASQUARE, X0),
            color = 'orange',
            xLabel = 'X',
            yLabel = 'dY/dX',
        ) ,
        Plots(# 3 3      # Yplot = f_concentrated2(Xplot)   AND    dXdYplot = 1/dYdX(Xplot)  
            xplot = f_concentrated2(np.linspace(0, (X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1), 1000), INVARIANT, ASQUARE, X0, Y0),
            yplot= 1/dYdX(np.linspace(0, (X0*(2*math.sqrt(ASQUARE)-1))/(math.sqrt(ASQUARE)-1), 1000),INVARIANT, ASQUARE, X0),
            color = 'green',
            xLabel = 'Y',
            yLabel = 'dX/dY',
        ) ,
    ]
]
