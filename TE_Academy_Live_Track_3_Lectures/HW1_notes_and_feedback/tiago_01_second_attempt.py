import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch

# The number of trades we want to simulate
NUMBER_OF_TRADES = 3
invariant = 10000  # Invariant = x0 * y0

def f(xplot):
    return invariant / xplot

def dydx(xplot, yplot):
    return -yplot/xplot

xplot = np.linspace(25, 250, 500)
yplot = f(xplot)
dydxplot = dydx(xplot, yplot)

def initializeAMM():
    global x, y, yprice_curve, xtrade, ytrade
    x = 100
    y = invariant / x
    xtrade = []
    ytrade = []
    yprice_curve = []

def trade():
    global x, y, yprice_curve, xtrade, ytrade
    nextx = nexty = 0
    while nextx <= 0 or nexty <= 0:
        deltax = norm.rvs(loc=0, scale=30, size=1)
        nextx = x + deltax[0]
        nexty = invariant / nextx if nextx != 0 else 0
        
    xtrade.append([x, nextx])
    ytrade.append([y, nexty])
    yprice_curve.append([-y/x, -nexty/nextx])
    x, y = nextx, nexty
    
initializeAMM()
for _ in range(NUMBER_OF_TRADES):
    trade()

# Creating subplots for each trade
for i in range(NUMBER_OF_TRADES):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # subplots are arranged in 1 row and 2 columns, size of each subplot

    # First subplot with the actual trades
    ax1.plot(xplot, yplot)
    ax1.scatter(
        [xtrade[i][0], xtrade[i][1]], 
        [ytrade[i][0], ytrade[i][1]], 
        color=['blue', 'red'], 
        s=80
    )
    ax1.set_title(f"Bonding curve for trade Nº {i + 1}")
    ax1.set_xlabel("Token X Balance")
    ax1.set_ylabel("Token Y Balance")
    ax1.grid(True)
    
    # Adding arrows
    arrow1 = FancyArrowPatch((xtrade[i][0], ytrade[i][0]), (xtrade[i][0], ytrade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow2 = FancyArrowPatch((xtrade[i][0], ytrade[i][1]), (xtrade[i][1], ytrade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow3 = FancyArrowPatch((xtrade[i][0], ytrade[i][0]), (xtrade[i][1], ytrade[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow2)
    ax1.add_patch(arrow3)

    # Detailed legend as in the original code
    legend_elements_ax1 = [
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=10, label=f'Before Trade ({round(xtrade[i][0],2)}, {round(ytrade[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='red', markeredgecolor='red', markersize=10, label=f'After Trade ({round(xtrade[i][1],2)}, {round(ytrade[i][1],2)})'),
        # line = plt.plot(
        # xplot, 
        # yplot, 
        # label=r"$y=x0*y0/x$ AND x0*y0={}".format(invariant))[0]
        mlines.Line2D([], [], color='#2596be', markeredgecolor='light-blue', markersize=10, label=r"$y=x0*y0/x$ AND x0*y0={}".format(invariant)),
        mlines.Line2D([], [], color='none', label=f'Δx = {xtrade[i][1] - xtrade[i][0]:.2f}'),
        mlines.Line2D([], [], color='none', label=f'Δy = {ytrade[i][1] - ytrade[i][0]:.2f}'),
        mlines.Line2D([], [], color='none', label=f'Initial Marginal Rate = {-ytrade[i][0]/xtrade[i][0]:.2f}'),
        mlines.Line2D([], [], color='none', label=f'Effective Rate = {- abs((ytrade[i][1] - ytrade[i][0]) / (xtrade[i][1] - xtrade[i][0])):.2f}'),
        mlines.Line2D([], [], color='none', label=f'Final Marginal Rate = {-ytrade[i][1]/xtrade[i][1]:.2f}'),
    ]
    ax1.legend(handles=legend_elements_ax1, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='12', fontsize='10', framealpha=1)

    # Second subplot empty
    ax2.set_title("PRICE CURVE GOES HERE")
    
    # ax2.axis('off')
    ax2.plot(xplot, dydxplot, color = 'orange')
    ax2.scatter(
        [xtrade[i][0], xtrade[i][1]], 
        [yprice_curve[i][0], yprice_curve[i][1]], 
        color=['blue', 'red'], 
        s=80
    )
    
    ax2.set_title(f"Price curve for trade Nº {i + 1}")
    ax2.set_xlabel("Token X Balance")
    ax2.set_ylabel("dy/dx")
    ax2.grid(True)
    
    
    # Detailed legend as in the original code
    legend_elements_ax2 = [
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=10, label=f'Before Trade ({round(xtrade[i][0],2)}, {round(yprice_curve[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='red', markeredgecolor='red', markersize=10, label=f'After Trade ({round(xtrade[i][1],2)}, {round(yprice_curve[i][1],2)})'),
        mlines.Line2D([], [], color='orange', markersize=10, label=r"dy/dx = -y/x"),
        mlines.Line2D([], [], color='none', label=f'Δx = {xtrade[i][1] - xtrade[i][0]:.2f}'),
        mlines.Line2D([], [], color='none', label=f'|Δy| = {abs(ytrade[i][1] - ytrade[i][0]):.2f}'),
        mlines.Line2D([], [], color='none', label=f'Initial Spot Price= {ytrade[i][0]/xtrade[i][0]:.2f}'),
        mlines.Line2D([], [], color='none', label=f'Effective Price = {abs((ytrade[i][1] - ytrade[i][0]) / (xtrade[i][1] - xtrade[i][0])):.2f}'),
        mlines.Line2D([], [], color='none', label=f'Final Spot Price = {ytrade[i][1]/xtrade[i][1]:.2f}'),
    ]
    ax2.legend(handles=legend_elements_ax2, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='12', fontsize='10', framealpha=1)

    # Plotting the integral area
    sectionx = np.linspace(xtrade[i][0], xtrade[i][1], 500)
    section_dydx = -f(sectionx)/sectionx 
    ax2.fill_between(sectionx, section_dydx, alpha=0.5) 

    # Adding text
    # Calculate the midpoint of the x and y ranges
    mid_x = (xtrade[i][0] + xtrade[i][1]) / 2
    mid_y = yprice_curve[i][1] / 2 + 0.1
    ax2.text(mid_x, mid_y, '|Δy|', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')
    
    # Adding arrows
    arrow4 = FancyArrowPatch((xtrade[i][0], yprice_curve[i][0]), (xtrade[i][0], yprice_curve[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow5 = FancyArrowPatch((xtrade[i][0], yprice_curve[i][1]), (xtrade[i][1], yprice_curve[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow6 = FancyArrowPatch((xtrade[i][0], yprice_curve[i][0]), (xtrade[i][1], yprice_curve[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    ax2.add_patch(arrow4)
    ax2.add_patch(arrow5)
    ax2.add_patch(arrow6)
    
    plt.tight_layout()
    plt.show()
