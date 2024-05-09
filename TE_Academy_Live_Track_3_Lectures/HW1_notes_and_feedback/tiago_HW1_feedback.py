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

xplot = np.linspace(25, 250, 500)
yplot = f(xplot)

def initializeAMM():
    global x, y, xtrade, ytrade
    x = 100
    y = invariant / x
    xtrade = []
    ytrade = []

def trade():
    global x, y, xtrade, ytrade
    nextx = nexty = 0
    while nextx <= 0 or nexty <= 0:
        deltax = norm.rvs(loc=0, scale=30, size=1)
        nextx = x + deltax[0]
        nexty = invariant / nextx if nextx != 0 else 0
    xtrade.append([x, nextx])
    ytrade.append([y, nexty])
    x, y = nextx, nexty

initializeAMM()
for _ in range(NUMBER_OF_TRADES):
    trade()

# Creating subplots for each trade
for i in range(NUMBER_OF_TRADES):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot with the actual trades
    ax1.plot(xplot, yplot)
    ax1.scatter([xtrade[i][0], xtrade[i][1]], [ytrade[i][0], ytrade[i][1]], color=['blue', 'red'], s=50)
    ax1.set_title(f"Trade Nº {i + 1} (TRAVERSAL UPON IMPLICIT CURVE)")
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
    legend_elements = [
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=10, label=f'Before Trade ({round(xtrade[i][0],2)}, {round(ytrade[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker='o', markerfacecolor='red', markeredgecolor='red', markersize=10, label=f'After Trade ({round(xtrade[i][1],2)}, {round(ytrade[i][1],2)})'),
        mlines.Line2D([], [], color='none', label=f'Δx = {xtrade[i][1] - xtrade[i][0]:.2f}'),
        mlines.Line2D([], [], color='none', label=f'Δy = {ytrade[i][1] - ytrade[i][0]:.2f}'),
        mlines.Line2D([], [], color='none', label=f'Initial Marginal Rate = {- ytrade[i][0]/xtrade[i][0]:.2f}'),
        mlines.Line2D([], [], color='none', label=f'Effective Rate = {- abs((ytrade[i][1] - ytrade[i][0]) / (xtrade[i][1] - xtrade[i][0])):.2f}'),
        mlines.Line2D([], [], color='none', label=f'Final Marginal Rate = {- ytrade[i][1]/xtrade[i][1]:.2f}'),
    ]
    ax1.legend(handles=legend_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='12', fontsize='10', framealpha=1)

    # Second subplot empty
    ax2.set_title("PRICE CURVE GOES HERE")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()