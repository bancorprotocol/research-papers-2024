import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from Constants import *
from Equations import *
from Trades import *
from Plots import *


initializeAMM()
for _ in range(NUMBER_OF_TRADES):
    xtrade,ytrade, xprice_curve, yprice_curve, xvtrade, yvtrade, xvprice_curve, yvprice_curve, Xtrade, Ytrade, Xprice_curve, Yprice_curve = trade()

def plot_trades(xtrade,ytrade, xprice_curve, yprice_curve, xvtrade, yvtrade, xvprice_curve, yvprice_curve, Xtrade, Ytrade, Xprice_curve, Yprice_curve):
    for n in range(NUMBER_OF_TRADES):
        # Create a figure
        fig = plt.figure(figsize=(16, 10.5), facecolor='#eeeef2')

        # Define width and height ratios for each column and row respectively
        width_ratios = [0.35, 2, 2, 2]
        height_ratios = [0.35, 2, 2, 2]

        # Create a GridSpec with 4x4 layout and specify width and height ratios
        gs = gridspec.GridSpec(4, 4, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)

        # Add subplots to the GridSpec and store them in a 2D list
        axes = [
            [fig.add_subplot(gs[i, j]) for j in range(4)]
            for i in range(4)
        ]

        # Iterate through each subplot to edit it
        for i in range(4):
            for j in range(4):
                ax = axes[i][j]
                if i == 0 or j == 0:
                    ax.axis('off')
                    if i==j:
                        ax.text(
                            plots[i][j].text1YCoords,
                            plots[i][j].text1XCoords, 
                            plots[i][j].text1Text, 
                            ha='left', va='center', fontsize=8, fontstyle='italic',fontweight='light'
                        )
                    else:
                        ax.text(
                            plots[i][j].text1XCoords, 
                            plots[i][j].text1YCoords, 
                            plots[i][j].text1Text, 
                            ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
                        )
                        ax.text(
                            plots[i][j].text2XCoords, 
                            plots[i][j].text2YCoords, 
                            plots[i][j].text2Text, 
                            ha='center', va='center', fontsize=9, fontstyle='italic'
                        )
                else:
                    # Plot
                    ax.plot(plots[i][j].xplot, plots[i][j].yplot, color=plots[i][j].color)
                    plots[i][j].get_scatter_data(n,i,j,xtrade,ytrade, xprice_curve, yprice_curve, xvtrade, yvtrade, xvprice_curve, yvprice_curve, Xtrade, Ytrade, Xprice_curve, Yprice_curve)
                    ax.scatter(
                        plots[i][j].scatterPointsX,
                        plots[i][j].scatterPointsY,
                        color=plots[i][j].scatterColors,
                        edgecolors=plots[i][j].scatterEdgeColors,
                        s=80
                    )
                    
                    # Labels
                    ax.set_xlabel(plots[i][j].xLabel)
                    ax.set_ylabel(plots[i][j].yLabel)
                    ax.grid(True)
                    
                    # Arrows
                    arrow1 = FancyArrowPatch(plots[i][j].arrow1[0], plots[i][j].arrow1[1], arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
                    arrow2 = FancyArrowPatch(plots[i][j].arrow2[0], plots[i][j].arrow2[1], arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
                    arrow3 = FancyArrowPatch(plots[i][j].arrow3[0], plots[i][j].arrow3[1], arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
                    ax.add_patch(arrow1)
                    ax.add_patch(arrow2)
                    ax.add_patch(arrow3)
                    
                    # Integral Area
                    if j>=2:
                        ax.fill_between(plots[i][j].integralX, plots[i][j].integralY, alpha=0.5, color = '#c6c6cd') 
                    
                    # Legend
                    if i == 1 and j ==1:
                        ax_elements = [
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=5, label=f'{round(plots[i][j].scatterPointsX[0],2)}, {round(plots[i][j].scatterPointsY[0],2)}'),
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor='red', markeredgecolor='red', markersize=5, label=f'{round(plots[i][j].scatterPointsX[1],2)}, {round(plots[i][j].scatterPointsY[1],2)}'),
                            mlines.Line2D([], [], color='none', label=plots[i][j].legendXtraLine1['label']),
                            mlines.Line2D([], [], color='none', label=plots[i][j].legendXtraLine2['label']),
                            mlines.Line2D([], [], color='none', label=plots[i][j].legendXtraLine3['label'])
                        ]
                    if i==1 and j>1:
                        ax_elements = [
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=5, label=f'{round(plots[i][j].scatterPointsX[0],2)}, {round(plots[i][j].scatterPointsY[0],2)}'),
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor='red', markeredgecolor='red', markersize=5, label=f'{round(plots[i][j].scatterPointsX[1],2)}, {round(plots[i][j].scatterPointsY[1],2)}'),
                            mlines.Line2D([], [], color='none', label=plots[i][j].legendXtraLine1['label'])
                        ] 
                    elif i>=2 and j>=2:
                        ax_elements = [
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=5, label=f'{round(plots[i][j].scatterPointsX[0],2)}, {round(plots[i][j].scatterPointsY[0],2)}'),
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor='red', markeredgecolor='red', markersize=5, label=f'{round(plots[i][j].scatterPointsX[1],2)}, {round(plots[i][j].scatterPointsY[1],2)}'),
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor=plots[i][j].legendXtraLine1['markerfacecolor'], markeredgecolor=plots[i][j].legendXtraLine1['markeredgecolor'], label=plots[i][j].legendXtraLine1['label']),
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor=plots[i][j].legendXtraLine2['markerfacecolor'], markeredgecolor=plots[i][j].legendXtraLine2['markeredgecolor'], label=plots[i][j].legendXtraLine2['label']),
                            mlines.Line2D([], [], color='none', label=plots[i][j].legendXtraLine3['label'])
                        ]
                        
                    elif i>=2 and j==1:
                        ax_elements = [
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=5, label=f'{round(plots[i][j].scatterPointsX[0],2)}, {round(plots[i][j].scatterPointsY[0],2)}'),
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor='red', markeredgecolor='red', markersize=5, label=f'{round(plots[i][j].scatterPointsX[1],2)}, {round(plots[i][j].scatterPointsY[1],2)}'),
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor=plots[i][j].legendXtraLine1['markerfacecolor'], markeredgecolor=plots[i][j].legendXtraLine1['markeredgecolor'], label=plots[i][j].legendXtraLine1['label']),
                            mlines.Line2D([], [], color='none', marker='o', markerfacecolor=plots[i][j].legendXtraLine2['markerfacecolor'], markeredgecolor=plots[i][j].legendXtraLine2['markeredgecolor'], label=plots[i][j].legendXtraLine2['label']),
                            mlines.Line2D([], [], color='none', label=plots[i][j].legendXtraLine3['label']),
                            mlines.Line2D([], [], color='none', label=plots[i][j].legendXtraLine4['label']),
                            mlines.Line2D([], [], color='none', label=plots[i][j].legendXtraLine5['label'])
                        ]
                    ax.legend(handles=ax_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='10', fontsize='7', framealpha=1)
        plt.tight_layout()
        plt.show()

plot_trades(xtrade,ytrade, xprice_curve, yprice_curve, xvtrade, yvtrade, xvprice_curve, yvprice_curve, Xtrade, Ytrade, Xprice_curve, Yprice_curve)