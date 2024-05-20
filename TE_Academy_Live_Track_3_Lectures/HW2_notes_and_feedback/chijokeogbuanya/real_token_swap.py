import matplotlib.pyplot as plt
import numpy as np

# Initial conditions
initial_TKNX_balance = 20
initial_TKNY_balance = 200
A = 2  # Scaling factor

# Calculate phigh and plow
phigh = (A**2 / (A - 1)**2) * (initial_TKNY_balance / initial_TKNX_balance)
plow = ((A - 1)**2 / A**2) * (initial_TKNY_balance / initial_TKNX_balance)

# Calculate the y-intercept and x-intercept for the real curve
yint = (initial_TKNY_balance * (2 * A - 1)) / (A - 1)
xint = (initial_TKNX_balance * (2 * A - 1)) / (A - 1)

# Using the real invariant equation (x + x0 * (A - 1)) * (y + y0 * (A - 1)) = A^2 * x0 * y0
# where x and y are the real balances of tokens X and Y.
# initial invariant for the real curve
initial_invariant_real = A**2 * initial_TKNX_balance * initial_TKNY_balance

# Change in Token X (swap scenario)
delta_TKNX = 30

# New token balances after swap
new_TKNX_balance = initial_TKNX_balance + delta_TKNX

# Calculate new TKNY balance after swap using the real curve equation
new_TKNY_balance = (initial_invariant_real / ((A - 1) * new_TKNX_balance + initial_TKNX_balance)) - initial_TKNY_balance * (A - 1)


initial_marginal_rate = -((initial_TKNY_balance + initial_TKNY_balance * (A - 1)) / (initial_TKNX_balance + initial_TKNX_balance * (A - 1)))
final_marginal_rate = -((new_TKNY_balance + initial_TKNY_balance * (A - 1)) / (new_TKNX_balance + initial_TKNX_balance * (A - 1)))


# Calculate change in Y balance using the provided formula
delta_TKNY = -((delta_TKNX * A**2 * initial_TKNX_balance * initial_TKNY_balance) / 
              ((initial_TKNX_balance + initial_TKNX_balance * (A - 1)) * 
               (initial_TKNX_balance + delta_TKNX + initial_TKNX_balance * (A - 1))))

# Check if the calculated new balance matches with the change in Y balance
new_TKNY_balance_check = initial_TKNY_balance + delta_TKNY

# Print the calculated values
print("Initial Marginal Rate:", initial_marginal_rate)
print("Final Marginal Rate:", final_marginal_rate)
print("Effective Price:", delta_TKNY / delta_TKNX)
print("Change in Y Balance:", delta_TKNY)
print("New Y Balance (from delta Y):", new_TKNY_balance_check)
print("New Y Balance (from invariant):", new_TKNY_balance)


# Calculate new TKNY balance after swap using the real curve equation
new_TKNY_balance = (initial_invariant_real / ((A - 1) * new_TKNX_balance + initial_TKNX_balance)) - initial_TKNY_balance * (A - 1)

# Range of token balances for plotting
x_range = np.linspace(1, 100, 100)
y_range_real = (initial_invariant_real / ((A - 1) * x_range + initial_TKNX_balance)) - initial_TKNY_balance * (A - 1)

# Plot the bonding curve and price curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


# Plot the price curve
# Using the provided ∂y/∂x formula for the price curve
def price_curve_derivative(x, x0, y0, A):
    return -((A**2 * x0 * y0) / ((x + x0 * (A - 1))**2))

# Effective Price
effective_price = delta_TKNY/delta_TKNX
print("Effective Price:", effective_price)

# Plot the price curve
x_range_real = np.linspace(1, new_TKNX_balance * 2, 100)
y_range_real_invariant = price_curve_derivative(x_range_real, initial_TKNX_balance, initial_TKNY_balance, A)

ax1.plot(x_range_real, y_range_real_invariant, 
         label=f'Price Curve: ∂y/∂x = -(x + x0 · (A − 1))^2/(A^2 · x0 · y0)\n'
               f'Initial Real X Balance: {initial_TKNX_balance}\n'
               f'Final Real X Balance: {new_TKNX_balance}\n'
               f'Effective Price: {effective_price}',
         color='green')


# Scatter plot for initial and final marginal rates on price curve
ax1.scatter(initial_TKNX_balance, initial_marginal_rate, color='red', label=f'Initial Marginal Rate: {initial_marginal_rate}')
ax1.scatter(new_TKNX_balance, final_marginal_rate, color='blue', label=f'Final Marginal Rate: {final_marginal_rate}')


# Add arrows to indicate direction of change in price curve
ax1.arrow(initial_TKNX_balance, initial_marginal_rate, delta_TKNX * 0.9, 0, color='red', linestyle='--', linewidth=1, head_width=0.5, head_length=2)
ax1.arrow(new_TKNX_balance, initial_marginal_rate, 0, (final_marginal_rate - initial_marginal_rate) * 0.86, color='blue', linestyle='--', linewidth=1, head_width=2, head_length=1)

# Add arrows to indicate direction of change in bonding curve
ax2.arrow(initial_TKNX_balance, initial_TKNY_balance, delta_TKNX * 0.9, 0, color='red', linestyle='--', linewidth=1, head_width=5, head_length=5)
ax2.arrow(new_TKNX_balance, initial_TKNY_balance, 0, delta_TKNY * 0.9, color='blue', linestyle='--', linewidth=1, head_width=5, head_length=5)

ax1.set_xlabel('Token X Balance')
ax1.set_ylabel('dy/dx')
ax1.set_title('Real Price Curve Visualization')
ax2.legend(loc='lower right')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(-20, 0)


# Print the maximum y-value in the range
print("Maximum y-value:", max(y_range_real))



# Plot the bonding curve
ax2.plot(x_range, y_range_real, label='Bonding Curve (Real): (x + x_0 * (A - 1)) * (y + y_0 * (A - 1)) = A^2 * x_0 * y_0', color='blue')

ax2.scatter(initial_TKNX_balance, initial_TKNY_balance, color='red', label=f'Initial Balances (x0, y0): ({initial_TKNX_balance}, {initial_TKNY_balance})')
ax2.scatter(new_TKNX_balance, new_TKNY_balance, color='blue', label=f'Final Balances (x1, y1): ({new_TKNX_balance}, {new_TKNY_balance})')

# Annotate phigh and plow
ax2.annotate(f'phigh: {phigh}', xy=(initial_TKNX_balance, initial_TKNY_balance), xytext=(initial_TKNX_balance + 20, initial_TKNY_balance + 50), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax2.annotate(f'plow: {plow}', xy=(initial_TKNX_balance, initial_TKNY_balance), xytext=(initial_TKNX_balance + 20, initial_TKNY_balance - 50), arrowprops=dict(facecolor='black', arrowstyle='->'))

# Annotate y-intercept and x-intercept
ax2.annotate(f'yint: {yint:.2f}', xy=(0, yint), xytext=(10, yint - 50), arrowprops=dict(facecolor='green', arrowstyle='->'))
ax2.annotate(f'xint: {xint:.2f}', xy=(xint, 0), xytext=(xint + 10, 50), arrowprops=dict(facecolor='green', arrowstyle='->'))

# Calculate the change in balance for the arrow lengths
arrow_length_delta_x = delta_TKNX  * 0.9
arrow_length_delta_y = (new_TKNY_balance - initial_TKNY_balance) * 0.89

# Red arrow (ΔX) from initial to new TKNX balance
ax2.arrow(initial_TKNX_balance, initial_TKNY_balance, arrow_length_delta_x, 0, color='red', linestyle='--', linewidth=1, head_width=5, head_length=4, label='ΔX')

# Blue arrow (ΔY) from initial to new TKNY balance
ax2.arrow(new_TKNX_balance, initial_TKNY_balance, 0, arrow_length_delta_y, color='blue', linestyle='--', linewidth=1, head_width=5, head_length=6, label='ΔY')

ax2.set_xlabel('Token X Balance')
ax2.set_ylabel('Token Y Balance')
ax2.set_title('Real Bonding Curve Visualization')
ax2.legend(loc='upper right')
ax2.grid(True)
ax2.set_ylim(0, yint + 50) 
ax2.set_xlim(0, xint + 50)


# Adding caption
caption = "These graphs depict a token swap performed for a system initially comprising {} TKNX and {} TKNY,\n where the TKNX balance is {} by {} tokens and the TKNY balance is {} by {} TKNY tokens.\n The initial marginal rate is {}, the final marginal rate is {}, and the effective rate of exchange for the swap is {}."
caption = caption.format(initial_TKNX_balance, initial_TKNY_balance, "increased" if delta_TKNX > 0 else "decreased", abs(delta_TKNX), "increased" if delta_TKNY > 0 else "decreased", abs(delta_TKNY), initial_marginal_rate, final_marginal_rate, effective_price)
fig.text(0.5, 0.01, caption, ha='center')

plt.tight_layout()
plt.show()
