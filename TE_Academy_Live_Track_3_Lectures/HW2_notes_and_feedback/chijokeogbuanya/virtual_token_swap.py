import matplotlib.pyplot as plt
import numpy as np

# Initial conditions
initial_TKNX_balance = 20
initial_TKNY_balance = 200
A = 2  # Scaling factor

# Calculate the initial invariant for the virtual curve
initial_invariant = initial_TKNX_balance * initial_TKNY_balance

# Virtual initial balances
initial_TKNX_virtual = A * initial_TKNX_balance
initial_TKNY_virtual = A * initial_TKNY_balance

# Change in Token X (swap scenario)
delta_TKNX = 30

# New token balances after swap
new_TKNX_balance = initial_TKNX_balance + delta_TKNX
new_TKNX_virtual = A * new_TKNX_balance

# Using the invariant equation for the virtual balances: x_v * y_v = A^2 * x_0 * y_0
new_TKNY_virtual = (A**2 * initial_invariant) / new_TKNX_virtual

# Calculate delta_TKNY_virtual
delta_TKNY_virtual = new_TKNY_virtual - initial_TKNY_virtual



# this change in y 
delta_TKNY = -((delta_TKNX * A**2 * initial_TKNX_balance * initial_TKNY_balance) / (new_TKNX_virtual * (new_TKNX_virtual + delta_TKNX)))
 


# Marginal rates using the provided formulas
# ∂yv / ∂xv = - (yv / xv) = - (A * y0 / A * x0) = - (y0 / x0)
initial_marginal_rate = - (initial_TKNY_virtual / initial_TKNX_virtual)
final_marginal_rate = - (new_TKNY_virtual / new_TKNX_virtual)



# Effective price using the virtual deltas
effective_price = delta_TKNY_virtual / (delta_TKNX * A)

# Print the calculated values
print("Initial Marginal Price:", initial_marginal_rate)
print("Final Marginal Price:", final_marginal_rate)



# Calculate phigh and plow
phigh = (A**2 / (A - 1)**2) * (initial_TKNY_balance / initial_TKNX_balance)
plow = ((A - 1)**2 / A**2) * (initial_TKNY_balance / initial_TKNX_balance)

# Print the calculated values
print("Initial Marginal Price:", initial_marginal_rate)
print("Final Marginal Price:", final_marginal_rate)
print("Effective Price:", effective_price)
print("Phigh:", phigh)
print("Plow:", plow)
print("Change in X:", delta_TKNX)
print("Change in Virtual Y:", delta_TKNY_virtual)
print("Change in Virtual actual Y:", delta_TKNY)
print("actual Y after:", delta_TKNY + initial_TKNY_balance)
print("Virtual Y balance:", new_TKNY_virtual)
print("Virtual X balance:", new_TKNX_virtual)

print("Actual Change in Y:", delta_TKNY_virtual / A)
# print("Actual Y balance:", new_TKNY_virtual / A)

# Range of token balances for plotting
x_range = np.linspace(1, 200, 100)
y_range_virtual = (A**2 * initial_invariant) / (A * x_range)

# Plot the bonding curve and price curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the price curve for the virtual invariant
ax1.plot(A * x_range, -(initial_invariant / (x_range**2)), 
         label='Price Curve: ∂y_v/∂x_v = -A^2 · x0 · y0 / (A · x)^2\n'
               f'Initial Virtual X Balance: {initial_TKNX_virtual}\n'
               f'Final Virtual X Balance: {new_TKNX_virtual}\n'
               f'Effective Price: {effective_price}', 
         color='green')
ax1.scatter(initial_TKNX_virtual, initial_marginal_rate, color='red', label=f'Initial Marginal rate (x0, dy_v/dx_v): {initial_marginal_rate}')
ax1.scatter(new_TKNX_virtual, final_marginal_rate, color='blue', label=f'Final Marginal rate (x1, dy_v/dx_v): {final_marginal_rate}')

# Arrows indicating the change in price direction for the virtual curve
arrow_length_x_virtual = delta_TKNX * A * 0.9
arrow_length_y_virtual = (final_marginal_rate - initial_marginal_rate) * 0.88

# Red arrow (ΔX) from initial to new TKNX virtual
ax1.arrow(initial_TKNX_virtual, initial_marginal_rate, arrow_length_x_virtual, 0, color='red', linestyle='--', linewidth=1, head_width=1, head_length=5, label='ΔX')


# Blue arrow (ΔY) from new TKNX virtual to new TKNY virtual
ax1.arrow(new_TKNX_virtual, initial_marginal_rate, 0, arrow_length_y_virtual, color='blue', linestyle='--', linewidth=1, head_width=5, head_length=1, label='ΔY')

ax1.set_xlabel('Virtual Token X Balance')
ax1.set_ylabel('dy_v/dx_v')
ax1.set_title('Virtual Price Curve visualization')
ax1.legend(loc='lower right')
ax1.grid(True)
ax1.set_ylim(-20, 0)



# Plot the bonding curve
ax2.plot(A * x_range, y_range_virtual, label='Bonding Curve: y_v = A^2 · x0 · y0 / (A · x)', color='blue')
ax2.scatter(initial_TKNX_virtual, initial_TKNY_virtual, color='red', label=f'Initial Balances (x0, y0): ({initial_TKNX_virtual}, {initial_TKNY_virtual})')
ax2.scatter(new_TKNX_virtual, new_TKNY_virtual, color='blue', label=f'Final Balances (x1, y1): ({new_TKNX_virtual}, {new_TKNY_virtual})')




# Adjust the arrows to be shorter and start from the correct points
arrow_length_x = delta_TKNX * A * 0.97
arrow_length_y = delta_TKNY_virtual * 0.97

# Red arrow (ΔX) from initial to new TKNX virtual
ax2.arrow(initial_TKNX_virtual, initial_TKNY_virtual, arrow_length_x, 0, color='red', linestyle='--', linewidth=1, head_width=5, head_length=5, label='ΔX')

# Blue arrow (ΔY) from new TKNX virtual to new TKNY virtual
ax2.arrow(new_TKNX_virtual, initial_TKNY_virtual, 0, arrow_length_y, color='blue', linestyle='--', linewidth=1, head_width=5, head_length=5, label='ΔY')

ax2.set_xlabel('Virtual Token X Balance')
ax2.set_ylabel('Virtual Token Y Balance')
ax2.set_title('Virtual Bonding Curve visualization')
ax2.legend(loc='upper right')
ax2.grid(True)
ax2.set_ylim(0, 500)

# Caption
caption = f"These graphs depict a token swap performed for a system initially comprising {initial_TKNX_balance} TKNX and {initial_TKNY_balance} TKNY,\nwhere the TKNX balance is increased by {delta_TKNX} tokens and the virtual TKNY balance is decreased by {abs(delta_TKNY_virtual):.2f} TKNY tokens.\nThe initial marginal rate is {initial_marginal_rate}, the final marginal rate is {final_marginal_rate}, and the effective rate of exchange for the swap is {effective_price}."
fig.text(0.5, 0.01, caption, ha='center')

plt.tight_layout()
plt.show()