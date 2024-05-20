# Imports
import matplotlib.pyplot as plt
import numpy as np

# initial token balances
initial_TKNX_balance = 20
initial_TKNY_balance = 200

# Bob's token swap scenario
delta_TKNX = 30
delta_TKNY = -((delta_TKNX * initial_TKNY_balance) / (initial_TKNX_balance + delta_TKNX))

# Calculate the new Token X and Token Y balances after the swap
new_TKNX_balance = initial_TKNX_balance + delta_TKNX
new_TKNY_balance = initial_TKNY_balance + delta_TKNY

# Calculate the initial invariants
initial_invariant = initial_TKNX_balance * initial_TKNY_balance


# Calculate the initial and final marginal prices
initial_marginal_price = initial_TKNY_balance / initial_TKNX_balance
final_marginal_price = new_TKNY_balance / new_TKNX_balance


# Marginal Price = - (y / x)
# the initial and final marginal prices
initial_marginal_price = - (initial_TKNY_balance / initial_TKNX_balance)
final_marginal_price = - (new_TKNY_balance / new_TKNX_balance)


# Effective Price = delta_TKNY / delta_TKNX = - (y / (x + delta_TKNX))
# Calculate the effective price
effective_price = - (initial_TKNY_balance / (initial_TKNX_balance + delta_TKNX))




# Print the calculated values
print("Initial Marginal Price:", initial_marginal_price)
print("Final Marginal Price:", final_marginal_price)
print("Effective Price:", effective_price)
print("Change in X:", delta_TKNX)
print("Change in Y:", delta_TKNY)
print("Y balance:", new_TKNY_balance)
print("X balance:", new_TKNX_balance)


# range of token balances for plotting bonding curve
x_range = np.linspace(0, 200, 100)
y_range = initial_invariant / x_range

# Plot the bonding curve with the token swap scenario
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))




# Plotting the price curve
ax1.plot(x_range, -(initial_TKNX_balance * initial_TKNY_balance) / (x_range ** 2), label='Price Curve: ∂y/∂x = -x0 · y0 / x^2\nInitial X Balance: {}\nFinal X Balance: {}\nEffective Price: {}'.format(initial_TKNX_balance, new_TKNX_balance, effective_price), color='green')
ax1.scatter(initial_TKNX_balance, -initial_TKNY_balance / initial_TKNX_balance, color='red', label=f'Initial Marginal rate (x0, dy/dx): {initial_marginal_price:.4f}')
ax1.scatter(new_TKNX_balance, -new_TKNY_balance / new_TKNX_balance, color='blue', label=f'Final Marginal rate (x1, dy/dx): {final_marginal_price:.4f}')

# Arrows indicating the change in price direction
arrow_length_x = delta_TKNX * 0.9
arrow_length_y = (final_marginal_price - initial_marginal_price) * 0.88
ax1.arrow(initial_TKNX_balance, -initial_TKNY_balance / initial_TKNX_balance, arrow_length_x, 0, color='red', linestyle='--', linewidth=1, head_width=1, head_length=5, label='ΔX')
ax1.arrow(initial_TKNX_balance + delta_TKNX, -initial_TKNY_balance / initial_TKNX_balance, 0, arrow_length_y, color='blue', linestyle='--', linewidth=1, head_width=5, head_length=1, label='ΔY')



ax1.set_xlabel('Token X Balance')
ax1.set_ylabel('dy/dx')
ax1.set_title('Reference Price Curve Visualization')
ax1.legend(loc='lower right')
ax1.grid(True)
ax1.set_ylim(-20, 0)


# Plot the bonding curve
ax2.plot(x_range, y_range, label='Bonding Curve: y = k / x', color='blue')
ax2.scatter(initial_TKNX_balance, initial_TKNY_balance, color='red', label=f'Initial Balances (x0, y0): ({initial_TKNX_balance}, {initial_TKNY_balance})')
ax2.scatter(new_TKNX_balance, new_TKNY_balance, color='green', label=f'Final Balances (x1, y1): ({new_TKNX_balance}, {new_TKNY_balance})')

# Adjust the arrows
arrow_length_x = delta_TKNX * 0.8
arrow_length_y = delta_TKNY * 0.9


# Red arrow (ΔX) from initial to new TKNX balance
ax2.arrow(initial_TKNX_balance, initial_TKNY_balance, arrow_length_x, 0, color='red', linestyle='--', linewidth=1, head_width=7, head_length=10, label='ΔX')


# Green arrow (ΔY) from new TKNX balance to new TKNY balance
ax2.arrow(new_TKNX_balance, initial_TKNY_balance, 0, arrow_length_y, color='green', linestyle='--', linewidth=1, head_width=7, head_length=10, label='ΔY')

ax2.set_xlabel('Token X Balance')
ax2.set_ylabel('Token Y Balance')
ax2.set_title('Reference Bonding Curve Visualization')
ax2.legend(loc='upper right')
ax2.grid(True)
ax2.set_ylim(0, 500)

# Adding caption
caption = "These graphs depict a token swap performed for a system initially comprising {} TKNX and {} TKNY,\n where the TKNX balance is {} by {} tokens and the TKNY balance is {} by {} TKNY tokens.\n The initial marginal rate is {}, the final marginal rate is {}, and the effective rate of exchange for the swap is {}."
caption = caption.format(initial_TKNX_balance, initial_TKNY_balance, "increased" if delta_TKNX > 0 else "decreased", abs(delta_TKNX), "increased" if delta_TKNY > 0 else "decreased", abs(delta_TKNY), initial_marginal_price, final_marginal_price, effective_price)
fig.text(0.5, 0.01, caption, ha='center')



plt.tight_layout()
plt.show()
