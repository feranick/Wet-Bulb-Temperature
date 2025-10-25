import math
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 1. Define the Wet Bulb Temperature function
def get_wet_bulb_temp(T, RH):
    """
    Calculates the Wet Bulb Temperature (Tw) based on Temperature (T) and 
    Relative Humidity (RH) using the provided formula.
    """
    T = float(T)
    RH = float(RH)
    
    if RH < 0:
        RH = 0 
        
    term1 = T * math.atan(0.151977 * math.sqrt(RH + 8.313659))
    term2 = math.atan(T + RH)
    term3 = math.atan(RH - 1.676331)
    term4 = 0.00391838 * math.pow(max(RH, 0), 1.5) * math.atan(0.023101 * RH)
    term5 = 4.686035
    
    Tw = term1 + term2 - term3 + term4 - term5
    return Tw

# 2. Define the equation for the solver
def equation_to_solve(unknown, fixed_var, solve_for, target_tw):
    """
    Generic function for fsolve, now accepting a 'target_tw'.
    If solve_for='T', 'unknown' is T and 'fixed_var' is RH.
    If solve_for='RH', 'unknown' is RH and 'fixed_var' is T.
    """
    val = unknown[0]
    
    if solve_for == 'T':
        T = val
        RH = fixed_var
    elif solve_for == 'RH':
        T = fixed_var
        RH = val
    
    # Use the passed 'target_tw' for the calculation
    return get_wet_bulb_temp(T, RH) - target_tw

# 3. Setup Plot
plt.figure(figsize=(12, 8))

# 4. Define Target Values
# You can change this list to any values you want to plot
#TARGET_TW_VALUES = [7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]
TARGET_TW_VALUES = [7, 8, 9, 10, 11, 12]

# To set the plot's x-limits properly, we'll find the overall min and max T
overall_t_min = float('inf')
overall_t_max = float('-inf')

print("Generating curves for multiple wet bulb temperatures...")

# 5. Loop and Generate Curves
for target_tw in TARGET_TW_VALUES:
    
    print(f"--- Processing Tw = {target_tw}°C ---")
    
    # --- Find T_min (at RH=100) ---
    # Good initial guess: T is slightly above Tw
    t_min_guess = target_tw + 0.1
    T_min = fsolve(equation_to_solve, [t_min_guess], args=(100.0, 'T', target_tw))[0] 

    # --- Find T_max (at RH=0) ---
    # Good initial guess: T is significantly above Tw
    t_max_guess = target_tw + 5
    T_max = fsolve(equation_to_solve, [t_max_guess], args=(0.0, 'T', target_tw))[0]
    
    print(f"  T_min (RH=100%): {T_min:.2f}°C")
    print(f"  T_max (RH=0%):   {T_max:.2f}°C")

    # Update overall plot limits
    overall_t_min = min(overall_t_min, T_min)
    overall_t_max = max(overall_t_max, T_max)

    # --- Generate the curve data ---
    temperatures = np.linspace(T_min, T_max, 100)
    humidities = []
    initial_guess_rh = [100.0] 

    for T in temperatures:
        # Solve for RH, passing T and target_tw
        rh_solution = fsolve(equation_to_solve, initial_guess_rh, args=(T, 'RH', target_tw))[0]
        
        if 0 <= rh_solution <= 100:
            humidities.append(rh_solution)
            initial_guess_rh = [rh_solution]
        else:
            # Clip at the boundaries
            humidities.append(max(0, min(100, rh_solution)))

    valid_temperatures = temperatures[:len(humidities)]
    
    # --- Plot the current curve ---
    plt.plot(valid_temperatures, humidities, label=f'{target_tw}°C', linewidth=2)

# 6. Finalize Plot
plt.title('Constant Wet Bulb Temperature Curves', fontsize=16)
plt.xlabel('Air Temperature (°C)', fontsize=12)
plt.ylabel('Relative Humidity (%)', fontsize=12)

# Set axis limits based on the min/max values found
#plt.xlim(math.floor(overall_t_min), math.ceil(overall_t_max))
#plt.ylim(0, 100)
#plt.xlim(math.floor(overall_t_min), 20)
plt.xlim(10, 20)
plt.ylim(10, 60)

highlight_xmin = 15
highlight_xmax = 17
highlight_ymin_norm = 35 / 70.0  # Normalized y-start
highlight_ymax_norm = 55 / 70.0  # Normalized y-end

plt.axvspan(highlight_xmin, highlight_xmax,
            ymin=highlight_ymin_norm, ymax=highlight_ymax_norm,
            color='red', alpha=0.3, label='Target Range')

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Target WBT') # Add a title to the legend

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(5.0))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))

# Save the plot
plot_filename = 'wet_bulb_curves_multiple.png'
plt.savefig(plot_filename)

print(f"\nPlot successfully generated and saved to '{plot_filename}'")
