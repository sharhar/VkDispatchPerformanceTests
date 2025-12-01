import matplotlib.pyplot as plt

# --- IEEE PUBLICATION CONFIGURATION ---
# IEEE column width is ~3.5 inches.
# If you set the figure width here, you don't have to scale it in LaTeX.
FULL_WIDTH = 7.16   # Full page width (double column)
COL_WIDTH = 3.5     # Single column width

params = {
    # Use a serif font to match the paper (Times New Roman family)
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    
    # Font sizes (8-10pt is standard for IEEE figures)
    'axes.labelsize': 9,
    'font.size': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    
    # Figure size: width = 3.5 inches, height = calculated based on golden ratio
    'figure.figsize': (COL_WIDTH, COL_WIDTH * 0.618), 
    
    # Use LaTeX for math rendering (optional, creates beautiful math)
    # Only enable this if you have a local LaTeX installation accessible to Python
    'text.usetex': False, 
}

plt.rcParams.update(params)
# --------------------------------------

# Example Plot
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='Signal', linewidth=1.5) # Thinner lines for small plots
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.legend(frameon=False) # IEEE usually prefers no box around legend
plt.grid(True, linestyle=':', alpha=0.6)

# Save
plt.savefig("fft_benchmark.pdf", bbox_inches="tight", pad_inches=0.02)