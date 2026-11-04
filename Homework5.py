"""
MAE 3440 - Heat Transfer: Homework 5, Problem 2
Finite Difference Method Solver for a Rectangular Rib (Longitudinal Fin)

This code solves for the 2D temperature distribution in a rectangular rib
using the Finite Difference Method (FDM) with an energy balance approach.

The rib has:
  - Base at y=0 held at T_b (Dirichlet BC)
  - Left and right surfaces exposed to convection (h, T_inf)
  - Top surface (tip) exposed to convection (h, T_inf)
  - Symmetry is exploited: only the LEFT HALF of the rib is solved

Node equations come from Table 4.2 in the FDM lecture notes:
  - Case 1 (Eq. 4.29): Interior node
  - Case 3 (Eq. 4.42a): Plane surface with convection
  - Case 3 with h=0:    Plane surface, adiabatic (symmetry line)
  - Case 4 (Eq. 4.43):  Exterior corner with convection

The 1D fin comparison uses Table 3.4 from the Steady 1D Conduction notes:
  - Case A (Eq. 3.72): Convective tip fin heat transfer rate
  - Case B (Eq. 3.76): Insulated (adiabatic) tip fin heat transfer rate
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# GIVEN PARAMETERS (from Problem Statement)
# =============================================================================
k = 10.0        # Thermal conductivity [W/m-K]
L = 0.008       # Rib length in y-direction [m] (8 mm)
w_default = 0.004  # Rib width in x-direction [m] (4 mm)
T_b = 45.0      # Base temperature [°C]
T_inf = 25.0    # Ambient air temperature [°C]
h = 600.0       # Convection coefficient [W/m²-K]
theta_b = T_b - T_inf  # Excess temperature at base [°C]


# =============================================================================
# FDM SOLVER FUNCTION (Exploiting Symmetry - Solves Left Half Only)
# =============================================================================
def solve_fdm(L, w, dx, dy, k, h, T_b, T_inf):
    """
    Solve 2D steady conduction in a rectangular rib using FDM.
    
    Uses symmetry about the vertical centerline (x = w/2) to solve 
    only the left half of the domain. The symmetry line acts as an
    adiabatic boundary (dT/dx = 0).
    
    Grid indexing: m (x-direction, columns), n (y-direction, rows)
      - m = 0 is the left surface (convection)
      - m = M is the symmetry line (adiabatic)
      - n = 0 is the base (T = T_b, known)
      - n = N is the tip (convection on top)
    
    Parameters
    ----------
    L  : float - Rib height [m]
    w  : float - Full rib width [m]
    dx : float - Grid spacing in x [m]
    dy : float - Grid spacing in y [m]
    k  : float - Thermal conductivity [W/m-K]
    h  : float - Convection coefficient [W/m²-K]
    T_b  : float - Base temperature [°C]
    T_inf: float - Ambient temperature [°C]
    
    Returns
    -------
    T_full : 2D array - Temperature field for FULL rib [°C]
    q_base : float    - Total heat transfer rate at the base per unit depth [W/m]
    x_full : 1D array - x-coordinates for full rib [m]
    y      : 1D array - y-coordinates [m]
    """
    
    # Number of nodes in half-domain
    half_w = w / 2.0
    Mx = int(round(half_w / dx))  # Number of x-intervals in half-domain
    Ny = int(round(L / dy))       # Number of y-intervals
    
    # Dimensionless Biot-like parameters
    Bi_x = h * dx / k   # hΔx/k - appears in convection boundary equations
    Bi_y = h * dy / k   # hΔy/k
    
    # Total number of UNKNOWN nodes (exclude base row n=0 where T=T_b)
    # Nodes: m = 0..Mx, n = 1..Ny
    num_x = Mx + 1   # columns in half-domain
    num_y = Ny        # rows of unknowns (n=1 to n=Ny)
    N_unknowns = num_x * num_y
    
    # Helper: map (m, n) to equation index (n starts at 1 for unknowns)
    def idx(m, n):
        return (n - 1) * num_x + m
    
    # Build coefficient matrix [A] and RHS vector [C]
    A = np.zeros((N_unknowns, N_unknowns))
    C = np.zeros(N_unknowns)
    
    r = (dx / dy) ** 2  # ratio for non-square grids: (Δx/Δy)²
    
    for n in range(1, Ny + 1):      # Row index (y-direction), 1 to Ny
        for m in range(0, Mx + 1):  # Column index (x-direction), 0 to Mx
            eq = idx(m, n)  # Equation number for this node
            
            # Identify boundary type for this node
            is_left   = (m == 0)       # Left surface (convection)
            is_right  = (m == Mx)      # Right boundary (symmetry = adiabatic)
            is_bottom = (n == 1)       # Adjacent to base (n-1 = 0 is known T_b)
            is_top    = (n == Ny)      # Top surface (convection)
            
            # Determine node type and write energy balance equation
            # Convention: all equations written as sum = 0
            # Each term: conduction from neighbor + convection terms = 0
            
            # ---- Count exposed convection surfaces ----
            conv_left  = is_left
            conv_top   = is_top
            conv_right = False  # Right is always adiabatic (symmetry)
            
            # ---- Build equation using energy balance ----
            # Conduction areas depend on whether node is on a boundary
            # (half control volume on boundary sides)
            
            # x-direction widths for control volume
            if is_left and is_right:
                # Single column in half-domain (Mx=0, very narrow rib)
                wx_left = 0      # no solid to the left
                wx_right = 0     # no solid to the right
                wx_cv = dx / 2   # control volume half-width
            elif is_left:
                wx_left = 0
                wx_right = dx
                wx_cv = dx / 2
            elif is_right:
                wx_left = dx
                wx_right = 0
                wx_cv = dx / 2
            else:
                wx_left = dx
                wx_right = dx
                wx_cv = dx
            
            # y-direction heights for control volume
            if is_top and is_bottom:
                wy_below = dy
                wy_above = 0
                wy_cv = dy / 2 + dy / 2  # shouldn't happen for L > dy
            elif is_bottom:
                wy_below = dy
                wy_above = dy
                wy_cv = dy
            elif is_top:
                wy_below = dy
                wy_above = 0
                wy_cv = dy / 2
            else:
                wy_below = dy
                wy_above = dy
                wy_cv = dy
                
            # ---- Conduction from neighbors (per unit depth) ----
            # q = k * (area/length) * (T_neighbor - T_node)
            # Area is the face area (height or width * 1 for unit depth)
            # Length is dx or dy
            
            coeff_self = 0.0
            
            # Left neighbor (m-1, n) - conduction in x
            if not is_left:
                face_y = wy_cv  # face height
                cond_coeff = k * face_y / dx
                A[eq, idx(m-1, n)] += cond_coeff
                coeff_self -= cond_coeff
            
            # Right neighbor (m+1, n) - conduction in x
            if not is_right:
                face_y = wy_cv
                cond_coeff = k * face_y / dx
                A[eq, idx(m+1, n)] += cond_coeff
                coeff_self -= cond_coeff
            elif is_right and not (is_left and is_right):
                # Symmetry: mirror image means T(m+1) = T(m-1)
                # which is equivalent to adiabatic - no flux, no term needed
                pass
            
            # Below neighbor (m, n-1) - conduction in y
            face_x_below = wx_cv if not (is_left and is_right) else dx / 2
            if is_left and not is_right:
                face_x_below = dx / 2
            elif is_right and not is_left:
                face_x_below = dx / 2
            elif not is_left and not is_right:
                face_x_below = dx
            else:
                face_x_below = dx / 2
                
            cond_coeff_below = k * face_x_below / dy
            if is_bottom:
                # n-1 = 0, T is known = T_b
                C[eq] -= cond_coeff_below * T_b
                coeff_self -= cond_coeff_below
            else:
                A[eq, idx(m, n-1)] += cond_coeff_below
                coeff_self -= cond_coeff_below
            
            # Above neighbor (m, n+1) - conduction in y
            if not is_top:
                face_x_above = face_x_below  # same width
                cond_coeff_above = k * face_x_above / dy
                A[eq, idx(m, n+1)] += cond_coeff_above
                coeff_self -= cond_coeff_above
            
            # ---- Convection terms ----
            # q_conv = h * A_conv * (T_inf - T_node)
            
            # Convection on left surface
            if conv_left:
                conv_area = wy_cv  # height of exposed face * unit depth
                conv_coeff = h * conv_area
                C[eq] -= conv_coeff * T_inf
                coeff_self -= conv_coeff
            
            # Convection on top surface
            if conv_top:
                if is_left and is_right:
                    conv_area_top = dx / 2
                elif is_left or is_right:
                    conv_area_top = dx / 2
                else:
                    conv_area_top = dx
                conv_coeff_top = h * conv_area_top
                C[eq] -= conv_coeff_top * T_inf
                coeff_self -= conv_coeff_top
            
            # Self coefficient
            A[eq, eq] += coeff_self
    
    # ---- Solve the system [A][T] = [C] ----
    T_unknown = np.linalg.solve(A, C)
    
    # ---- Reconstruct the full temperature field ----
    # Half-domain array (including base row)
    T_half = np.zeros((Ny + 1, Mx + 1))
    T_half[0, :] = T_b  # Base row
    for n in range(1, Ny + 1):
        for m in range(0, Mx + 1):
            T_half[n, m] = T_unknown[idx(m, n)]
    
    # Mirror to get full domain
    # Full domain: m = 0 to 2*Mx (columns)
    T_full = np.zeros((Ny + 1, 2 * Mx + 1))
    T_full[:, :Mx + 1] = T_half          # Left half
    T_full[:, Mx:] = T_half[:, ::-1]     # Right half (mirrored)
    
    # Coordinates
    x_half = np.linspace(0, half_w, Mx + 1)
    x_full = np.linspace(0, w, 2 * Mx + 1)
    y = np.linspace(0, L, Ny + 1)
    
    # ---- Calculate heat transfer rate at the base ----
    # q_base = sum of conduction from T_b into n=1 nodes, for the half domain, times 2
    q_base_half = 0.0
    for m in range(0, Mx + 1):
        if m == 0 or m == Mx:
            face_width = dx / 2
        else:
            face_width = dx
        q_base_half += k * face_width * (T_b - T_half[1, m]) / dy
    
    q_base = 2.0 * q_base_half  # Full rib by symmetry
    
    return T_full, q_base, x_full, y


# =============================================================================
# 1D FIN SOLUTION FUNCTION (from Table 3.4, Slide 58)
# =============================================================================
def fin_1d(L, w, k, h, T_b, T_inf, tip='convective'):
    """
    1D fin solution for a rectangular longitudinal rib.
    
    Per unit depth into the page:
      - Cross-section area: A_c = w * 1
      - Perimeter: P = 2 * 1 (both sides exposed)
      - m² = hP / (kA_c)
      - M = sqrt(hPkA_c) * theta_b
    
    Parameters
    ----------
    tip : str - 'convective' (Case A, Eq. 3.72) or 'insulated' (Case B, Eq. 3.76)
    
    Returns
    -------
    q_fin : float - Fin heat transfer rate per unit depth [W/m]
    T_fin : callable - Temperature distribution T(y)
    """
    theta_b = T_b - T_inf
    P = 2.0          # Perimeter per unit depth [m]
    A_c = w * 1.0    # Cross-section per unit depth [m²]
    
    m = np.sqrt(h * P / (k * A_c))
    M = np.sqrt(h * P * k * A_c) * theta_b
    
    if tip == 'insulated':
        # Case B - Adiabatic tip (Eq. 3.76)
        q_fin = M * np.tanh(m * L)
        def T_dist(y):
            return T_inf + theta_b * np.cosh(m * (L - y)) / np.cosh(m * L)
    
    elif tip == 'convective':
        # Case A - Convective tip (Eq. 3.72)
        hmk = h / (m * k)
        q_fin = M * (np.sinh(m * L) + hmk * np.cosh(m * L)) / \
                    (np.cosh(m * L) + hmk * np.sinh(m * L))
        def T_dist(y):
            num = np.cosh(m * (L - y)) + hmk * np.sinh(m * (L - y))
            den = np.cosh(m * L) + hmk * np.sinh(m * L)
            return T_inf + theta_b * num / den
    
    return q_fin, T_dist, m


# =============================================================================
# PART 1: GRID REFINEMENT STUDY (Δx = Δy = 2mm, 1mm, 0.5mm)
# =============================================================================
print("=" * 70)
print("PART 1: GRID REFINEMENT STUDY")
print("=" * 70)
print(f"\nGiven: k={k} W/m-K, L={L*1000}mm, w={w_default*1000}mm")
print(f"       T_b={T_b}°C, T_inf={T_inf}°C, h={h} W/m²-K\n")

grid_sizes = [0.002, 0.001, 0.0005]  # [m]
grid_labels = ['2 mm', '1 mm', '0.5 mm']
results_grid = {}

# Also compute 1D fin solutions for comparison
q_insulated, T_insulated, m_fin = fin_1d(L, w_default, k, h, T_b, T_inf, 'insulated')
q_convective, T_convective, _ = fin_1d(L, w_default, k, h, T_b, T_inf, 'convective')

print(f"1D Fin Solutions (for comparison):")
print(f"  m = {m_fin:.4f} 1/m,  mL = {m_fin*L:.4f}")
print(f"  Insulated tip:  q_f = {q_insulated:.4f} W/m")
print(f"  Convective tip: q_f = {q_convective:.4f} W/m\n")

# Set up figure for temperature contour plots
fig_contour, axes_contour = plt.subplots(1, 3, figsize=(18, 7))
fig_contour.suptitle('Temperature Distribution in Rib Cross-Section\n(Grid Refinement Study)', 
                      fontsize=14, fontweight='bold')

for i, (ds, label) in enumerate(zip(grid_sizes, grid_labels)):
    T_full, q_base, x, y = solve_fdm(L, w_default, ds, ds, k, h, T_b, T_inf)
    results_grid[label] = {'T': T_full, 'q': q_base, 'x': x, 'y': y, 'ds': ds}
    
    print(f"Grid: Δx = Δy = {label}")
    print(f"  Grid size: {len(x)} x {len(y)} nodes (full domain)")
    print(f"  q_base (FDM)       = {q_base:.4f} W/m")
    print(f"  q_base (conv. tip) = {q_convective:.4f} W/m")
    print(f"  Difference from convective tip: {(q_base - q_convective)/q_convective*100:.2f}%")
    
    # Print centerline temperatures at key locations
    mid_col = len(x) // 2  # centerline column
    print(f"  Centerline temperatures:")
    for y_val in [0.002, 0.004, 0.006, 0.008]:
        row = int(round(y_val / ds))
        if row < len(y):
            print(f"    y={y_val*1000:.0f}mm: T = {T_full[row, mid_col]:.4f}°C")
    print()
    
    # Contour plot
    ax = axes_contour[i]
    X, Y = np.meshgrid(x * 1000, y * 1000)  # Convert to mm
    levels = np.linspace(T_inf, T_b, 25)
    cf = ax.contourf(X, Y, T_full, levels=levels, cmap='hot_r')
    cs = ax.contour(X, Y, T_full, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')
    ax.set_title(f'Δx = Δy = {label}\nq = {q_base:.2f} W/m', fontsize=11)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_aspect('equal')

fig_contour.subplots_adjust(right=0.88, top=0.85)
cbar_ax = fig_contour.add_axes([0.90, 0.15, 0.02, 0.65])
fig_contour.colorbar(cf, cax=cbar_ax, label='Temperature [°C]')
fig_contour.savefig('grid_refinement_contours.png', dpi=150, bbox_inches='tight')
print("Contour plots saved.\n")


# ---- Centerline Temperature Comparison Plot ----
fig_centerline, ax_cl = plt.subplots(figsize=(8, 6))
y_fine = np.linspace(0, L, 200)

# Plot 1D fin solutions
ax_cl.plot(y_fine * 1000, T_insulated(y_fine), 'b--', linewidth=2, label='1D Fin: Insulated Tip (Eq. 3.76)')
ax_cl.plot(y_fine * 1000, T_convective(y_fine), 'r--', linewidth=2, label='1D Fin: Convective Tip (Eq. 3.72)')

# Plot FDM centerline results
markers = ['s', 'o', '^']
colors = ['green', 'purple', 'orange']
for i, (label, data) in enumerate(results_grid.items()):
    mid_col = len(data['x']) // 2
    centerline_T = data['T'][:, mid_col]
    ax_cl.plot(data['y'] * 1000, centerline_T, marker=markers[i], color=colors[i],
               markersize=5, linewidth=1.2, label=f'FDM Δ={label} (centerline)')

ax_cl.set_xlabel('y [mm] (distance from base)', fontsize=12)
ax_cl.set_ylabel('Temperature [°C]', fontsize=12)
ax_cl.set_title('Centerline Temperature Distribution:\nFDM vs 1D Fin Solutions', fontsize=13, fontweight='bold')
ax_cl.legend(fontsize=9, loc='lower left')
ax_cl.grid(True, alpha=0.3)
ax_cl.set_xlim([0, 8])
ax_cl.set_ylim([T_inf - 1, T_b + 1])
fig_centerline.tight_layout()
fig_centerline.savefig('centerline_comparison.png', dpi=150, bbox_inches='tight')
print("Centerline comparison plot saved.\n")


# ---- Grid Convergence Summary Table ----
print("-" * 60)
print("GRID CONVERGENCE SUMMARY")
print("-" * 60)
print(f"{'Grid':>10s} {'Nodes':>10s} {'q_FDM [W/m]':>14s} {'q_conv [W/m]':>14s} {'% Diff':>10s}")
print("-" * 60)
for label, data in results_grid.items():
    nx, ny = data['T'].shape[1], data['T'].shape[0]
    pct = (data['q'] - q_convective) / q_convective * 100
    print(f"{label:>10s} {f'{nx}x{ny}':>10s} {data['q']:>14.4f} {q_convective:>14.4f} {pct:>9.2f}%")
print("-" * 60)


# =============================================================================
# VERIFICATION: Check against Problem 1 hand calculations (Δx = Δy = 2mm)
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION: Compare to Problem 1 Hand Calculations (Δx=Δy=2mm)")
print("=" * 70)

data_2mm = results_grid['2 mm']
T_2mm = data_2mm['T']
x_2mm = data_2mm['x']
y_2mm = data_2mm['y']

# Hand-calculated temperatures from Problem 1
T_hand = {
    'T1 (0,2mm)': 39.3231, 'T2 (2mm,2mm)': 40.0007,
    'T3 (0,4mm)': 35.7286, 'T4 (2mm,4mm)': 36.3567,
    'T5 (0,6mm)': 33.4526, 'T6 (2mm,6mm)': 33.9690,
    'T7 (0,8mm)': 32.1726, 'T8 (2mm,8mm)': 32.6139,
}

# Extract corresponding FDM values
# m=0 is x=0 (col 0), m=1 is x=2mm (col 1 in half, which is col 1 in full)
# But in the full domain: col 0 = x=0, col 1 = x=2mm, col 2 = x=4mm
T_code = {
    'T1 (0,2mm)': T_2mm[1, 0], 'T2 (2mm,2mm)': T_2mm[1, 1],
    'T3 (0,4mm)': T_2mm[2, 0], 'T4 (2mm,4mm)': T_2mm[2, 1],
    'T5 (0,6mm)': T_2mm[3, 0], 'T6 (2mm,6mm)': T_2mm[3, 1],
    'T7 (0,8mm)': T_2mm[4, 0], 'T8 (2mm,8mm)': T_2mm[4, 1],
}

print(f"\n{'Node':<18s} {'Hand Calc [°C]':>15s} {'Code [°C]':>12s} {'Diff [°C]':>12s}")
print("-" * 60)
all_match = True
for key in T_hand:
    diff = abs(T_hand[key] - T_code[key])
    match = "✓" if diff < 0.01 else "✗"
    if diff >= 0.01:
        all_match = False
    print(f"{key:<18s} {T_hand[key]:>15.4f} {T_code[key]:>12.4f} {diff:>11.4f}  {match}")

q_hand = 106.7613
q_code = data_2mm['q']
print(f"\n{'q_base [W/m]':<18s} {q_hand:>15.4f} {q_code:>12.4f} {abs(q_hand-q_code):>11.4f}")
print(f"\nVerification: {'PASSED - All values match hand calculations' if all_match else 'DISCREPANCY FOUND'}")


# =============================================================================
# PART 2: ASPECT RATIO STUDY (L/w = 1 to 8, L = 8mm, Δx = Δy = 1mm)
# =============================================================================
print("\n\n" + "=" * 70)
print("PART 2: ASPECT RATIO STUDY (L/w = 1 to 8)")
print("=" * 70)
print(f"Fixed: L = {L*1000} mm, Δx = Δy = 0.5 mm")
print(f"Varying w from {L*1000} mm (L/w=1) to {L/8*1000} mm (L/w=8)\n")

ds_ar = 0.0005  # Use 0.5 mm grid for all aspect ratios (divides all widths evenly)
aspect_ratios = [1, 2, 3, 4, 5, 6, 7, 8]
ar_results = []

print(f"{'L/w':>5s} {'w [mm]':>8s} {'dx [mm]':>8s} {'q_FDM':>12s} {'q_conv_tip':>12s} {'q_ins_tip':>12s} {'%Diff(conv)':>12s}")
print("-" * 75)

for Lw in aspect_ratios:
    w_i = L / Lw
    
    # Use 0.5mm grid for all, which divides evenly for all cases
    half_w = w_i / 2
    ds_use = ds_ar
    n_half = half_w / ds_use
    if abs(n_half - round(n_half)) > 1e-10:
        # Use a grid that divides evenly
        n_cells = max(1, int(round(half_w / ds_use)))
        ds_use = half_w / n_cells
    
    # FDM solution
    T_full, q_fdm, x_ar, y_ar = solve_fdm(L, w_i, ds_use, ds_use, k, h, T_b, T_inf)
    
    # 1D fin solutions
    q_conv_tip, _, _ = fin_1d(L, w_i, k, h, T_b, T_inf, 'convective')
    q_ins_tip, _, _ = fin_1d(L, w_i, k, h, T_b, T_inf, 'insulated')
    
    pct_diff = (q_fdm - q_conv_tip) / q_conv_tip * 100
    
    ar_results.append({
        'Lw': Lw, 'w': w_i, 'q_fdm': q_fdm, 
        'q_conv': q_conv_tip, 'q_ins': q_ins_tip, 'pct': pct_diff,
        'ds': ds_use
    })
    
    print(f"{Lw:>5d} {w_i*1000:>8.2f} {ds_use*1000:>8.2f} {q_fdm:>12.4f} {q_conv_tip:>12.4f} {q_ins_tip:>12.4f} {pct_diff:>11.2f}%")

print("-" * 75)


# ---- Aspect Ratio Plot ----
fig_ar, ax_ar = plt.subplots(figsize=(9, 6))

Lw_vals = [r['Lw'] for r in ar_results]
q_fdm_vals = [r['q_fdm'] for r in ar_results]
q_conv_vals = [r['q_conv'] for r in ar_results]
q_ins_vals = [r['q_ins'] for r in ar_results]

ax_ar.plot(Lw_vals, q_fdm_vals, 'ko-', linewidth=2, markersize=8, label='FDM (2D)', zorder=3)
ax_ar.plot(Lw_vals, q_conv_vals, 'rs--', linewidth=2, markersize=7, label='1D Fin: Convective Tip (Eq. 3.72)')
ax_ar.plot(Lw_vals, q_ins_vals, 'b^--', linewidth=2, markersize=7, label='1D Fin: Insulated Tip (Eq. 3.76)')

ax_ar.set_xlabel('Aspect Ratio L/w', fontsize=12)
ax_ar.set_ylabel('Fin Heat Transfer Rate q_f [W/m]', fontsize=12)
ax_ar.set_title('Effect of Rib Aspect Ratio on Heat Transfer Rate\n(L = 8 mm, Δx = Δy = 0.5 mm)', 
                fontsize=13, fontweight='bold')
ax_ar.legend(fontsize=10)
ax_ar.grid(True, alpha=0.3)
ax_ar.set_xticks(Lw_vals)
fig_ar.tight_layout()
fig_ar.savefig('aspect_ratio_study.png', dpi=150, bbox_inches='tight')
print("\nAspect ratio plot saved.\n")


# ---- Percentage difference plot ----
fig_pct, ax_pct = plt.subplots(figsize=(9, 5))
pct_conv = [(r['q_fdm'] - r['q_conv']) / r['q_conv'] * 100 for r in ar_results]
pct_ins = [(r['q_fdm'] - r['q_ins']) / r['q_ins'] * 100 for r in ar_results]

ax_pct.bar([x - 0.15 for x in Lw_vals], pct_conv, width=0.3, color='red', alpha=0.7, 
           label='vs Convective Tip')
ax_pct.bar([x + 0.15 for x in Lw_vals], pct_ins, width=0.3, color='blue', alpha=0.7, 
           label='vs Insulated Tip')
ax_pct.axhline(y=0, color='black', linewidth=0.5)
ax_pct.set_xlabel('Aspect Ratio L/w', fontsize=12)
ax_pct.set_ylabel('% Difference (FDM vs 1D Fin)', fontsize=12)
ax_pct.set_title('FDM vs 1D Fin: Percentage Difference by Aspect Ratio', fontsize=13, fontweight='bold')
ax_pct.legend(fontsize=10)
ax_pct.grid(True, alpha=0.3, axis='y')
ax_pct.set_xticks(Lw_vals)
fig_pct.tight_layout()
fig_pct.savefig('aspect_ratio_pct_diff.png', dpi=150, bbox_inches='tight')
print("Percentage difference plot saved.\n")


# ---- Contour plots for select aspect ratios ----
fig_ar_contours, axes_arc = plt.subplots(2, 4, figsize=(20, 10))
fig_ar_contours.suptitle('Temperature Distribution for Various Aspect Ratios (L/w)\nL = 8 mm, Δx = Δy = 0.5 mm', 
                          fontsize=14, fontweight='bold')

for i, Lw in enumerate(aspect_ratios):
    w_i = L / Lw
    # Use same grid as the study
    ds_use = ar_results[i]['ds']
    T_full, q_fdm, x_ar, y_ar = solve_fdm(L, w_i, ds_use, ds_use, k, h, T_b, T_inf)
    
    row, col = i // 4, i % 4
    ax = axes_arc[row, col]
    
    # Only make contour if grid is at least 2x2
    if T_full.shape[0] >= 2 and T_full.shape[1] >= 2:
        X, Y = np.meshgrid(x_ar * 1000, y_ar * 1000)
        levels = np.linspace(T_inf, T_b, 20)
        cf = ax.contourf(X, Y, T_full, levels=levels, cmap='hot_r')
    else:
        ax.text(0.5, 0.5, 'Grid too\ncoarse', transform=ax.transAxes, 
                ha='center', va='center')
    ax.set_title(f'L/w = {Lw}\nw = {w_i*1000:.1f} mm, q = {q_fdm:.1f} W/m', fontsize=10)
    ax.set_xlabel('x [mm]', fontsize=8)
    ax.set_ylabel('y [mm]', fontsize=8)

fig_ar_contours.subplots_adjust(right=0.88, top=0.88, hspace=0.35, wspace=0.35)
cbar_ax2 = fig_ar_contours.add_axes([0.90, 0.15, 0.02, 0.7])
fig_ar_contours.colorbar(cf, cax=cbar_ax2, label='Temperature [°C]')
fig_ar_contours.savefig('aspect_ratio_contours.png', dpi=150, bbox_inches='tight')
print("Aspect ratio contour plots saved.\n")


# =============================================================================
# ENERGY BALANCE VERIFICATION
# =============================================================================
print("=" * 70)
print("ENERGY BALANCE VERIFICATION")
print("=" * 70)
print("\nFor each grid, verify: q_base (conduction in) = q_conv (convection out)\n")

for label, data in results_grid.items():
    T = data['T']
    ds = data['ds']
    x = data['x']
    y = data['y']
    nx, ny = T.shape[1], T.shape[0]
    
    # q_base already computed
    q_base = data['q']
    
    # Compute total convection from all exposed surfaces (full rib)
    # Left surface (x=0): all rows n=1 to Ny
    q_conv_total = 0.0
    
    # Left surface (column 0)
    for n in range(1, ny):
        if n == ny - 1:
            face_h = ds / 2  # top corner, half cell
        else:
            face_h = ds
        q_conv_total += h * face_h * (T[n, 0] - T_inf)
    
    # Right surface (last column) - same as left by symmetry
    for n in range(1, ny):
        if n == ny - 1:
            face_h = ds / 2
        else:
            face_h = ds
        q_conv_total += h * face_h * (T[n, -1] - T_inf)
    
    # Top surface (row Ny): all columns
    for m in range(0, nx):
        if m == 0 or m == nx - 1:
            face_w = ds / 2  # corner, half cell
        else:
            face_w = ds
        q_conv_total += h * face_w * (T[-1, m] - T_inf)
    
    error = abs(q_base - q_conv_total) / q_base * 100
    print(f"  Δ={label}: q_base = {q_base:.6f} W/m, q_conv = {q_conv_total:.6f} W/m, "
          f"error = {error:.6f}%  {'✓' if error < 0.1 else '✗'}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\n--- Grid Refinement (L/w = 2, w = 4mm) ---")
print(f"{'Grid':>10s} {'q_FDM':>12s} {'q_conv_tip':>12s} {'q_ins_tip':>12s}")
for label, data in results_grid.items():
    print(f"{label:>10s} {data['q']:>12.4f} {q_convective:>12.4f} {q_insulated:>12.4f}")

print(f"\nAs the grid is refined, the FDM solution converges.")
print(f"The 2D FDM gives a LOWER heat transfer rate than both 1D fin models")
print(f"because it captures the temperature variation across the rib width.")

print("\n--- Aspect Ratio Study (L = 8mm constant) ---")
print(f"As L/w increases (rib gets thinner):")
print(f"  - The 1D fin assumption becomes MORE accurate")
print(f"  - The FDM and 1D convective tip results converge")
print(f"  - This is because a thinner rib has less temperature variation across its width")
print(f"As L/w decreases (rib gets wider):")
print(f"  - 2D effects become more significant")
print(f"  - The 1D fin model increasingly OVERPREDICTS heat transfer")
print(f"  - The FDM captures the real physics better")

plt.show()
print("\nDone!")