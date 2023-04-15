import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 

if os.getenv('PLOTSTYLE') is not None: 
    plt.style.use(os.getenv('PLOTSTYLE'))
RED = "#dd3015"
GREEN = "#489A8C"
DARK = "#1C2C22"
GOLD = "#F87D16"
WHITE = "#FFFFFF"
BLUE = "#2E6C96"
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[RED, GREEN, DARK, GOLD, WHITE, BLUE]) 
from helper_functions import get_numpy_arrays_from_local_data
from physics_functions import calculate_boostrap_current, find_j_max_from_boostrap_current,calculate_alpha
# conn = get_allas_connection()

local_dir = '/home/kitadam/ENR_Sven/ped_ssm/local_data_ptot_wmhd_plh'
shot_num =  35185 # 38933- no ELMs regime # 33616 37450
# 39914 - InterELM analysis 

# 35899 or 37620 or 37818
[profiles, mps, radii, times], mp_names = get_numpy_arrays_from_local_data(shot_num, local_dir)
ne, te = profiles[:, 0], profiles[:, 1]
pe = ne*te*(1.602e-19) 
colors = [(t - times[0] )/ (times[-1] - times[0]) for t in times]
colors_t = [(c, 0.5, 0.5) for c in colors]

X, Y = np.meshgrid(radii[0], times)
fig, ax = plt.subplots(2, 3, subplot_kw={"projection": "3d"}, figsize=(20, 10))

surf = ax[0, 0].plot_surface(X, Y, 1e-20*ne, cmap=mpl.cm.Spectral_r, linewidth=0)
surf = ax[0, 1].plot_surface(X, Y, (1e-4*te), cmap=mpl.cm.Spectral_r, linewidth=0)
surf = ax[0, 2].plot_surface(X, Y, pe / 1000.0, cmap=mpl.cm.Spectral_r, linewidth=0)
# for a in ax: 
#     a.set_xlabel(r'$\rho$', labelpad=10.0)
#     a.set_ylabel('time (s)', labelpad=10.0)

ax[0, 0].set_title('$n_e$ ($10^{20}$ m$^{-3}$)', y=0.98)
ax[0, 1].set_title('$T_e$ (keV)', y=0.98)
ax[0, 2].set_title('$P_e$ (kPa)', y=0.98)
# fig.suptitle(f'AUG {shot_num}: electron profiles via IDA', y=0.9)

boostrap_currents = np.zeros_like(ne)
pressure_gradients = np.zeros_like(ne)
jb_maxes, alpha_maxes =  np.zeros_like(times), np.zeros_like(times)
for n, idx in enumerate(times): 
    ahor = mps[n, mp_names.index('ahor')]
    q95 =  abs(mps[n, mp_names.index('q95')])
    rgeo =  mps[n, mp_names.index('Rgeo')]
    bt = abs(mps[n, mp_names.index('BTF')])
    vp = abs(mps[n, mp_names.index('Vol')])

    boostrap_current_approx = calculate_boostrap_current(pe[n], te[n], ne[n], radii[n], rgeo, ahor, q95, bt)
    # pressure_gradient = -np.gradient(pe[n])
    alpha_gradient = calculate_alpha(pe[n], radii[n], vp, rgeo)
    boostrap_currents[n] = boostrap_current_approx
    pressure_gradients[n] = alpha_gradient
    max_jb, rad_max_jb, index_max_jb = find_j_max_from_boostrap_current(boostrap_current_approx, radii[n])
    jb_maxes[n] = max_jb / np.mean(boostrap_current_approx)
    alpha, rho_alpha, p_alpha = find_j_max_from_boostrap_current(alpha_gradient, radii[n])
    alpha_maxes[n] = alpha

# fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(20, 10))

surf = ax[1, 0].plot_surface(X, Y, boostrap_currents, cmap=mpl.cm.Spectral_r, linewidth=0)
surf = ax[1, 1].plot_surface(X, Y, pressure_gradients, cmap=mpl.cm.Spectral_r, linewidth=0)


for a in ax.ravel(): 
    a.set_xlabel(r'$\rho$', labelpad=10.0)
    a.set_ylabel('time (s)', labelpad=10.0)

ax[1, 0].set_title('$j_B$ [MA]', y=0.98)
ax[1, 1].set_title(r'$\nabla p_e$ (kPa)', y=0.98)

ax[1, 2].scatter(alpha_maxes, times, jb_maxes,  c=colors_t)
ax[1, 2].set_xlabel('$j_B$ max')
ax[1, 2].set_zlabel(r'$\nabla p_e$ max (kPa)')
fig.suptitle(f'AUG {shot_num}: bootstrap current approximation', y=0.9)
fig.subplots_adjust(wspace=0.01)

from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(20, 10)) 

gs = GridSpec(2, 2, width_ratios=[2, 1])
jb_ax = fig.add_subplot(gs[0, 0])
alpha_ax = fig.add_subplot(gs[1, 0])
both_ax = fig.add_subplot(gs[:, 1])
hull = np.argsort(jb_maxes)
both_ax.scatter(alpha_maxes, jb_maxes, c=colors_t)
jb_ax.scatter(times, jb_maxes)
alpha_ax.scatter(times, alpha_maxes)

t1, t2 = 3.017, 3.043
time_window = np.logical_and(times > t1, times < t2)

windowed_alpha, windowed_jb, windowed_times = alpha_maxes[time_window], jb_maxes[time_window], times[time_window]
windowed_radii, windowed_pe = radii[time_window], pe[time_window]
colors = [(t - windowed_times[0] )/ (windowed_times[-1] - windowed_times[0]) for t in windowed_times]
# colors_t = [(c, 0.5, 0.5) for c in colors]
max_jb, min_jb = max(windowed_jb), min(windowed_jb)
# colors = [(t - min_jb) / (max_jb - min_jb) for t in windowed_jb]
colors_t = [(c, 0.5, 0.5) for c in colors]
fig = plt.figure(figsize=(20, 10)) 
gs = GridSpec(2, 2)
jb_ax = fig.add_subplot(gs[0, 0])
alpha_ax = fig.add_subplot(gs[1, 0])
both_ax = fig.add_subplot(gs[0, 1])
p_ax = fig.add_subplot(gs[1, 1])


both_ax.scatter(windowed_alpha, windowed_jb, c=colors_t)
jb_ax.scatter(windowed_times, windowed_jb, c=colors_t)

jb_ax.set_ylabel('$j_B$ max')
both_ax.set_ylabel('$j_B$ max')
alpha_ax.set_ylabel(r'$\nabla p_e$ max (kPa)')
both_ax.set_xlabel(r'$\nabla p_e$ max (kPa)')

alpha_ax.scatter(windowed_times, windowed_alpha, c=colors_t)
for idx in range(len(colors_t)): 
    p_ax.plot(windowed_radii[idx], windowed_pe[idx], c=colors_t[idx])

ped_window = np.logical_and(windowed_radii[0] > 0.9, windowed_radii[0]< 1.01)
ped_pressure = max(windowed_pe[0, :][ped_window])
p_ax.set_xlim(0.9, 1.01)
p_ax.set_ylim(0.0, ped_pressure + 0.1*ped_pressure)
plt.show()