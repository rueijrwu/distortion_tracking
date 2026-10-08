
#%% 
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navarro_GRIN import calculate_grin_parameters
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Calibri",
    "mathtext.fontset": "cm",
    "svg.fonttype": "none"})

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'tk')

import win32com.client
from win32com.client import constants as const
import pythoncom

import numpy as np
lens_folder = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 
    "..", "lens")
)


# CodeV COM interface helper functions
def cv_eval(cv, cmd, *args):
    joined_args = " ".join(args)
    return float(cv.EvaluateExpression(f"({cmd} {joined_args})"))


def grin_glass_command(A, age=20, wavelength=850.0):
    params = calculate_grin_parameters(age, A, wavelength)
    n_o = params["n_o"]
    
    return (
        f"prv;\n"
        f"pwl {wavelength:.2f};\n"
        f"'NAV_{age}' {n_o:.8f};\n"
        f"udg 0.1;\n"
        f"udg c1 {A:.6f};\n"
        f"udg c2 {wavelength:.6f};\n"
        f"end"
    )


def trace_chief_ray(
        cv, 
        zoom,
        field,
        position="y", 
        aperture_surfaces=None, 
        semi_aperture_size=None, 
        default=np.nan,
        check_error=True):
    """Trace ray if surface stops are within aperture size (if any), else return NaN."""
    ret = cv.Command(f"rsi r1 {field} {zoom} si")
    if 'Error' in ret and check_error:
        return default
    if aperture_surfaces and semi_aperture_size:
        for surface, limit in zip(aperture_surfaces, semi_aperture_size):
            pos = cv_eval(cv, f"{position} r1 {field}", zoom, surface)
            if np.abs(pos) > limit:
                return default
    return cv_eval(cv, f"{position} r1 {field} {zoom} si")


#%%
# Initialize COM and start CodeV
pythoncom.CoUninitialize()
pythoncom.CoInitialize()

cv = win32com.client.Dispatch("CodeV.Application")
cv.StartingDirectory = lens_folder
cv.StartCodeV()
cv.Command(f"pth len app {lens_folder}\\")
cv.Command(f"pth seq app {lens_folder}\\")

#%%
# Set up parameters
filename = "p14_sigma_navarro_perfect_src.len"
AGE = 20
WAVELENGTH = 850.0
EYE_ROTATION_MAX = 20
DP_MAX = 5
Z_MAX = -1
Z_MIN = 1
N_EYE_ROTATION = 21
N_DP = 5
N_Z = 3
EYE_ROTATION = np.linspace(
    -EYE_ROTATION_MAX, EYE_ROTATION_MAX, N_EYE_ROTATION)
DP = np.linspace(0, DP_MAX, N_DP)
DZ = np.linspace(Z_MIN, Z_MAX, N_Z)

data_p11 = np.zeros([N_Z, 1, N_EYE_ROTATION], dtype=np.double)
data_p12 = np.zeros_like(data_p11)
data_p10 = np.zeros_like(data_p11)

data_p41 = np.zeros([N_Z, N_DP, N_EYE_ROTATION], dtype=np.double)
data_p42 = np.zeros_like(data_p41)
data_p40 = np.zeros_like(data_p41)


# Load the sequence file and get lens data
cv.Command(f"res {filename}")

S_RC = "s\"RC\""
S_CORNEA_F = "s\"CorneaF\""
S_CORNEA_B = "s\"CorneaB\""
S_LENS_F = "s\"LensF\""
S_LENS_F_D = "s\"LensF_D\""
S_LENS_B = "s\"LensB\""
S_CAM_STOP = "s\"CAM_STP\""
S_IMG = "si"

P4_ZOOM = "z1"
P1_ZOOM = "z2"
SRC0_FIELD = "f1"
SRC1_FIELD = "f2"
SRC2_FIELD = "f3"

for zIdx, zz in enumerate(DZ):
    print(f"Tracing rays for Z={zz:.2f} mm ({zIdx+1}/{N_Z})...")
    cmd_zz = f"thi {S_IMG} {zz};"
    cv.Command(cmd_zz)
    for dIdx, A in enumerate(DP):
        print(f"Accommodation={A:.2f} D ({dIdx+1}/{N_DP})...")
        # Navarro 2014 equations from navarro_GRIN.py
        params = calculate_grin_parameters(AGE, A, WAVELENGTH)
        aquesous_thickness = params["t_aq"]
        cornea_thickness = params["t_co"]
        
        corneaF_radius = params["cornea_ant"]["Rcy"]
        corneaF_conic = params["cornea_ant"]["Qy"]
        corneaB_radius = params["cornea_pos"]["Rcy"]
        corneaB_conic = params["cornea_pos"]["Qy"]
        
        lensF_radius = params["Ra"]
        lensB_radius = params["Rp"]
        lensF_thickness = params["t_le"]
        lensF_conic = params["q_a"]
        lensB_conic = params["q_p"]

        cmd_lens_data = (
            f"thi {S_RC} {P4_ZOOM} {cornea_thickness};"
            f"rdy {S_RC} {P4_ZOOM} {corneaF_radius};"
            f"k {S_RC} {P4_ZOOM} {corneaF_conic};"
            f"thi {S_CORNEA_B} {P4_ZOOM} {aquesous_thickness};"
            f"rdy {S_CORNEA_B} {P4_ZOOM} {corneaB_radius};"
            f"k {S_CORNEA_B} {P4_ZOOM} {corneaB_conic};"
            f"thi {S_LENS_F} {P4_ZOOM} {lensF_thickness};"
            f"rdy {S_LENS_F} {P4_ZOOM} {lensF_radius};"
            f"rdy {S_LENS_B} {P4_ZOOM} {lensB_radius};"
            f"k {S_LENS_F} {P4_ZOOM} {lensF_conic};"
            f"k {S_LENS_B} {P4_ZOOM} {lensB_conic};"
        )
        cv.Command(cmd_lens_data)

        # Update GRIN glass lens
        cmd_glass = grin_glass_command(
            A, age=AGE, wavelength=WAVELENGTH)
        cv.Command(cmd_glass)

        # eye rotation and tracing chief ray
        for rxIdx, rotX in enumerate(EYE_ROTATION):
            # print(f"Eye Rotation={rotX:.2f} deg ({rxIdx+1}/{N_EYE_ROTATION})...")
            cv.Command(f"ade {S_RC} {rotX};set vig")

            # P1. no accommodation
            if dIdx == 0:
                data_p10[zIdx, dIdx, rxIdx] = trace_chief_ray(
                    cv, SRC0_FIELD, P1_ZOOM)
                data_p11[zIdx, dIdx, rxIdx] = trace_chief_ray(
                    cv, SRC1_FIELD, P1_ZOOM)
                data_p12[zIdx, dIdx, rxIdx] = trace_chief_ray(
                    cv, SRC2_FIELD, P1_ZOOM)
            # P4
            data_p40[zIdx, dIdx, rxIdx] = trace_chief_ray(
                cv, SRC0_FIELD, P4_ZOOM)
            data_p41[zIdx, dIdx, rxIdx] = trace_chief_ray(
                cv, SRC1_FIELD, P4_ZOOM)
            data_p42[zIdx, dIdx, rxIdx] = trace_chief_ray(
                cv, SRC2_FIELD, P4_ZOOM)
#%%
# cv.StopCodeV()
# pythoncom.CoUninitialize()

#%%
output_folder = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
output_file = "p14_sigma_navarro_perfect_src_z.npz"
np.savez_compressed(
    os.path.join(output_folder, output_file),
    EYE_ROTATION=EYE_ROTATION,
    DP=DP,
    DZ=DZ,
    data_p11=data_p11,
    data_p12=data_p12,
    data_p10=data_p10, 
    data_p41=data_p41,
    data_p42=data_p42,
    data_p40=data_p40)

#%%
fontsize = 26
legend_fontsize = 10
ZIDX = 1
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
ax1.plot(EYE_ROTATION, -data_p11[ZIDX, 0, :], label="P11")
ax1.plot(EYE_ROTATION, -data_p12[ZIDX, 0, :], label="P12")
ax1.plot(EYE_ROTATION, -data_p10[ZIDX, 0, :], label="P10")
for aIdx, dp in enumerate(DP):
    #dIdx = int(dIdx * 10)
    line = ax1.plot(EYE_ROTATION, -data_p41[ZIDX, aIdx, :], label=f"DP={dp:.2f}")
    color = line[0].get_color()
    ax1.plot(EYE_ROTATION, -data_p42[ZIDX, aIdx, :], label=f"DP={dp:.2f}", color=color)
    ax1.plot(EYE_ROTATION, -data_p40[ZIDX, aIdx, :], label=f"DP={dp:.2f}", color=color)
ax1.set_xlabel("Eye Rotation (deg)", fontsize=fontsize)
ax1.set_ylabel("Position (mm)", fontsize=fontsize)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.legend(fontsize=legend_fontsize)
ax1.grid()
plt.show()

# %%
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
for aIdx, dp in enumerate(DP):
    d44 = np.abs(data_p42[ZIDX, aIdx, :] - data_p41[ZIDX, aIdx, :])
    line = ax1.plot(EYE_ROTATION, d44, label=f"DP={dp:.2f}")
ax1.set_xlabel("Eye Rotation (deg)", fontsize=fontsize)
ax1.set_ylabel("Position (mm)", fontsize=fontsize)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.legend(fontsize=legend_fontsize)
ax1.grid()
plt.show()
