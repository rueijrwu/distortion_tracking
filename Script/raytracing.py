
#%% 
import os
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
lens_folder = os.path.abspath(os.path.join(os.getcwd(), "..", "lens"))


# Eye model refrative index helper functions
def Cauchy(wl, na, nb, nc, nd):
    return na + nb / wl**2 + nc / wl**4 + nd / wl**6


WAVELENGTH = 850.0  # nm
COEFF_850 = np.array([
    [ 1.3623935136e+00, -1.8001392205e-05,  3.6735803323e-06],
    [ 6.7394986702e-02, -9.7774753376e-04, -6.4222012087e-05],
    [-3.9461873882e-02, -6.4768189371e-04,  4.6215222695e-04],
    [ 4.4424787588e-02,  2.2422863661e-03, -1.0416671225e-03],
    [-5.5515479042e-02, -8.6352875872e-04,  1.0665069364e-03],
    [ 3.5725367937e-02, -4.4653401125e-04, -5.7214066466e-04],
    [-1.2033695765e-02,  4.1750060378e-04,  1.6559364161e-04],
    [ 2.0486551193e-03, -1.1051053515e-04, -2.4340126390e-05],
    [-1.3968088637e-04,  1.0004847342e-05,  1.4125818106e-06]
])


CAUCHY_LENS_CORE    = [1.401105, 6.576875E3, -6.162814E8, 5.958617E13]
CAUCHY_LENS_EDGE    = [1.354665, 6.358883E3, -5.958546E8, 5.761117E13]
RIDX_LENS_CORE      = Cauchy(WAVELENGTH, *CAUCHY_LENS_CORE)
RIDX_LENS_EDGE      = Cauchy(WAVELENGTH, *CAUCHY_LENS_EDGE)
DELTA_RIDX_LENS     = RIDX_LENS_CORE - RIDX_LENS_EDGE


# accommodation realted parameters
AQUEOUS_THICKNESS_SLP = -0.04
LENS_THICKNESS_SLP = 0.06
LENS_F_RADIUS_SLP = -0.65
LENS_F_CONIC_SLP = -0.4
LENS_B_RADIUS_SLP = 0.17
LENS_B_CONIC_SLP = -0.25
LENS_SEMI_AP_SLP = -0.259 / 2


# CodeV COM interface helper functions
def cv_eval(cv, cmd, *args):
    joined_args = " ".join(args)
    return float(cv.EvaluateExpression(f"({cmd} {joined_args})"))


def grin_glass_command(A, lens_semi_ap):
    AA = A ** np.arange(COEFF_850.shape[1])
    URN_C = np.append(COEFF_850 @ AA, -DELTA_RIDX_LENS / (lens_semi_ap ** 2))

    coef_cmds = ";\n".join(
        f"udg c{i} {val:.6e}" for i, val in enumerate(URN_C[1:], start=1)
    )
    return (
        f"prv;\n"
        f"pwl {WAVELENGTH:.2f};\n"
        f"'Lens' {URN_C[0]:.6f};\n"
        f"udg 0.1;\n"
        f"{coef_cmds};\n"
        f"end"
    )


def trace_chief_ray(
        cv, 
        zoom,
        position="y", 
        aperture_surfaces=None, 
        semi_aperture_size=None, 
        default=np.nan,
        check_error=True):
    """Trace ray if surface stops are within aperture size (if any), else return NaN."""
    ret = cv.Command(f"rsi r1 fl {zoom} si")
    if 'Error' in ret and check_error:
        return default
    if aperture_surfaces and semi_aperture_size:
        for surface, limit in zip(aperture_surfaces, semi_aperture_size):
            pos = cv_eval(cv, f"{position} r1 fl", zoom, surface)
            if np.abs(pos) > limit:
                return default
    return cv_eval(cv, f"{position} r1 fl {zoom} si")


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
filename = "p14_sigma_atch_perfect_src.len"
EYE_ROTATION_MAX = 20
DP_MAX = 5
Z_MAX = -1
Z_MIN = 1
N_EYE_ROTATION = 101
N_DP = 51
N_Z = 21
EYE_ROTATION = np.linspace(-EYE_ROTATION_MAX, EYE_ROTATION_MAX, N_EYE_ROTATION)
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
S_CORNEA_B = "s\"CorneaB\""
S_LENS_F = "s\"LensF\""
S_LENS_F_D = "s\"LensF_D\""
S_LENS_B = "s\"LensB\""
S_CAM_STOP = "s\"CAM_STP\""
S_IMG = "si"

P41_ZOOM = "z1"
P42_ZOOM = "z2"
P40_ZOOM = "z3"
P11_ZOOM = "z4"
P12_ZOOM = "z5"
P10_ZOOM = "z6"

AQUEOUS_THICKNESS = cv_eval(cv, "thi", P41_ZOOM, S_CORNEA_B)
LENS_THICKNESS = cv_eval(cv, "thi", P41_ZOOM, S_LENS_F)
LENS_F_RADIUS = cv_eval(cv, "rdy", P41_ZOOM, S_LENS_F)
LENS_B_RADIUS = cv_eval(cv, "rdy", P41_ZOOM, S_LENS_B)
LENS_F_CONIC = cv_eval(cv, "k", P41_ZOOM, S_LENS_F)
LENS_B_CONIC = cv_eval(cv, "k", P41_ZOOM, S_LENS_B)
CORNEA_F_SEMI_AP = cv_eval(cv, "cir", S_CORNEA_B)
LENS_SEMI_AP = cv_eval(cv, "cir", S_LENS_F)

for zIdx, zz in enumerate(DZ):
    print(f"Tracing rays for Z={zz:.2f} mm ({zIdx+1}/{N_Z})...")
    cmd_zz = f"thi {S_IMG} {zz};"
    cv.Command(cmd_zz)
    for dIdx, A in enumerate(DP):
        print(f"Accommodation={A:.2f} D ({dIdx+1}/{N_DP})...")
        # Update eye parameters
        aquesous_thickness = AQUEOUS_THICKNESS + AQUEOUS_THICKNESS_SLP * A
        lensF_thickness = LENS_THICKNESS + LENS_THICKNESS_SLP * A
        lensF_radius = LENS_F_RADIUS + LENS_F_RADIUS_SLP * A
        lensF_conic = LENS_F_CONIC + LENS_F_CONIC_SLP * A
        lensB_radius = LENS_B_RADIUS + LENS_B_RADIUS_SLP * A
        lensB_conic = LENS_B_CONIC + LENS_B_CONIC_SLP * A
        lens_semi_ap = LENS_SEMI_AP + LENS_SEMI_AP_SLP * np.power(A, 0.81)

        cmd_lens_data = (
            f"thi {S_CORNEA_B} {P41_ZOOM} {aquesous_thickness};"
            f"thi {S_LENS_F} {P41_ZOOM} {lensF_thickness};"
            f"rdy {S_LENS_F} {P41_ZOOM} {lensF_radius};"
            f"rdy {S_LENS_B} {P41_ZOOM} {lensB_radius};"
            f"k {S_LENS_F} {P41_ZOOM} {lensF_conic};"
            f"k {S_LENS_B} {P41_ZOOM} {lensB_conic};"
        )
        cv.Command(cmd_lens_data)

        # Update GRIN glass lens
        cmd_glass = grin_glass_command(A, lens_semi_ap)
        cv.Command(cmd_glass)

        # eye rotation and tracing chief ray
        for rxIdx, rotX in enumerate(EYE_ROTATION):
            print(f"Eye Rotation={rotX:.2f} deg ({rxIdx+1}/{N_EYE_ROTATION})...")
            cv.Command(f"ade {S_RC} {rotX};set vig")

            # P1. no accommodation
            if dIdx == 0:
                data_p11[zIdx, dIdx, rxIdx] = trace_chief_ray(cv, P11_ZOOM)
                data_p12[zIdx, dIdx, rxIdx] = trace_chief_ray(cv, P12_ZOOM)
                data_p10[zIdx, dIdx, rxIdx] = trace_chief_ray(cv, P10_ZOOM)
            # P4
            data_p41[zIdx, dIdx, rxIdx] = trace_chief_ray(
                cv, 
                P41_ZOOM,
                position="y"
            )
            
            data_p42[zIdx, dIdx, rxIdx] = trace_chief_ray(
                cv, 
                P42_ZOOM,
                position="y"
            )

            data_p40[zIdx, dIdx, rxIdx] = trace_chief_ray(
                cv, 
                P40_ZOOM,
                position="y"
            )   
#%%
cv.StopCodeV()
pythoncom.CoUninitialize()

#%%
output_folder = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
output_file = "p14_sigma_atch_perfect_src_z.npz"
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

# fontsize = 26
# legend_fontsize = 10

# fig = plt.figure(figsize=(8, 6))
# ax1 = fig.add_subplot(111)
# ax1.plot(EYE_ROTATION, -data_p11[0, :], label="P11")
# ax1.plot(EYE_ROTATION, -data_p12[0, :], label="P12")
# ax1.plot(EYE_ROTATION, -data_p10[0, :], label="P10")
# for dIdx, dp in enumerate(DP):
#     #dIdx = int(dIdx * 10)
#     line = ax1.plot(EYE_ROTATION, -data_p41[dIdx, :], label=f"DP={dp:.2f}")
#     color = line[0].get_color()
#     ax1.plot(EYE_ROTATION, -data_p42[dIdx, :], label=f"DP={dp:.2f}", color=color)
#     ax1.plot(EYE_ROTATION, -data_p40[dIdx, :], label=f"DP={dp:.2f}", color=color)
# ax1.set_xlabel("Eye Rotation (deg)", fontsize=fontsize)
# ax1.set_ylabel("Position (mm)", fontsize=fontsize)
# ax1.tick_params(axis='both', which='major', labelsize=fontsize)
# ax1.legend(fontsize=legend_fontsize)
# ax1.grid()
