"""
Day-1 sanity runner:
- If a TUM-style GT file isn't found, it generates a tiny synthetic trajectory
  and uses it as both GT and EST to test the metrics (ATE≈0, RPE≈0).
- If found, it loads GT and (for now) sets EST=GT (you'll replace with tracker output later).
"""

import os, sys, math
import numpy as np
from pathlib import Path
from eval.metrics import ate_rmse, rpe_trans_rmse

DATA_DIR = Path("data")  # point this to your dataset (Colab: symlink Drive to ./data)

def se3_from_xyz_rpy(x, y, z, roll=0, pitch=0, yaw=0):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    R = Rz @ Ry @ Rx
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=[x,y,z]
    return T

def load_tum_gt(gt_path):
    """
    Expects TUM format:
    timestamp tx ty tz qx qy qz qw
    Returns list of 4x4 world_T_cam (cam pose in world).
    """
    Ts = []
    with open(gt_path,"r") as f:
        for line in f:
            if line.strip()=="" or line.startswith("#"): continue
            t, tx, ty, tz, qx, qy, qz, qw = map(float, line.strip().split())
            # quaternion to rotation (w,x,y,z = qw,qx,qy,qz)
            qw,qx,qy,qz = qw,qx,qy,qz
            R = quat_to_rot(qw,qx,qy,qz)
            T = np.eye(4); T[:3,:3]=R; T[:3,3]=[tx,ty,tz]
            Ts.append(T)
    return Ts

def quat_to_rot(w,x,y,z):
    n = math.sqrt(w*w+x*x+y*y+z*z)
    w,x,y,z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def synthetic_traj(N=50):
    Ts=[]
    for i in range(N):
        x = 0.01*i
        y = 0.0
        z = 0.0
        yaw = 0.01*i
        Ts.append(se3_from_xyz_rpy(x,y,z,0,0,yaw))
    return Ts

def main():
    # Try a common TUM GT location; change this path to your actual sequence
    default_gt = DATA_DIR / "tum" / "fr1_desk" / "groundtruth.txt"
    if default_gt.exists():
        gt_T = load_tum_gt(default_gt)
        est_T = gt_T.copy()  # placeholder; replace later with your tracker output
        print("[info] Loaded GT from", default_gt)
    else:
        print("[warn] No GT found at", default_gt, "-- using synthetic sanity trajectory.")
        gt_T = synthetic_traj(60)
        est_T = gt_T.copy()

    gt_T = np.asarray(gt_T)
    est_T = np.asarray(est_T)

    ate = ate_rmse(gt_T, est_T)
    rpe = rpe_trans_rmse(gt_T, est_T, delta=1)
    print(f"ATE_RMSE (m): {ate:.6f}")
    print(f"RPE_trans_RMSE (m/frame): {rpe:.6f}")

if __name__ == "__main__":
    main()
