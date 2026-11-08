"""Standalone P4 combined Brown gaze model and its numerical inverse.

Public coordinates are millimetres; A is diopters and ADE/BDE are degrees.
Point arrays have shape (..., fields, 2), with state parameters broadcast over
the leading axes. The frozen order-2 coefficients are loaded from baseline_check.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data/p4_combined_brown/baseline_check/combined_model_order2_1.npz"
L = 1.235429592627
IDENTITY = np.array([1., 0., 0., 1., 0., 0., 0., 0., 0.])
OUTER = np.array([0, 1, 2, 3, 5, 6, 7, 8])
K = np.array([-.0185004967, .00023989235714, .0015756063948,
              -.0014256378340, .00066200789981])


def _barrel_k_s(A):
    v = np.asarray(A, float) / 4
    return (sum(K[i] * v**i for i in range(5)), 1 - .081595476571*v)


def barrel(points, A):
    """Physical B_A from the on-axis calibration (mm -> mm)."""
    p = np.asarray(points, float)
    a = np.asarray(A, float)
    shape = np.broadcast_shapes(p.shape[:-1], a.shape)
    p = np.broadcast_to(p, shape+(2,)); a = np.broadcast_to(a, shape)
    k, s = _barrel_k_s(a)
    r = np.linalg.norm(p, axis=-1)
    rho = r.copy(); k0 = K[0]
    for _ in range(10):
        rho -= (rho + k0*rho**3-r)/(1+3*k0*rho**2)
    u = p * np.divide(rho, r, out=np.ones_like(r), where=r>0)[...,None]
    q = s[...,None]*u
    out = q*(1+k[...,None]*np.sum(q*q, axis=-1, keepdims=True))
    return np.where((a==0)[...,None],p,out)


def inverse_barrel(points, A):
    """Invert B_A on its small monotone radial branch."""
    y=np.asarray(points,float); a=np.asarray(A,float)
    shape=np.broadcast_shapes(y.shape[:-1],a.shape)
    y=np.broadcast_to(y,shape+(2,)); a=np.broadcast_to(a,shape)
    k,s=_barrel_k_s(a); target=np.linalg.norm(y,axis=-1)
    # Undo D_k(A), divide by s, then apply D_k(0).
    q=target.copy()
    for _ in range(12): q-=(q+k*q**3-target)/(1+3*k*q**2)
    rho=np.divide(q,s,out=np.zeros_like(q),where=s!=0)
    r0=rho*(1+K[0]*rho*rho)
    return y*np.divide(r0,target,out=np.ones_like(target),where=target>0)[...,None]


def load_coefficients(path=MODEL_PATH):
    """Return (order, C) where C has shape (14*(order+1),9)."""
    with np.load(path,allow_pickle=False) as d:
        order=int(d['selected_legendre_order'])
        c=np.asarray(d['c1_coef'],float).copy()
    if order != 2 or c.shape != (42,9):
        raise ValueError(f"Expected frozen order-2 42x9 coefficients; got order={order}, shape={c.shape}")
    return order,c


def parameters(A,ADE,BDE,coefficients=None):
    order,c=load_coefficients() if coefficients is None else coefficients
    A,ADE,BDE=np.broadcast_arrays(np.asarray(A,float),np.asarray(ADE,float),np.asarray(BDE,float))
    tx,ty=BDE/20,ADE/20
    mono=[(i,d-i) for d in range(1,5) for i in range(d,-1,-1)]
    tv=np.stack([tx**i*ty**j for i,j in mono],axis=-1)
    leg=np.polynomial.legendre.legvander(A/2-1,order)
    basis=(leg[..., :,None]*tv[...,None,:]).reshape(A.shape+(42,))
    return IDENTITY+basis@c


def forward(r0,A,ADE,BDE,coefficients=None,return_denominator=False):
    """Evaluate B_A -> projective H -> Brown-Conrady -> radial r2 (all mm).

    r0 shape (...,fields,2); A/ADE/BDE broadcast to r0.shape[:-2].
    """
    r0=np.asarray(r0,float); lead=r0.shape[:-2]
    A,ADE,BDE=[np.broadcast_to(np.asarray(x,float),lead) for x in (A,ADE,BDE)]
    p=parameters(A,ADE,BDE,coefficients)
    b=barrel(r0,A[...,None])/L
    x,y=b[...,0],b[...,1]
    m11,m12,m21,m22,qx,qy,p1,p2,g=[p[...,i,None] for i in range(9)]
    den=1+qx*x+qy*y
    ux=(m11*x+m12*y)/den; uy=(m21*x+m22*y)/den
    wx=ux+2*p1*ux*uy+p2*(3*ux*ux+uy*uy)
    wy=uy+p1*(ux*ux+3*uy*uy)+2*p2*ux*uy
    r2=wx*wx+wy*wy
    pred=L*np.stack((wx*(1+g*r2),wy*(1+g*r2)),axis=-1)
    return (pred,den) if return_denominator else pred


def inverse(r,A,ADE,BDE,coefficients=None,return_denominator=False):
    """Invert forward map; output fixed-reference field coordinates in mm.

    Inverts radial r2 by scalar Newton, Brown-Conrady by 2-D Newton, projective
    H algebraically, then B_A. Coefficient/state broadcasting matches forward().
    """
    y=np.asarray(r,float); lead=y.shape[:-2]
    A,ADE,BDE=[np.broadcast_to(np.asarray(x,float),lead) for x in (A,ADE,BDE)]
    pars=parameters(A,ADE,BDE,coefficients)
    m11,m12,m21,m22,qx,qy,p1,p2,g=[pars[...,i,None] for i in range(9)]
    z=y/L
    # Undo radial r2: z = w*(1+g*|w|^2), a radial scalar cubic.
    rout=np.linalg.norm(z,axis=-1); rw=rout.copy()
    for _ in range(20): rw-=(rw*(1+g*rw**2)-rout)/(1+3*g*rw**2)
    z=z*np.divide(rw,rout,out=np.ones_like(rout),where=rout>0)[...,None]
    brown_target=z.copy()
    # Newton solve Brown-Conrady alone in projective-H output coordinates.
    for _ in range(30):
        x,yc=z[...,0],z[...,1]
        f1=x+2*p1*x*yc+p2*(3*x*x+yc*yc)
        f2=yc+p1*(x*x+3*yc*yc)+2*p2*x*yc
        # The Brown target is the radial-inverted vector, held separately.
        e1=f1-brown_target[...,0]; e2=f2-brown_target[...,1]
        j11=1+2*p1*yc+6*p2*x
        j12=2*p1*x+2*p2*yc
        j21=2*p1*x+2*p2*yc
        j22=1+6*p1*yc+2*p2*x
        det=j11*j22-j12*j21
        dx=(j22*e1-j12*e2)/det; dy=(-j21*e1+j11*e2)/det
        z-=np.stack((dx,dy),-1)
    ux,uy=z[...,0],z[...,1]
    # H^{-1}: u = M b/(1+q.b), hence b=(M-u q^T)^-1 u.
    aa=m11-ux*qx; ab=m12-ux*qy; ac=m21-uy*qx; ad=m22-uy*qy
    hd=aa*ad-ab*ac
    bx=(ad*ux-ab*uy)/hd; by=(-ac*ux+aa*uy)/hd
    den=1+qx*bx+qy*by
    b=np.stack((bx,by),-1)
    out=inverse_barrel(b*L,A[...,None])
    return (out,den) if return_denominator else out
