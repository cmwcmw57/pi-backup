import numpy as np, os, sys
from eval_analyze import smoothness
E=sys.argv[1] if len(sys.argv)>1 else "/workspace/logs/eval"
def vm(done,settle=8):
    T,N=done.shape; m=np.ones((T,N),bool); m[done.astype(bool)]=False
    for dt in range(1,settle+1):
        s=np.zeros_like(done,bool); s[dt:]=done[:-dt].astype(bool); m[s]=False
    return m
def feats(p):
    z=np.load(p,allow_pickle=True); m=vm(z["done"])
    cmd,vel=z["cmd"],z["vel"]
    exy=np.linalg.norm(cmd[...,:2]-vel[...,:2],axis=-1); eyaw=np.abs(cmd[...,2]-vel[...,2])
    r=float(z["wheel_radius_m"]); xy=np.asarray(z["module_xy"],float)
    th=z["steer"]; ww=z["wheel_w"]
    x=xy[:,0][None,None,:]; y=xy[:,1][None,None,:]
    vx=vel[...,0:1];vy=vel[...,1:2];wz=vel[...,2:3]
    vlong=(vx-wz*y)*np.cos(th)+(vy+wz*x)*np.sin(th); surf=ww*r
    den=np.maximum.reduce([np.abs(surf),np.abs(vlong),np.full_like(surf,0.05)])
    slip=(surf-vlong)/den; imb=np.abs(slip[...,[0,2]].mean(-1)-slip[...,[1,3]].mean(-1))
    ml,mr=z["mu_l"],z["mu_r"]; tot=ml+mr; dmu=np.abs(np.where(tot>0,(mr-ml)/np.maximum(tot,1e-6),0))
    am=np.linalg.norm(z["action"],axis=-1)
    csp=np.linalg.norm(cmd[...,:2],axis=-1); asp=np.linalg.norm(vel[...,:2],axis=-1)
    cw=np.abs(cmd[...,2]); aw=np.abs(vel[...,2])
    corr=np.corrcoef(dmu[m],am[m])[0,1] if am[m].std()>1e-6 else 0.0
    sm=smoothness(z,m)
    return dict(xy=exy[m].mean(),yaw=eyaw[m].mean(),slip=imb[m].mean(),res=am[m].mean(),corr=corr,vr=asp[m].mean()/csp[m].mean(),wr=aw[m].mean()/cw[m].mean(),**sm)
# 사용: compare_all.py [DIR] [이름=파일stem ...]  (파일 목록 주면 그것만, 첫 항목이 기준)
files=[("IK기준","baseline_ik"),("Res3","res3"),("Res8_old","res8"),("Direct8","d8"),("Res8_new","res8_new"),("Res8_slip","res8_slip"),("D8_base","d8_base"),("D8_smooth","d8_smooth")]
if len(sys.argv)>2:
    files=[tuple(a.split("=",1)) for a in sys.argv[2:]]
base_fn=files[0][1]
b=None
hdr="%-9s %7s %7s %7s %6s %6s %5s %5s   vsIK(xy/yaw/slip %%)" % ("policy","xy","yaw","slip","resid","r~dmu","vrat","wrat")
print(hdr)
for nm,fn in files:
    p="%s/%s.npz"%(E,fn)
    if not os.path.exists(p): continue
    f=feats(p)
    if fn==base_fn: b=f
    cmp=""
    if b is not None and fn!=base_fn:
        cmp="   %+.1f/%+.1f/%+.1f"%(100*(b["xy"]-f["xy"])/b["xy"],100*(b["yaw"]-f["yaw"])/b["yaw"],100*(b["slip"]-f["slip"])/b["slip"])
    print("%-9s %7.4f %7.4f %7.4f %6.2f %+6.2f %5.2f %5.2f%s"%(nm,f["xy"],f["yaw"],f["slip"],f["res"],f["corr"],f["vr"],f["wr"],cmp))
print()
print("매끄러움 (측정신호, 낮을수록 부드러움)")
print("%-9s %8s %8s %8s %8s %8s %8s %8s %8s" % ("policy","yaw_acc","str_rate","str_jerk","str_flip","whl_acc","tilt","side/min","dual%"))
for nm,fn in files:
    p="%s/%s.npz"%(E,fn)
    if not os.path.exists(p): continue
    f=feats(p)
    print("%-9s %8.3f %8.3f %8.1f %8.3f %8.2f %8.3f %8.3f %8.0f"%(nm,f["yaw_acc"],f["steer_rate"],f["steer_jerk"],f["steer_flip"],f["wheel_acc"],f["tilt_rate"],f["side_switch"],100*f["dual_zone"]))
