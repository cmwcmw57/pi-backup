# ================================================================
# run.py
# 설명: 기존 Isaac 경기장·로버를 임무 feedback으로 구동하고 영상·costmap·시계열을 기록한다.
# 사용법: /isaac-sim/python.sh -m experiments.site_sim.run --mission MISSION --urdf ROVER --geometry JSON --scene-usd USD --output NEW_RUN
# ================================================================
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import time

# 영상 장면이 붙은 timestamp보다 늦은 시뮬 렌더 지연(2026-10-17 GT 회전속도와의 시차 스캔으로 측정, 약 0.10 s).
RENDER_LAG_S=0.1


def validate_limits(sim_s: float, wall_s: float) -> None:
    """물리와 wall 실행 상한을 유한한 범위로 제한한다."""
    if not math.isfinite(sim_s) or not 0<sim_s<=900 or not math.isfinite(wall_s) or not 0<wall_s<=3600:
        raise ValueError('bounded sim (0,900] and wall (0,3600] limits required')


def filter_noise_overrides(args: argparse.Namespace) -> dict:
    """필터가 가정하는 IMU 잡음 덮어쓰기: filter_acc_noise는 IMU.NoiseAcc 절댓값, filter_gyro_noise_scale은 DEFAULT_NOISE의 IMU.NoiseGyro 배율."""
    from experiments.localization.online import DEFAULT_NOISE
    out={}
    if getattr(args,'filter_acc_noise',None) is not None:out['IMU.NoiseAcc']=float(args.filter_acc_noise)
    scale=getattr(args,'filter_gyro_noise_scale',None)
    if scale is not None:
        if not scale>0:raise ValueError('--filter-gyro-noise-scale must be > 0')
        out['IMU.NoiseGyro']=DEFAULT_NOISE['IMU.NoiseGyro']*float(scale)
    freeze=getattr(args,'filter_gyro_bias_init_sigma_dps',None)
    if freeze is not None and float(freeze) <= .01:out['IMU.GyroWalk']=1e-12   # 시작 정지 평균으로 bias를 묶고 주행 중 갱신을 막는다
    return out


def build_estimator(args: argparse.Namespace, start_pose, actual_start_pose, calibration: dict):
    """실행 옵션(IMU 프로필·시드·누락·방향 관측·회전 영상 σ)으로 OnlineEstimator를 만든다."""
    from experiments.localization.imu_realism import ImuRealism
    from experiments.localization.online import OnlineEstimator
    imu_realism=(ImuRealism.stress if getattr(args,'imu_profile','nominal')=='stress' else ImuRealism)(
        seed=getattr(args,'imu_seed',0),vibration_scale=getattr(args,'imu_vibration_scale',1.),
        drop_probability=float(getattr(args,'imu_drop_prob',0.))) if getattr(args,'imu_realism',False) else None
    return OnlineEstimator(start_pose,calibration,lag_s=RENDER_LAG_S,adaptive_visual_noise=getattr(args,'adaptive_visual_noise',False),
                           imu_realism=imu_realism,direction_sigma=getattr(args,'velocity_direction_sigma',None),sensor_start_world_from_base=actual_start_pose,
                           gyro_bias_init_hold_s=None if getattr(args,'filter_gyro_bias_init_sigma_dps',None) is None else float(args.start_hold_s),
                           absolute_attitude_sigma_deg=getattr(args,'filter_absolute_attitude_sigma_deg',None),
                           gyro_bias_sigma_dps=.03 if getattr(args,'filter_gyro_bias_init_sigma_dps',None) is None else float(args.filter_gyro_bias_init_sigma_dps),
                           visual_schedule=None if getattr(args,'visual_turn_scale',None) is None else {'turn_rotation_scale':float(args.visual_turn_scale),'turn_position_scale':float(args.visual_turn_scale)})


def policy_imu_metadata(rl) -> dict:
    """실제 정책 객체의 IMU 모델·부팅 bias 보정 시간을 실행 기록에 남긴다."""
    return {'imu_for_policy': ('physics state, no noise or bias' if rl.imu_realism is None
                               else 'physics state with simulated IMU noise and bias'),
            'imu_realism': None if rl.imu_realism is None else rl.imu_realism.parameters(),
            'gyro_bias_estimation_s': float(rl.bias_s)}


def apply_follower_args(mission, args: argparse.Namespace) -> None:
    """--follower blend이면 blend 추종기를 켜고, --blend-max-yaw-rate가 있으면 각속도 상한(0 초과 max_yaw_rate_rad_s 이하)을 설정에 넣는다."""
    if getattr(args,'follower',None)=='blend':mission.settings['follower_blend']=1.
    rate=getattr(args,'blend_max_yaw_rate',None)
    if rate is not None:
        if not math.isfinite(rate) or not 0<rate<=mission.settings['max_yaw_rate_rad_s']:
            raise ValueError('--blend-max-yaw-rate must be in (0, max_yaw_rate_rad_s]')
        mission.settings['blend_max_yaw_rate_rad_s']=float(rate)


def rtx_setting_overrides(args: argparse.Namespace) -> list[tuple[str, int]]:
    """--rtx-aa-op(0 끔·1 TAA·2 FXAA·3 DLSS·4 RTXAA)·--dlss-exec-mode(0 Performance·1 Balanced·2 Quality·3 Auto)가 있으면 carb 설정 (키, 값) 목록으로 만든다."""
    out=[]
    for attr,key,high in (('rtx_aa_op','/rtx/post/aa/op',4),('dlss_exec_mode','/rtx/post/dlss/execMode',3)):
        value=getattr(args,attr,None)
        if value is None:continue
        if type(value) is not int or not 0<=value<=high:raise ValueError(f'--{attr.replace("_","-")} must be an integer in [0, {high}]')
        out.append((key,value))
    return out


def load_mission(args: argparse.Namespace):
    """임무 폴더를 읽고 추종기·구역 MPPI 실행 옵션을 설정에 반영한다."""
    from experiments.handheld_mapping.site_mission import SiteMission
    mission=SiteMission.load(args.mission,load_mesh=False)
    if getattr(args,'stanley_gain',0.)>0:mission.settings['stanley_gain']=float(args.stanley_gain)
    apply_follower_args(mission,args)
    if getattr(args,'navigation','path')=='region-mppi':
        speed=float(getattr(args,'mppi_max_speed',.15))
        if not math.isfinite(speed) or not 0<speed<=.15:raise ValueError('--mppi-max-speed must be in (0, 0.15]')
        mission.settings['max_speed_m_s']=speed
    return mission


def observation_options(args: argparse.Namespace) -> tuple[str, float]:
    """기존 동기 관측은 유지하고 새 비교 조건에만 합의한 주기를 적용한다."""
    mode=getattr(args,'region_observation','depth-sync')
    if mode not in ('depth-sync','depth','pixel','none'):raise ValueError('invalid region observation mode')
    hz=getattr(args,'region_observation_hz',None)
    if hz is None:hz={'depth-sync':2.5,'depth':1.,'pixel':5.,'none':0.}[mode]
    if mode!='none' and (not math.isfinite(hz) or not 0<hz<=15):raise ValueError('region observation Hz must be in (0,15]')
    return mode,float(hz)


def region_observation_metadata(observation: dict, now_s: float) -> dict:
    """원래 영상 시각과 렌더 지연을 구분해 관측 age·처리 시간을 수치로 기록한다."""
    stamp=float(observation['stamp_s'])
    valid_stamp=math.isfinite(stamp) and stamp>=0.
    return {key:observation.get(key) for key in ('valid','reason','processing_ms','observation_mode')} | {
        'stamp_s':stamp if valid_stamp else None,
        'age_s':now_s-stamp if valid_stamp else None,
        'content_stamp_s':stamp-RENDER_LAG_S if valid_stamp else None,
        'content_age_s':now_s-stamp+RENDER_LAG_S if valid_stamp else None}


def run(args: argparse.Namespace) -> dict:
    validate_limits(args.max_sim_s,args.max_wall_s)
    from experiments.site_sim.physics_sync import validate_options
    validate_options(args)
    parity_physics=getattr(args,'parity_physics',False)
    import numpy as np
    from experiments.handheld_mapping.storage import atomic_json
    from experiments.site_sim.recording import RunRecorder
    from experiments.site_sim.estimator_log import EstimatorLog
    mission=load_mission(args)
    actuator_model=getattr(args,'actuator_model','legacy')
    soil_model=getattr(args,'soil_model','fixed')
    if soil_model=='ksrc-legacy' and getattr(args,'mu',None) is None:
        raise ValueError('ksrc-legacy soil requires explicit --mu as mu_max')
    if actuator_model=='ksrc-pi' and (not getattr(args,'rl_policy',None) or abs(getattr(args,'control_dt',.05)-.02)>1e-9):
        raise ValueError('ksrc-pi requires the existing RL policy and control_dt 0.02')
    stereo_hz=float(getattr(args,'stereo_hz',0.))
    if not math.isfinite(stereo_hz) or not 0<=stereo_hz<=30:raise ValueError('stereo_hz must be in [0,30]')
    control_dt=float(getattr(args,'control_dt',.05))
    if not math.isfinite(control_dt) or abs(control_dt/.005-round(control_dt/.005))>1e-6 or not .005<=control_dt<=.05:
        raise ValueError('control_dt must be a multiple of the 5 ms physics step within [0.005, 0.05]')
    if getattr(args,'rl_policy',None) and abs(control_dt-.02)>1e-9:raise ValueError('--rl-policy needs --control-dt 0.02 (policy trained at 50 Hz)')
    if getattr(args,'rl_policy',None) and not getattr(args,'ksrc_root',None):raise ValueError('--rl-policy needs --ksrc-root')
    region_mode=getattr(args,'navigation','path')=='region-mppi'
    if region_mode and (not getattr(args,'estimator',False) or stereo_hz<=0):
        raise ValueError('region-mppi requires --estimator and --stereo-hz')
    if region_mode and args.fault_injection:raise ValueError('region-mppi requires a physical attempt without fault injection')
    observer=None;observation_worker=None;next_observation=0.;wall_yaw=None;wall_yaw_count=0;wall_track=None;wall_track_count=0;track_lat=[];track_front=[];track_corner=[]
    observation_mode,observation_hz=observation_options(args)
    recorder=RunRecorder(mission,args.output,fault_injection=args.fault_injection,control_dt=control_dt,
                         pose_source='orb_eskf_estimate' if getattr(args,'estimator',False) else 'sim_ground_truth')
    stereo_writer=None;next_stereo=0.;estimator=orb=estimator_log=None;frame_info=None
    atomic_json(Path(args.output)/'run_options.json',{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()})   # 재생 화면에 어떤 설정이었는지 보이게
    if getattr(args,'estimator',False) and stereo_hz<=0:raise ValueError('--estimator needs --stereo-hz')
    app=drive=world=None;status='failed';reason='initialization_failed';cameras={};wall_start=time.monotonic()
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    previous_signal=signal.signal(signal.SIGTERM,interrupted)
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,
                           'width':800,'height':600,'anti_aliasing':0})
        import omni.usd
        import omni.replicator.core as rep
        import carb.settings
        rtx=carb.settings.get_settings();rtx_keys=('/rtx/post/aa/op','/rtx/post/dlss/execMode','/rtx/rendermode')
        print('SITE_SIM_RTX before',{k:rtx.get(k) for k in rtx_keys},flush=True)
        for key,value in rtx_setting_overrides(args):rtx.set(key,value)
        print('SITE_SIM_RTX after',{k:rtx.get(k) for k in rtx_keys},flush=True)
        from isaacsim.core.api.robots import Robot
        from pxr import Gf,UsdGeom
        from PIL import Image
        from experiments.orbslam3_ros2.isaac_scene import build_isaac_scene
        from experiments.orbslam3_ros2.drive_runtime import DriveSession
        from experiments.site_sim.drive import SimulationDrive
        site=mission.site;c=site.cells([mission.start[:2]])[0];ground=float(site.height_m[c[1],c[0]])
        print('SITE_SIM_STAGE scene_import',flush=True)
        world,h=build_isaac_scene(urdf=args.urdf,scene_usd=args.scene_usd,output=Path(args.output),
            start_position=(*mission.start[:2],ground+.23))
        for _ in range(5):app.update()
        stage=omni.usd.get_context().get_stage()
        if parity_physics:
            from experiments.site_sim.physics_sync import bind_ground_friction
            parity_ground=bind_ground_friction(stage)
        if soil_model=='ksrc-legacy':
            from experiments.site_sim.ksrc_soil import prepare_contact_reports
            prepare_contact_reports(stage)
        if actuator_model=='ksrc-pi':
            from experiments.site_sim.ksrc_actuation import configure_joint_physics
            from experiments.orbslam3_ros2.swerve import SwerveGeometry
            changed=configure_joint_physics(stage,SwerveGeometry.from_json(args.geometry))
            atomic_json(Path(args.output)/'actuator_physics.json',{'model':actuator_model,'joints':changed,
                'armature_kg_m2':.0098,'steer_vmax_rad_s':5.45,'servo_accel_rad_s2':7.7,'servo_delay_s':.034,
                'wheel_pi':'firmware/common/wheel_pi.c','drive_gate':'firmware/common/drive_align_gate.c'})
        if getattr(args,'mu',None) is not None and soil_model=='fixed':
            from experiments.site_sim.friction import bind_friction
            grounds,wheels=bind_friction(stage,float(args.mu));print('SITE_SIM_FRICTION',args.mu,'ground meshes',grounds,'wheel colliders',wheels,flush=True)
        from experiments.site_sim.lighting import configure_lighting
        lighting=configure_lighting(stage,getattr(args,'lighting_profile','diffuse'))
        atomic_json(Path(args.output)/'lighting.json',lighting)
        from experiments.site_sim.wall_material import set_black_walls
        if getattr(args,'wall_material','black')=='black':atomic_json(Path(args.output)/'wall_material.json',set_black_walls(stage))
        else:atomic_json(Path(args.output)/'wall_material.json',{'material':'legacy','physics_changed':False})
        if args.sand_texture:
            from pxr import Sdf,UsdShade,Vt
            sand=stage.GetPrimAtPath('/World/SandScene/Sand');pts=np.asarray(UsdGeom.Mesh(sand).GetPointsAttr().Get(),np.float32)
            lo=pts[:,:2].min(0);span=pts[:,:2].max(0)-lo
            print('SITE_SIM_SAND_EXTENT',lo.tolist(),span.tolist(),flush=True)
            # Texture is 0.68 mm/px: 8192x6144 px covers 5.6 x 4.2 m, anchored at the sand corner.
            uv=(pts[:,:2]-lo)/np.array([5.6,4.2],np.float32)
            UsdGeom.PrimvarsAPI(sand).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv))
            material=UsdShade.Material.Define(stage,'/World/Looks/SandTexture')
            shader=UsdShade.Shader.Define(stage,'/World/Looks/SandTexture/Shader');shader.CreateIdAttr('UsdPreviewSurface')
            reader=UsdShade.Shader.Define(stage,'/World/Looks/SandTexture/StReader');reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
            texture=UsdShade.Shader.Define(stage,'/World/Looks/SandTexture/Texture');texture.CreateIdAttr('UsdUVTexture')
            texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(args.sand_texture))
            texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
            texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.);shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(sand).Bind(material)
        overview=UsdGeom.Camera.Define(stage,'/World/ObserverCamera')
        overview.CreateFocalLengthAttr(18.);overview.CreateHorizontalApertureAttr(20.)
        overview.CreateVerticalApertureAttr(15.);overview.CreateClippingRangeAttr(Gf.Vec2f(.1,100))
        center=site.origin+np.asarray(site.observed.shape[::-1])*site.resolution/2
        transform=Gf.Matrix4d().SetLookAt(Gf.Vec3d(float(center[0]),float(center[1])-4.5,5.5),
                                       Gf.Vec3d(float(center[0]),float(center[1]),.35),Gf.Vec3d(0,0,1)).GetInverse()
        overview_op=UsdGeom.Xformable(overview).AddTransformOp();overview_op.Set(transform)
        # The latest URDF already mounts the camera 20 deg down; no optical-frame compensation.
        # Measured D435 IR 640x480 intrinsics (fx=fy=377.186, principal point 320.54/241.73).
        FX,CX,CY=377.186126708984,320.54248046875,241.732162475586
        for path in (h.left_camera_path,h.right_camera_path):
            camera=UsdGeom.Camera(stage.GetPrimAtPath(path))
            camera.GetFocalLengthAttr().Set(FX*20./640)
            camera.GetHorizontalApertureOffsetAttr().Set((320-CX)*20./640)
            camera.GetVerticalApertureOffsetAttr().Set((CY-240)*15./480)
        camera_paths=[('overview','/World/ObserverCamera'),('front',h.left_camera_path)]
        if stereo_hz:camera_paths.append(('right',h.right_camera_path))
        for key,path in camera_paths:
            product=rep.create.render_product(path,(800,600) if key=='overview' else (640,480))
            annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach(product)
            cameras[key]=(product,annotator)
        if stereo_hz:
            from experiments.site_sim.stereo_capture import StereoCapture
            from experiments.orbslam3_ros2.isaac_scene import _column_world_matrix
            base=_column_world_matrix(stage.GetPrimAtPath(h.frame_paths['base_link']))
            left=_column_world_matrix(stage.GetPrimAtPath(h.frame_paths['camera_infra1_optical_frame']))
            right=_column_world_matrix(stage.GetPrimAtPath(h.frame_paths['camera_infra2_optical_frame']))
            stereo_writer=StereoCapture(Path(args.output)/'stereo',{
                'width':640,'height':480,'fx':FX,'fy':FX,'cx':CX,'cy':CY,'distortion':[0.,0.,0.,0.],
                'T_base_left_optical':(np.linalg.inv(base)@left).tolist(),
                'T_base_right_optical':(np.linalg.inv(base)@right).tolist(),
                'T_left_right':(np.linalg.inv(left)@right).tolist(),
                'requested_hz':stereo_hz,'clock':'simulation relative to control start','timestamp_unit':'ns',
                'intrinsics_source':'configured USD pinhole camera; not real sensor calibration',
                'capture':'both RGB annotators after the same world.step(render=True); no IMU',
                'urdf_sha256':__import__('hashlib').sha256(Path(args.urdf).read_bytes()).hexdigest(),
                'archive_images':bool(getattr(args,'record_images',False))},
                archive=bool(getattr(args,'record_images',False)),transport_dir=Path(args.orb_io)/'frames')
        if getattr(args,'estimator',False):
            from experiments.localization.online import DEFAULT_NOISE
            imu_world=_column_world_matrix(stage.GetPrimAtPath(h.frame_paths['imu_sensor_frame']))
            base_from_imu=np.linalg.inv(base)@imu_world;base_from_cam=np.linalg.inv(base)@left
            estimator_calibration={'ROS.T_c1_base':np.linalg.inv(base_from_cam).tolist(),
                                   'IMU.T_b_c1':(np.linalg.inv(base_from_imu)@base_from_cam).tolist(),**DEFAULT_NOISE,
                                   **filter_noise_overrides(args)}
        from experiments.site_sim.visuals import install_rover_visuals
        install_rover_visuals(stage,args.urdf)
        try:   # 경기장(x 2.4~6.6, y ±1.7) 밖에 놓인 물체가 있는지 진단용으로 한 번 기록한다.
            from pxr import Usd
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
            for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim) or not prim.IsActive():continue
                box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if box.IsEmpty():continue
                lo,hi=np.array(box.GetMin()),np.array(box.GetMax());mid=(lo+hi)/2
                if not (2.4<mid[0]<6.6 and -1.7<mid[1]<1.7):   # 경기장 밖에 놓인 물체
                    print('SITE_SIM_OUTSIDE_PRIM',prim.GetPath(),np.round(mid,2).tolist(),np.round(hi-lo,2).tolist(),
                          'visible' if UsdGeom.Imageable(prim).ComputeVisibility()!='invisible' else 'hidden',flush=True)
        except Exception as error:print('SITE_SIM_ORIGIN_PRIM scan failed',error,flush=True)
        robot=world.scene.add(Robot(prim_path=h.frame_paths['base_link'],name='rover'))
        world.set_simulation_dt(physics_dt=.005,rendering_dt=control_dt)
        print('SITE_SIM_STAGE physics_initialization',flush=True)
        world.initialize_physics();robot.initialize()
        from experiments.site_sim.start_noise import start_offset
        offset_x,offset_y,offset_yaw=start_offset(int(getattr(args,'start_seed',0)),float(getattr(args,'start_noise_xy_m',0.)),float(getattr(args,'start_noise_yaw_deg',0.)))
        yaw=float(mission.start[2])+offset_yaw;position0,_=robot.get_world_pose()
        robot.set_world_pose(position=np.asarray(position0,float)+np.array([offset_x,offset_y,0.]),orientation=np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)]))
        atomic_json(Path(args.output)/'start_perturbation.json',{'dx_m':offset_x,'dy_m':offset_y,'dyaw_rad':offset_yaw,'seed':int(getattr(args,'start_seed',0)),
            'note':'로버만 이만큼 틀어 놓고 추정기는 이상적인 시작 자세를 믿는다'})
        session=DriveSession(robot,args.geometry);world.play()
        if actuator_model=='ksrc-pi':
            from experiments.site_sim.ksrc_actuation import configure_drives
            configure_drives(session)
        soil=None
        if soil_model=='ksrc-legacy':
            from experiments.site_sim.ksrc_soil import IsaacSoil
            if parity_physics:
                from experiments.site_sim.physics_sync import TensorMaterialSoil
                soil=TensorMaterialSoil(stage,robot,session,float(args.mu))
            else:soil=IsaacSoil(stage,robot,session,float(args.mu))
            soil.initialize()
        if getattr(args,'wheel_effort',None) is not None:
            session.controller.set_max_efforts(np.full(4,float(args.wheel_effort),dtype=float),session.wheel_indices)
        print('SITE_SIM_STAGE settling',flush=True)
        actuator=None
        if parity_physics:
            from experiments.site_sim.physics_sync import CoulombActuation
            from experiments.site_sim.rl_residual import load_student_runtime
            actuator=CoulombActuation(robot,session,load_student_runtime(Path(args.ksrc_root)).kc)
        for i in range(400):
            if time.monotonic()-wall_start>args.max_wall_s:raise TimeoutError('startup wall timeout')
            session.stop()
            if parity_physics:actuator.stop();soil.apply()
            world.step(render=i%20==0)
        if parity_physics:
            from experiments.site_sim.physics_sync import physics_evidence
            atomic_json(Path(args.output)/'parity_physics.json',physics_evidence(stage,soil,parity_ground))
        p,q=robot.get_world_pose()
        if not np.isfinite(np.r_[p,q]).all() or np.linalg.norm(p[:2]-mission.start[:2])>.1 or not ground+.08<p[2]<ground+.35:
            raise ValueError(f'rover settling failed: {p}')
        if getattr(args,'estimator',False):
            from experiments.site_sim.orb_client import OrbClient
            from scipy.spatial.transform import Rotation
            start_rotation=Rotation.from_euler('z',-offset_yaw).as_matrix()@Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()   # 틀기 전(이상적) 자세
            start_pose=np.eye(4);start_pose[:3,:3]=start_rotation;start_pose[:3,3]=np.asarray(p,float)-np.array([offset_x,offset_y,0.])
            actual_start_pose=np.eye(4);actual_start_pose[:3,:3]=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix();actual_start_pose[:3,3]=np.asarray(p,float)   # 로버의 실제 시작 자세(센서 이력용)
            estimator=build_estimator(args,start_pose,actual_start_pose,estimator_calibration);imu_realism=estimator.imu_realism
            atomic_json(Path(args.output)/'estimator_calibration.json',{
                'calibration':estimator_calibration,'start_world_from_base':start_pose.tolist(),
                'render_lag_s':RENDER_LAG_S,'adaptive_visual_noise':estimator.adaptive_visual_noise,
                'imu_realism':None if imu_realism is None else imu_realism.parameters(),
                'simulation_only':True})
            orb=OrbClient(Path(args.orb_io));estimator_log=EstimatorLog(Path(args.output))
            print('SITE_SIM_ESTIMATOR ready (control uses ORB+ESKF estimate; GT is logged only)',flush=True)
        from experiments.site_sim.drive import SlipInjection
        slip=SlipInjection(getattr(args,'wheel_gain_left',1.),getattr(args,'wheel_gain_right',1.),getattr(args,'slip_start_s',0.),
                           math.inf if getattr(args,'slip_end_s',None) is None else args.slip_end_s)
        rl=None
        if getattr(args,'rl_policy',None):
            from experiments.site_sim.rl_residual import RlResidual
            from experiments.localization.imu_realism import ImuRealism as _Imu
            if args.start_hold_s<2.:raise ValueError('RL boot bias needs --start-hold-s >= 2')
            rl=RlResidual(Path(args.ksrc_root),args.rl_policy,dt=control_dt,bias_s=2.,
                          imu_realism=(_Imu.stress if getattr(args,'imu_profile','nominal')=='stress' else _Imu)(seed=int(getattr(args,'imu_seed',0))+1,vibration_scale=getattr(args,'imu_vibration_scale',1.)) if getattr(args,'imu_realism',False) else None)
            atomic_json(Path(args.output)/'rl_residual.json',{'policy':args.rl_policy,'ksrc_root':str(args.ksrc_root),'control_dt_s':control_dt,
                'meta':{k:rl.meta.get(k) for k in ('task','checkpoint','obs_dim','student_history','control_dt_s','res8_steer_scale_rad','res8_wheel_scale','res8_rate_limit')},
                'simulation_only':True,**policy_imu_metadata(rl)})
            print('SITE_SIM_RL residual policy loaded:',args.rl_policy,flush=True)
        delay=None
        if getattr(args,'command_delay_ms',None):
            from experiments.site_sim.drive import ActuationDelay
            delay=ActuationDelay(args.command_delay_ms[0]/1000,args.command_delay_ms[1]/1000,int(getattr(args,'delay_seed',0)))
        runtime=None
        if region_mode:
            from experiments.handheld_mapping.region_mission import RegionMission
            from experiments.handheld_mapping.region_observation import RegionObservation
            from experiments.handheld_mapping.sensor_safety import SensorSafety
            scenario_path=Path(args.mission)/'scenario.json'   # 새 경기장: 구역·통로·단계 좌표. 없으면 옛 gyusa 경기장 값
            scenario=json.loads(scenario_path.read_text()) if scenario_path.exists() else None
            if scenario and 'regions_file' in scenario:
                regions_data=json.loads((Path(args.mission)/scenario['regions_file']).read_text());scenario['regions']=regions_data['regions'];scenario['portals']=regions_data['portals']
            runtime=RegionMission(mission,safety=SensorSafety(require_imu=True),hold_s=args.start_hold_s,scenario=scenario)
            runtime.corner_gate=bool(args.corner_gate)
            runtime.planner.max_lateral=float(args.mppi_vy);runtime.planner.clearance_weight=float(args.mppi_clearance);runtime.planner.clearance_margin=float(args.mppi_clearance_margin)
            observation_cal=json.loads((Path(args.output)/'stereo/calibration.json').read_text())
            if getattr(args,'wall_plan',False):
                from experiments.site_sim.wall_track import WallTrack
                wall_track=WallTrack(observation_cal,mission.site)
            if getattr(args,'wall_yaw',False):
                from experiments.site_sim.wall_yaw import WallYaw
                wall_yaw=WallYaw(observation_cal,site=mission.site,sigma_deg=args.wall_yaw_sigma_deg)
            if observation_mode=='depth-sync':observer=RegionObservation(mission.site,observation_cal,runtime.regions)
            elif observation_mode!='none':
                from experiments.handheld_mapping.async_region_observation import AsyncRegionObservation,next_observation_stamp
                observation_worker=AsyncRegionObservation(mission.site,observation_cal,runtime.regions,mode=observation_mode)
        if actuator_model=='ksrc-pi' and actuator is None:
            from experiments.site_sim.ksrc_actuation import FirmwareActuation
            actuator=FirmwareActuation(robot,session,rl.runtime.kc)
        drive=SimulationDrive(robot,session,mission,fault_injection=args.fault_injection,hold_s=args.start_hold_s,estimator=estimator,slip=slip,rl=rl,delay=delay,runtime=runtime,actuator=actuator)
        t0=float(world.current_time);next_control=0.;next_render=0.;next_front=0.;next_overview=0.;last_camera_error=None;rtx_capture_logged=False
        print('SITE_SIM_READY',json.dumps({'pose':p.tolist(),'output':str(args.output)}),flush=True)
        try:   # 안정 뒤 로버 시각 형상(ObserverVisual)과 로버·경기장 밖 최상위 prim의 월드 위치를 기록한다(진단).
            from pxr import Usd
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
            for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
                path=str(prim.GetPath());top=path.split('/')[1:3]
                if 'ObserverVisual' in path or (len(top)==2 and path.count('/')==2 and top[0]=='World' and top[1] not in ('SandScene',)):
                    box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    mid=np.round((np.array(box.GetMin())+np.array(box.GetMax()))/2,2).tolist() if not box.IsEmpty() else None
                    print('SITE_SIM_PRIM_AFTER_SETTLE',path,mid,flush=True)
        except Exception as error:print('SITE_SIM_PRIM_AFTER_SETTLE scan failed',error,flush=True)
        try:   # 경기장 USD 안에 로버·바퀴 이름을 가진 prim이 있는지(복사본 후보) 진단한다.
            keywords=('rover','wheel','chassis','base_link','robot','rocker')
            for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/SandScene'),Usd.TraverseInstanceProxies()):
                if any(k in prim.GetName().lower() for k in keywords):
                    print('SITE_SIM_ARENA_NAME',prim.GetPath(),prim.GetTypeName(),flush=True)
            print('SITE_SIM_ARENA_PRIM_COUNT',sum(1 for _ in Usd.PrimRange(stage.GetPrimAtPath('/World/SandScene'),Usd.TraverseInstanceProxies())),flush=True)
        except Exception as error:print('SITE_SIM_ARENA_NAME scan failed',error,flush=True)
        atomic_json(Path(args.output)/'ready.json',{'position_m':p.tolist(),'wall_s':time.monotonic()-wall_start})
        while app.is_running():
            sim_s=float(world.current_time)-t0;wall_s=time.monotonic()-wall_start
            if sim_s>=args.max_sim_s or wall_s>=args.max_wall_s:status='timeout';reason='sim_or_wall_limit';break
            if observation_worker is not None:
                observation=observation_worker.poll()
                if observation is not None:drive.runtime.observe_depth(observation)
            if sim_s+1e-9>=next_control:
                row=drive.step(sim_s);row['wall_s']=wall_s
                if region_mode:row['region_observation']=region_observation_metadata(drive.runtime.observation,sim_s)
                if soil is not None:row['soil']=soil.record.copy()
                if estimator is not None:row['estimator_frame']=frame_info
                if estimator is not None:
                    estimator_log.write({'kind':'pose','stamp_s':sim_s,
                        'position_m':row['position_m'],'quaternion_wxyz':row['quaternion_wxyz'],
                        'estimated_position_m':row['estimated_position_m']})
                row['mission']['camera_error']=last_camera_error
                recorder.observe(row,drive.runtime.events);next_control+=control_dt
                if sim_s>=5 and row['mission']['reason'] in ('invalid_pose','pose_jump','unsafe_observed_motion','unsafe_local_segment'):
                    status='failed';reason=row['mission']['reason'];break
                if row['mission']['state'] in ('complete','failed'):
                    status=row['mission']['state'];reason=row['mission']['reason'];break
                p=np.asarray(row['position_m']);q=np.asarray(row['quaternion_wxyz'])
                if p[2]<ground-.1 or 1-2*(q[1]**2+q[2]**2)<math.cos(math.radians(70)):
                    status='failed';reason='fallen_or_tipped';break
            stereo_due=stereo_writer is not None and sim_s+1e-9>=next_stereo
            render_due=stereo_due or time.monotonic()>=next_render
            if actuator is not None:actuator.apply()
            if soil is not None:soil.apply()
            world.step(render=render_due)
            if estimator is not None:
                gp,gq=robot.get_world_pose()
                sample=estimator.on_step(float(world.current_time)-t0,gp,gq)
                if sample is not None:
                    start,end,gyro,force=sample
                    if region_mode:drive.runtime.observe_imu(end,float(world.current_time)-t0,True)
                    estimator_log.write({'kind':'imu','stamp_s':end,'dt_s':end-start,
                        'gyro':gyro.tolist(),'force':force.tolist()})
            if render_due:
                next_render=time.monotonic()+.2
                # Chase view from above and behind so the small rover stays visible inside the walls.
                rp=np.asarray(row['position_m'],float)
                overview_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(rp[0],rp[1]-.35,rp[2]+2.6),
                    Gf.Vec3d(rp[0],rp[1],rp[2]),Gf.Vec3d(0,0,1)).GetInverse())
                for key,(_,annotator) in cameras.items():
                    try:
                        rgba=np.asarray(annotator.get_data())
                        if rgba.ndim!=3 or rgba.shape[2]<3:continue
                        path=recorder.live/f'{key}.jpg';tmp=path.with_suffix('.tmp')
                        Image.fromarray(rgba[:,:,:3]).save(tmp,format='JPEG',quality=82);tmp.replace(path)
                    except Exception as error:last_camera_error=str(error)
                if getattr(args,'record_images',False) and getattr(args,'save_overview_s',0.)>0 and sim_s>=next_overview:
                    next_overview=sim_s+args.save_overview_s
                    try:
                        rgba=np.asarray(cameras['overview'][1].get_data())
                        if rgba.ndim==3 and rgba.shape[2]>=3:
                            (Path(args.output)/'overview_frames').mkdir(exist_ok=True)
                            Image.fromarray(rgba[:,:,:3]).save(Path(args.output)/'overview_frames'/f'{sim_s:07.2f}.jpg',format='JPEG',quality=88)
                    except Exception as error:last_camera_error=str(error)
                if getattr(args,'record_images',False) and args.save_front_s>0 and sim_s>=next_front:
                    next_front=sim_s+args.save_front_s
                    try:
                        rgba=np.asarray(cameras['front'][1].get_data())
                        if rgba.ndim==3 and rgba.shape[2]>=3:
                            (Path(args.output)/'front_frames').mkdir(exist_ok=True)
                            Image.fromarray(rgba[:,:,:3]).save(Path(args.output)/'front_frames'/f'{sim_s:07.2f}.png')
                    except Exception as error:last_camera_error=str(error)
                if stereo_due:
                    left=np.asarray(cameras['front'][1].get_data());right=np.asarray(cameras['right'][1].get_data())
                    if left.ndim!=3 or right.ndim!=3:raise RuntimeError('stereo annotators not ready')
                    rp,rq=robot.get_world_pose()
                    stamp_ns=round((float(world.current_time)-t0)*1e9)
                    names=stereo_writer.write(stamp_ns,left[:,:,:3],right[:,:,:3],rp.tolist(),rq.tolist())
                    if not rtx_capture_logged:print('SITE_SIM_RTX capture',{k:rtx.get(k) for k in rtx_keys},flush=True);rtx_capture_logged=True
                    next_stereo+=1./stereo_hz
                    if orb is not None:
                        result=orb.track(stamp_ns,*names)
                        info=estimator.on_frame(stamp_ns*1e-9,result['state'],result['map_id'],result['world_from_camera'])
                        frame_info={'stamp_s':stamp_ns*1e-9,'orb_state':result['state'],'map_id':result['map_id'],
                                    'keypoints':result['keypoints'],'accepted':bool(info.get('accepted')),
                                    'reason':info.get('reason'),'nis':info.get('nis'),'residual':info.get('residual'),
                                    'position_scale':info.get('position_scale'),'rotation_scale':info.get('rotation_scale'),
                                    'world_from_camera':result['world_from_camera'].tolist(),
                                    'filter_time_s':estimator.filter_time,
                                    'velocity_imu_world_m_s':estimator.filter.velocity.tolist(),
                                    'gyro_bias_rad_s':estimator.filter.gyro_bias.tolist(),
                                    'accel_bias_m_s2':estimator.filter.accel_bias.tolist()}
                        estimator_log.write({'kind':'frame',**frame_info})
                        if wall_track is not None:
                            wall_track_count+=1
                            if wall_track_count%3==2:
                                from scipy.spatial.transform import Rotation as _R
                                body=estimator.filter.pose@estimator.imu_from_base
                                roll,pitch,yaw_est=_R.from_matrix(body[:3,:3]).as_euler('xyz')
                                rate=float((estimator.base_from_imu[:3,:3]@(estimator.filter_gyro-estimator.filter.gyro_bias))[2])
                                if runtime is not None and runtime.stage.startswith('return') and runtime.state == 'descending':
                                    m=wall_track.measure(left[:,:,:3],body[:2,3].copy(),float(yaw_est),float(roll),float(pitch),rate)
                                else:m=dict(reason='stage',lateral_delta=None,front_delta=None,axis=None,lateral_err=None,front_d=None,width=None)
                                if runtime is not None and runtime.corner_gate:
                                    cf=wall_track.front_only(left[:,:,:3],float(yaw_est),float(roll),float(pitch),rate) if runtime.stage in runtime.CORNER_GATE and runtime.state == 'descending' else None
                                    track_corner=(track_corner+[cf['d']])[-3:] if cf is not None and cf['d'] is not None else []
                                    runtime.front_obs=dict(stamp=stamp_ns*1e-9,d=float(np.median(track_corner))) if len(track_corner)==3 and max(track_corner)-min(track_corner)<=.04 else None
                                track_lat=(track_lat+[m['lateral_delta']])[-3:] if m['lateral_delta'] is not None else (track_lat[-3:] if not m['reason'] else [])
                                track_front=(track_front+[m['front_delta']])[-3:] if m['front_delta'] is not None else (track_front[-3:] if not m['reason'] else [])
                                if m['axis'] is not None:
                                    heading=np.array([math.cos(m['axis']),math.sin(m['axis'])]);lateral=np.array([-heading[1],heading[0]])
                                    offset=runtime.plan_offset.copy()
                                    for history,spread,axis_vector in ((track_lat,.04,lateral),(track_front,.03,heading)):
                                        if len(history)==3 and max(history)-min(history)<=spread:
                                            current=float(offset@axis_vector);target=float(np.median(history))
                                            mismatch=target-current
                                            trust=math.exp(-(mismatch/.05)**2)     # 계획 위치가 벽 측정과 어긋날수록 추정을 덜 믿는다
                                            offset=offset+float(np.clip((1.-trust)*mismatch,-.12,.12))*axis_vector     # 측정 축 성분만, 어긋남에 비례해 측정 쪽으로
                                    if float(np.linalg.norm(offset))>.4:offset=offset*(.4/float(np.linalg.norm(offset)))
                                    runtime.plan_offset=offset
                                estimator_log.write({'kind':'wall_plan','stamp_s':stamp_ns*1e-9,'stage':None if runtime is None else runtime.stage,'reason':m['reason'],
                                                     'lateral_delta':m['lateral_delta'],'front_delta':m['front_delta'],'offset':runtime.plan_offset.tolist(),
                                                     'est_xy':body[:2,3].tolist(),'gt_xy':[float(rp[0]),float(rp[1])]})
                        if wall_yaw is not None:
                            wall_yaw_count+=1
                            if wall_yaw_count%3==1:
                                from scipy.spatial.transform import Rotation as _R
                                body=estimator.filter.pose@estimator.imu_from_base
                                roll,pitch,yaw_est=_R.from_matrix(body[:3,:3]).as_euler('xyz')
                                rate=float((estimator.base_from_imu[:3,:3]@(estimator.filter_gyro-estimator.filter.gyro_bias))[2])
                                measurement=wall_yaw.measure(left[:,:,:3],stamp_ns*1e-9,float(yaw_est),float(roll),float(pitch),rate,body[:2,3])
                                if measurement['valid']:
                                    update=estimator.on_wall_yaw(measurement['yaw'],wall_yaw.sigma)
                                    measurement.update(accepted=bool(update['accepted']),innovation_rad=update['innovation_rad'],nis=update['nis'])
                                estimator_log.write({'kind':'wall_yaw','stamp_s':stamp_ns*1e-9,'yaw_est':float(yaw_est),'yaw_sigma_deg':float(np.degrees(np.sqrt(estimator.filter.covariance[8,8]))),**measurement})
                        if observer is not None and stamp_ns*1e-9>=next_observation:
                            observation=observer.update(left[:,:,:3],right[:,:,:3],
                                estimator.filter.pose@estimator.imu_from_base,stamp_ns*1e-9)
                            drive.runtime.observe_depth(observation)
                            next_observation=stamp_ns*1e-9+1./observation_hz
                        elif observation_worker is not None and stamp_ns*1e-9>=next_observation:
                            observation_worker.submit(names[0].read_bytes(),names[1].read_bytes() if observation_mode=='depth' else b'',
                                                      estimator.filter.pose@estimator.imu_from_base,stamp_ns*1e-9)
                            next_observation=next_observation_stamp(stamp_ns*1e-9,observation_hz)
                recorder.publish()
    except KeyboardInterrupt as error:status='interrupted';reason=str(error)
    except Exception as error:
        reason=f'{type(error).__name__}: {error}'
        import traceback
        traceback.print_exc()
    finally:
        if observation_worker is not None:observation_worker.close()
        if stereo_writer is not None:stereo_writer.close()
        if estimator_log is not None:estimator_log.close()
        if drive is not None:
            try:
                drive.stop()
                atomic_json(Path(args.output)/'stop.json',{'wheel_velocity_targets_rad_s':[0.]*4,'applied':True})
                if world is not None:
                    if parity_physics:
                        from experiments.site_sim.physics_sync import settle_stopped
                        settle_stopped(world,actuator,soil,20)
                    else:
                        for _ in range(20):world.step(render=False)
            except Exception as error:status='failed';reason+='; stop failed: '+str(error)
        try:
            summary=recorder.finalize(status,reason)
        except Exception as error:
            raise RuntimeError(f'{reason}; finalization failed: {error}') from error
        finally:
            signal.signal(signal.SIGTERM,previous_signal)
            if app is not None:app.close()
    print('SITE_SIM_FINISHED',json.dumps(summary),flush=True)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser()
    for name in ('mission','urdf','geometry','scene-usd','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-sim-s',type=float,default=900.);parser.add_argument('--max-wall-s',type=float,default=3600.)
    parser.add_argument('--sand-texture',type=Path,default=None)
    parser.add_argument('--save-front-s',type=float,default=0.)
    parser.add_argument('--record-images',action='store_true',help='영상 원본·재생 프레임 보관(기본 OFF, ORB용 임시 두 장과 live 추적뷰만 유지)')
    parser.add_argument('--parity-physics',action='store_true',help='원래 바퀴 형상에서 Claude 정합의 solver 마찰·쿨롱 마찰·고정 토양 계수 적용')
    parser.add_argument('--save-overview-s',type=float,default=0.)
    parser.add_argument('--stanley-gain',type=float,default=0.)
    parser.add_argument('--follower',choices=('unicycle','blend'),default=None,help='경로 추종기: unicycle(기본, 없으면 같음)·blend(yaw+vy 혼합)')
    parser.add_argument('--blend-max-yaw-rate',type=float,default=None,help='blend 추종기의 각속도 상한 [rad/s] (기본 0.09)')
    parser.add_argument('--wheel-gain-left',type=float,default=1.);parser.add_argument('--wheel-gain-right',type=float,default=1.)
    parser.add_argument('--slip-start-s',type=float,default=0.);parser.add_argument('--slip-end-s',type=float,default=None)
    parser.add_argument('--stereo-hz',type=float,default=0.)
    parser.add_argument('--lighting-profile',choices=('legacy','diffuse'),default='diffuse',help='diffuse: 강한 환경광·약한 태양광, legacy: 기존 광원 ×0.1')
    parser.add_argument('--wall-material',choices=('black','legacy'),default='black',help='black: 무광 검정 벽(기본), legacy: 원본 벽 재질')
    parser.add_argument('--start-hold-s',type=float,default=0.)
    parser.add_argument('--navigation',choices=('path','region-mppi'),default='path')
    parser.add_argument('--region-observation',choices=('depth-sync','depth','pixel','none'),default='depth-sync')
    parser.add_argument('--region-observation-hz',type=float,default=None)
    parser.add_argument('--mppi-max-speed',type=float,default=.15,help='구역 MPPI 최대 전진/후퇴 속도 [m/s], 평지·등판 공통 (기본 0.15)')
    parser.add_argument('--actuator-model',choices=('legacy','ksrc-pi'),default='legacy',help='시뮬 하위 구동: 기존 joint drive 또는 KSRC 서보·C PI·게이트')
    parser.add_argument('--soil-model',choices=('fixed','ksrc-legacy'),default='fixed',help='고정 접촉 또는 peer 바퀴별 슬립·가상 침하 모델 (--mu는 mu_max)')
    parser.add_argument('--estimator',action='store_true')
    parser.add_argument('--wall-plan',action='store_true',help='벽 수직거리로 구한 가운데·진행방향 위치 보정을 MPPI 계획용 위치에만 반영(추정기·명령은 그대로)')
    parser.add_argument('--mppi-vy',type=float,default=0.,help='MPPI가 낼 수 있는 옆 속도 상한 [m/s] (0이면 vy 없음)')
    parser.add_argument('--mppi-clearance',type=float,default=0.,help='MPPI 벽 여유 비용 가중치 (0이면 꺼짐)')
    parser.add_argument('--mppi-clearance-margin',type=float,default=.12,help='여유 비용이 0이 되는 벽 거리 = 안전 한계 + 이 값 [m]')
    parser.add_argument('--corner-gate',action='store_true',help='모서리(return9·return7 끝)에서 앞벽 측정 거리가 지도 기대값에 가까울 때만 다음 단계로 넘어간다')
    parser.add_argument('--wall-yaw',action='store_true',help='영상 앞 벽선과 지도 복도축으로 절대 yaw를 ESKF에 융합')
    parser.add_argument('--wall-yaw-sigma-deg',type=float,default=5.)
    parser.add_argument('--adaptive-visual-noise',action='store_true',help='실험적 경사·회전별 영상 noise 배율(기본 OFF)')
    parser.add_argument('--mu',type=float,default=None,help='지면·바퀴 마찰계수(정지=동). 없으면 기본 재질')
    parser.add_argument('--wheel-effort',type=float,default=None,help='바퀴 최대 토크 [N m] (기본 DriveSession 1.0, 실기 데이터시트 정지 토크 약 1.96)')
    parser.add_argument('--command-delay-ms',type=float,nargs=2,default=None,metavar=('LO','HI'),help='명령→조인트 적용 지연 범위 [ms] (실기 17.6~46.5)')
    parser.add_argument('--delay-seed',type=int,default=0)
    parser.add_argument('--imu-drop-prob',type=float,default=0.,help='IMU 패킷 누락 확률(--imu-realism 필요, 실측 0.8%%)')
    parser.add_argument('--start-noise-xy-m',type=float,default=0.,help='로버 시작 위치 오차 σ [m] (추정기는 이상적 시작을 믿음)')
    parser.add_argument('--start-noise-yaw-deg',type=float,default=0.)
    parser.add_argument('--start-seed',type=int,default=0)
    parser.add_argument('--control-dt',type=float,default=.05,help='제어·IMU 구간 주기 [s] (기본 0.05, RL은 0.02)')
    parser.add_argument('--rl-policy',default=None,help='KSRC Res8 정책 경로(확장자 없이). 있으면 swerve 목표 아래에 잔차를 얹는다(기본 OFF)')
    parser.add_argument('--ksrc-root',default=None,help='ksrc 체크아웃 경로(pi/policy/student_runtime.py, sim/libksrc_common.so 필요)')
    parser.add_argument('--imu-realism',action='store_true',help='실기(모래 15° 램프 student v3 로그)에서 잰 IMU 바이어스·잡음·이동 진동을 더함(기본 OFF)')
    parser.add_argument('--imu-seed',type=int,default=0)
    parser.add_argument('--imu-profile',choices=('nominal','stress'),default='nominal',help='--imu-realism의 정지 백색 잡음: nominal=맑은 정지 실측, stress=이전 모래 구간 값(가속도 약 40배)')
    parser.add_argument('--imu-vibration-scale',type=float,default=1.,help='이동 진동 배율')
    parser.add_argument('--visual-turn-scale',type=float,default=None,help='회전 중 영상 관측 σ 배율(기본 4, --adaptive-visual-noise 필요)')
    parser.add_argument('--filter-absolute-attitude-sigma-deg',type=float,default=None,help='지정하면 ORB 절대 자세를 이 σ[deg]로 필터에 느슨하게 묶는다(시뮬 평가용, 기본 꺼짐)')
    parser.add_argument('--filter-acc-noise',type=float,default=None,help='필터가 가정하는 IMU 가속도 잡음 IMU.NoiseAcc [m/s²/√Hz] (기본 0.002)')
    parser.add_argument('--filter-gyro-bias-init-sigma-dps',type=float,default=None,help='지정하면 start_hold_s 정지 구간의 gyro 평균으로 필터 gyro bias를 초기화하고 z bias σ 를 이 값 [dps]으로 줄인다(기본 꺼짐)')
    parser.add_argument('--filter-gyro-noise-scale',type=float,default=None,help='필터가 가정하는 IMU 자이로 잡음 IMU.NoiseGyro 의 배율 (기본 1, 예: 20)')
    parser.add_argument('--velocity-direction-sigma',type=float,default=None,help='조향·바퀴 측정 이동 방향(크기는 안 씀)을 속도 방향 관측으로 쓰는 σ [m/s] (기본 꺼짐)')
    parser.add_argument('--rtx-aa-op',type=int,default=None,help='렌더러 안티앨리어싱 /rtx/post/aa/op (0 끔·1 TAA·2 FXAA·3 DLSS·4 RTXAA, 기본 변경 없음)')
    parser.add_argument('--dlss-exec-mode',type=int,default=2,help='DLSS 모드 /rtx/post/dlss/execMode (0 Performance·1 Balanced·2 Quality·3 Auto, 기본 2=Quality; 0이면 예전 Isaac 기본 Performance)')
    parser.add_argument('--orb-io',type=Path,default=Path('/workspace/orbio'))
    parser.add_argument('--fault-injection',action='store_true');args=parser.parse_args();code=1
    try:
        summary=run(args);code=0 if summary['status']=='complete' else 2
    finally:
        Path(args.output).parent.joinpath('exit.txt').write_text(str(code)+'\n')
    raise SystemExit(code)


if __name__=='__main__':main()
