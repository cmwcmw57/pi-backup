"""CAD 실측 rover_description/mjcf/rover.xml 에 지형·액추에이터·센서를 주입한다.

**왜 생성에서 로드로 바뀌었나 (2026-10-05)**
예전에는 config.py 의 치수로 원시도형(박스/실린더) 모델을 **생성**했다.  그런데
CAD 실측본(ros/rover_description)과 대조하니 치수 몇 개가 아니라 조립본 자체가
달랐다:
    바퀴 반경   39.95 -> 47.503 mm   (+18.9%, v_max 0.318 -> 0.378 m/s)
    조향축 x    105.0 -> 102.431 mm
    바퀴중심 y  조향축 기준 0 -> 9.53 mm 바깥쪽
    로커->바퀴 수직 108.43 -> 114.313 mm
    배터리      VEGA 650mAh 60g -> Zippy2800 178.1g
    서보        조향모듈 -> **로커 링크**
    Pi5 71.2g / D435 72.0g 실측 추가
config.py 의 파라메트릭 질량 모델(PETG 밀도 x 인필 x 부피)로는 이걸 재현할 수
없고, 부품별 실측 관성 텐서는 더더욱 못 만든다.  그래서 **질량·관성·충돌·시각·
관절 기하는 CAD 를 단일 진실 원천으로 삼고**, 시뮬 전용 요소만 주입한다.

rover.xml 이 이 용도로 알맞은 이유:
  - 바퀴 충돌이 **실린더** (r 47.503, 반폭 18.6mm) -> terramech 의 슬립/침하
    모델이 그대로 붙는다 (geom_type == mjGEOM_CYLINDER 를 찾는다).
  - 차체 충돌이 **박스 분해** (base 8, 로커 3, 조향 2) -> 메시 충돌 비용 없음.
    config.py 의 leg_segs 수작업 단면 분해를 CAD 분해가 대체한다.
  - 디퍼렌셜 equality(rocker 1:1 역방향)가 이미 있다.
  - 속도 실측 39,282 step/s (실시간 78.6배) -- 병목이 아니다.

**주입/덮어쓰는 것과 그 이유**
  지형          hfield/plane + terrain class.  CAD 에는 무한 평면만 있다.
  조향 가동범위 CAD 의 +-90 도는 **가정값**이다 (model.json/assumptions:
                "replace with actual wire/mechanical limits").  STEP 실측으로
                정한 config.steer_*_deg (+-90) 를 쓴다.
  바퀴 액추에이터 CAD 는 velocity(kv 0.2, forcerange +-1).  우리는 motor(토크,
                gear = mot_tau_stall 1.96) + 관절감쇠로 **DC 모터 속도 드룹**을
                모델링한다.  env._reward 의 토크/전력 계산
                (tau = duty*gear - wheel_damping*omega) 이 이 모델에 의존한다.
  센서          s_vel(velocimeter) 과 관절별 jointpos/jointvel 추가.
  이름          env.py / terramech.py 가 쓰는 기존 이름으로 **바꾼다**.  반대로
                env 를 고치지 않는 이유: rover_description 은 ROS 패키지 원본으로
                두고, 매핑 지점을 한 곳(RENAME)에 모으기 위해서다.

좌표계는 펌웨어와 동일: x=전진, y=좌측, z=위.
"""
import pathlib
import xml.etree.ElementTree as ET

from config import RoverCfg

CORNERS = [  # (이름, 전후부호, 좌우부호).  기구학 코드가 참조한다.
    ("fl", +1, +1), ("rl", -1, +1),
    ("fr", +1, -1), ("rr", -1, -1),
]

# ros/rover_description = CAD 실측 단일 진실 원천
PKG = (pathlib.Path(__file__).resolve().parents[2] / "ros" / "rover_description")
ROVER_XML = PKG / "mjcf" / "rover.xml"
MESH_DIR = PKG / "meshes"

# CAD 이름 -> 기존 sim/rl 이름.  env.py(st_/wh_/a_st_/a_wh_/s_quat/s_gyro/s_acc/
# root/ground) 와 terramech.py(wheel_/steer_/wh_) 가 쓰는 이름에 맞춘다.
RENAME = {
    "floating_base": "root",
    "base_link": "chassis",
    "left_rocker_joint": "rock_l", "right_rocker_joint": "rock_r",
    "left_rocker_link": "rocker_l", "right_rocker_link": "rocker_r",
    "imu_site": "imu",
    "base_orientation": "s_quat", "imu_gyro": "s_gyro",
    "imu_accelerometer": "s_acc",
}
for _w, _, _ in CORNERS:
    RENAME[f"{_w}_steering_joint"] = f"st_{_w}"
    RENAME[f"{_w}_wheel_joint"] = f"wh_{_w}"
    RENAME[f"{_w}_steering_link"] = f"steer_{_w}"
    RENAME[f"{_w}_wheel_link"] = f"wheel_{_w}"

# 이름이 들어가는 속성 전부.  하나라도 빠지면 참조가 끊겨 컴파일이 실패한다.
_NAME_ATTRS = ("name", "joint", "joint1", "joint2", "body", "body1", "body2",
               "site", "objname", "childclass", "class")


def _rename_tree(el):
    for a in _NAME_ATTRS:
        v = el.get(a)
        if v in RENAME:
            el.set(a, RENAME[v])
    for ch in el:
        _rename_tree(ch)


def _find_joint(root, name):
    for b in root.iter("body"):
        for j in b.findall("joint"):
            if j.get("name") == name:
                return j
    return None


def build(c: RoverCfg, terrain: str = "plane", hfield_n: int = 384,
          hfield_size=(1.5, 2.5), hfield_z: float = 1.0,
          visual: bool = False) -> str:
    """terrain="hfield" 면 빈 높이맵을 만들어 둔다.  실제 데이터는 런타임에
    m.hfield_data 로 덮어쓴다 -- 리셋마다 XML 을 다시 컴파일할 필요가 없다.

    visual=False 면 **시각 전용 메시를 전부 뺀다** (contype=0 인 mesh geom 과
    그 asset).  충돌은 박스/실린더가 담당하므로 물리에 전혀 영향이 없다.
    실측: 메시를 포함하면 vertex 219,134 / face 398,710 이고 워커당 RSS
    512MB -> 16 워커면 8.0GB 다 (가용 11GB).  SubprocVecEnv 가 워커 0 에서
    ConnectionResetError 로 죽은 원인이 이것이다.  학습은 렌더링을 하지
    않으므로 기본을 False 로 둔다.  view.py 등 보는 쪽에서만 True.
    """
    if not ROVER_XML.is_file():
        raise FileNotFoundError(
            f"CAD 모델이 없습니다: {ROVER_XML}\n"
            "ros/rover_description 패키지가 있어야 합니다 (CAD 실측 원천).")
    tree = ET.parse(ROVER_XML)
    root = tree.getroot()
    root.set("model", "ksrc_swerve_rover")
    _rename_tree(root)

    # ---- compiler: 메시 경로를 절대경로로 (from_xml_string 은 상대경로를 못 푼다)
    comp = root.find("compiler")
    comp.set("meshdir", str(MESH_DIR))
    comp.set("angle", "radian")
    comp.set("autolimits", "true")
    comp.set("inertiafromgeom", "false")   # CAD 관성 텐서를 쓴다

    # ---- option: 마찰 원뿔.  terramech 가 종/횡 마찰을 따로 쓰므로 elliptic.
    opt = root.find("option")
    opt.set("timestep", "0.002")
    opt.set("integrator", "implicitfast")
    opt.set("cone", "elliptic")

    # ---- 지형 ----------------------------------------------------------
    world = root.find("worldbody")
    for g in list(world.findall("geom")):
        if g.get("name") == "ground":      # CAD 의 무한 평면을 치운다
            world.remove(g)
    asset = root.find("asset")
    if terrain == "hfield":
        ET.SubElement(asset, "hfield", {
            "name": "arena", "nrow": str(hfield_n), "ncol": str(hfield_n),
            "size": f"{hfield_size[0]} {hfield_size[1]} {hfield_z} 0.5"})
        ground = {"name": "ground", "type": "hfield", "hfield": "arena",
                  "class": "terrain"}
    else:
        ground = {"name": "ground", "type": "plane", "size": "20 20 0.1",
                  "class": "terrain"}
    # 지형/조명은 worldbody 맨 앞에 (body 보다 먼저 와야 XML 스키마가 맞다)
    world.insert(0, ET.Element("geom", ground))
    world.insert(0, ET.Element("light", {
        "pos": "0 0 3", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"}))

    # ---- default: 지형/타이어 클래스 추가 -------------------------------
    dflt = root.find("default")
    terr_d = ET.SubElement(dflt, "default", {"class": "terrain"})
    ET.SubElement(terr_d, "geom", {
        "rgba": "0.55 0.52 0.48 1", "friction": "1.0 0.02 0.002", "condim": "4"})

    # ---- 관절 감쇠/아마추어: DC 모터 드룹과 서보 특성 --------------------
    for w, _, _ in CORNERS:
        j = _find_joint(root, f"wh_{w}")
        # b = tau_stall / w_noload.  이게 곧 DC 모터의 속도 droop 이다.
        j.set("damping", f"{c.wheel_damping:.5f}")
        j.set("armature", "0.0015")
        s = _find_joint(root, f"st_{w}")
        lo, hi = c.steer_range_rad(w)
        # CAD 의 +-90 은 가정값.  STEP 실측 기반 config 값으로 덮는다.
        s.set("range", f"{lo:.5f} {hi:.5f}")
        s.set("limited", "true")
        s.set("damping", "0.02")
        s.set("armature", "0.002")
    for side in ("l", "r"):
        j = _find_joint(root, f"rock_{side}")
        j.set("range", f"{-c.rocker_lim:.4f} {c.rocker_lim:.4f}")
        j.set("damping", "0.05")
        j.set("armature", "0.001")

    # ---- 디퍼렌셜 비: config.diff_ratio 로 덮는다 -----------------------
    eq = root.find("equality").find("joint")
    eq.set("polycoef", f"0 {c.diff_ratio} 0 0 0")
    eq.set("solimp", "0.99 0.999 0.001")

    # ---- 액추에이터: CAD 것을 버리고 config 모델로 ----------------------
    act = root.find("actuator")
    for ch in list(act):
        act.remove(ch)
    for w, _, _ in CORNERS:
        lo, hi = c.steer_range_rad(w)
        ET.SubElement(act, "position", {
            "name": f"a_st_{w}", "joint": f"st_{w}",
            "kp": f"{c.steer_kp}", "kv": f"{c.steer_kv}",
            "ctrlrange": f"{lo:.4f} {hi:.4f}",
            "forcerange": f"{-c.steer_tau} {c.steer_tau}"})
    for w, _, _ in CORNERS:
        # gear = 정지토크.  ctrl in [-1,1] -> 인가토크.  속도 droop 은 joint
        # damping 이 담당한다 (velocity 액추에이터로 바꾸면 env._reward 의
        # tau = duty*gear - damping*omega 계산이 의미를 잃는다).
        ET.SubElement(act, "motor", {
            "name": f"a_wh_{w}", "joint": f"wh_{w}",
            "gear": f"{c.mot_tau_stall}", "ctrlrange": "-1 1",
            "forcerange": f"{-c.mot_tau_stall} {c.mot_tau_stall}"})

    # ---- 시각 전용 메시 제거 (물리 무영향) -------------------------------
    if not visual:
        used = set()
        for b in root.iter("body"):
            for g in list(b.findall("geom")):
                if g.get("type") == "mesh" and g.get("contype") == "0":
                    b.remove(g)
                elif g.get("mesh"):
                    used.add(g.get("mesh"))
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used:
                asset.remove(mesh)

    # ---- 센서 ----------------------------------------------------------
    sen = root.find("sensor")
    # **CAD 의 버그 수정**: rover.xml 의 base_orientation 은
    #     <framequat objtype="body" objname="base_link"/>
    # 인데, MuJoCo 의 objtype="body" 는 바디 프레임이 아니라 **관성 주축
    # 프레임**(xipos/ximat)이다.  CAD 가 전체 관성 텐서를 주므로 주축이 바디
    # 프레임과 돌아가 있고, 실측 결과 평지 정자세에서 roll 47.9도 / pitch
    # -84.3도 가 읽혀 매 리셋마다 tip(>60도) 으로 즉시 종료됐다.
    # 바디 프레임은 objtype="xbody" 다.
    for q in sen.findall("framequat"):
        if q.get("name") == "s_quat":
            q.set("objtype", "xbody")
    ET.SubElement(sen, "velocimeter", {"name": "s_vel", "site": "imu"})
    for w, _, _ in CORNERS:
        ET.SubElement(sen, "jointpos", {"name": f"s_st_{w}", "joint": f"st_{w}"})
        ET.SubElement(sen, "jointvel", {"name": f"s_wh_{w}", "joint": f"wh_{w}"})

    return ET.tostring(root, encoding="unicode")
