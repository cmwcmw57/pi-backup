# ================================================================
# viewer.py
# 설명: 서버 run의 완성된 관찰 파일만 loopback HTTP로 제공한다. Isaac/ML 의존성은 없다.
# 사용법: python3 -m experiments.site_sim.viewer --output RUN/output --port 8765
# ================================================================
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs,urlsplit


OPTION_LABELS=(('navigation','주행 모드'),('rl_policy','RL'),('imu_realism','실제 IMU'),('imu_drop_prob','IMU 누락'),('command_delay_ms','명령 지연 ms'),('mu','μ'),
               ('wheel_effort','바퀴 토크'),('control_dt','제어 주기 s'),('start_noise_xy_m','시작 위치 노이즈 m'),('start_noise_yaw_deg','시작 yaw 노이즈 °'),
               ('estimator','추정 위치로 주행'),('adaptive_visual_noise','영상 신뢰도 조정'),('stanley_gain','Stanley'),('follower','추종기'),('blend_max_yaw_rate','blend ω'),('wheel_gain_left','왼쪽 바퀴 배율'))


def run_options_text(output: Path) -> str:
    """run_options.json에서 켜 둔 설정만 한 줄로 만든다(없으면 빈 문자열)."""
    path=output/'run_options.json'
    if not path.is_file():return ''
    options=json.loads(path.read_text());parts=[]
    for key,label in OPTION_LABELS:
        value=options.get(key)
        if value is None or value is False or (type(value) in (int,float) and value==0) or (key=='wheel_gain_left' and value==1.0):continue   # True는 켜짐으로 표시
        parts.append(label if value is True else f'{label} {value}')
    return ' · '.join(parts) if parts else '기본 설정'


def replay_payload(output: Path, limit: int=2500, mission: Path | None=None) -> dict:
    """telemetry.jsonl에서 재생용 시계열(시각, 실제·추정 xy, yaw, 단계, 오차)과 저장된 관찰 화면 시각을 만든다."""
    path=output/'telemetry.jsonl'
    with path.open() as stream:count=sum(1 for line in stream if line.strip())
    if not count:raise ValueError('empty telemetry')
    step=max(1,count//limit)
    # MPPI 후보를 포함한 긴 기록도 전체 JSON을 메모리에 펼치지 않는다.
    with path.open() as stream:
        picked=[json.loads(line) for i,line in enumerate(line for line in stream if line.strip())
                if i%step==0 or i==count-1]
    data={'t':[],'gt':[],'est':[],'yaw':[],'state':[],'route':[],'err_cm':[],'frames':[],'front':[]}
    region_replay=any(row.get('mission',{}).get('navigation',{}).get('mode')=='region-mppi' for row in picked)
    if region_replay:data['navigation']=[]
    for row in picked:
        gt=row['position_m'][:2];est=row.get('estimated_position_m',row['position_m'])[:2];w=row['quaternion_wxyz']
        data['t'].append(round(row['stamp_s'],2));data['gt'].append([round(gt[0],4),round(gt[1],4)]);data['est'].append([round(est[0],4),round(est[1],4)])
        data['yaw'].append(round(math.atan2(2*(w[0]*w[3]+w[1]*w[2]),1-2*(w[2]**2+w[3]**2)),4))
        phase=row.get('mission',{});data['state'].append(phase.get('state'));data['route'].append(phase.get('route'))
        data['err_cm'].append(round(math.dist(gt,est)*100,2))
        if region_replay:
            data['navigation'].append({k:v for k,v in phase.get('navigation',{}).items()
                                       if k not in ('candidate_trajectories','regions','portals')})
    data['options']=run_options_text(output)
    if mission is not None:   # 벽·단차까지 거리(정답 위치 기준). 임무 폴더가 있을 때만(노트북 뷰어)
        import numpy as np
        from experiments.handheld_mapping.site_mission import SiteMission
        from experiments.site_sim.clearance import clearance_map
        site=SiteMission.load(mission,load_mesh=False).site;distance=clearance_map(site.blocked,site.resolution)
        cells=site.cells(np.array(data['gt']));inside=(cells[:,0]>=0)&(cells[:,0]<distance.shape[1])&(cells[:,1]>=0)&(cells[:,1]<distance.shape[0])
        values=np.zeros(len(cells));values[inside]=distance[cells[inside,1],cells[inside,0]]
        data['clear_cm']=[round(float(min(v,9.99))*100,1) for v in values]
    frames=output/'overview_frames'
    if frames.is_dir():data['frames']=sorted(float(p.stem) for p in frames.glob('*.jpg') if p.stem.replace('.','',1).isdigit())
    stereo=output/'stereo/mav0/cam0/data'
    if stereo.is_dir():data['front']=sorted(int(p.stem)/1e9 for p in stereo.glob('*.png') if p.stem.isdigit())
    return data


def serve(output: Path, port: int, mission: Path | None=None) -> ThreadingHTTPServer:
    """허용된 run 파일과 정적 HTML만 제공하는 독립 관찰 서버를 연다."""
    output=Path(output).resolve();html=Path(__file__).with_name('viewer.html').read_bytes()
    files={'/state':('state.json','application/json'),'/map':('map.json','application/json'),
           '/frame/overview.jpg':('overview.jpg','image/jpeg'),'/frame/front.jpg':('front.jpg','image/jpeg')}

    cache={}

    class Handler(BaseHTTPRequestHandler):
        """허용 경로만 읽고 미준비·불완전 파일을 명시적으로 거부한다."""
        def log_message(self,*args) -> None:pass
        def do_GET(self) -> None:
            split=urlsplit(self.path);path=split.path;code=200;mime='text/plain';body=b'not found'
            if path=='/':body=html;mime='text/html; charset=utf-8'
            elif path=='/replay':
                try:
                    if not (output/'telemetry.jsonl').is_file():code=404
                    else:
                        if 'replay' not in cache:cache['replay']=replay_payload(output,mission=mission)
                        body=json.dumps(cache['replay'],allow_nan=False).encode();mime='application/json'
                except (OSError,ValueError,KeyError,TypeError):code=503;body=b'not ready'
            elif path=='/replay_front':
                try:
                    times=cache['replay']['front'] if 'replay' in cache else replay_payload(output,mission=mission)['front']
                    wanted=float(parse_qs(split.query)['t'][0]);nearest=min(times,key=lambda t:abs(t-wanted))
                    body=(output/'stereo/mav0/cam0/data'/f'{round(nearest*1e9)}.png').read_bytes();mime='image/png'
                except (KeyError,ValueError,IndexError):code=400;body=b'bad request'
                except OSError:code=404
            elif path=='/replay_frame':
                try:
                    times=cache['replay']['frames'] if 'replay' in cache else replay_payload(output,mission=mission)['frames']
                    wanted=float(parse_qs(split.query)['t'][0]);nearest=min(times,key=lambda t:abs(t-wanted))
                    body=(output/'overview_frames'/f'{nearest:07.2f}.jpg').read_bytes();mime='image/jpeg'
                except (KeyError,ValueError,IndexError):code=400;body=b'bad request'
                except OSError:code=404
            elif path in files:
                relative,mime=files[path]
                try:
                    body=(output/'live'/relative).read_bytes()
                    if mime=='application/json':
                        data=json.loads(body)
                        if path=='/state':
                            data['age_s']=max(0.,time.time()-data['updated_wall_s'])
                            data['stale']=data.get('status')=='running' and data['age_s']>3.
                            data['image_age_s']={key:(max(0.,time.time()-(output/'live'/f'{key}.jpg').stat().st_mtime)
                                if (output/'live'/f'{key}.jpg').is_file() else None) for key in ('overview','front')}
                            body=json.dumps(data,allow_nan=False).encode()
                except (OSError,ValueError,KeyError,TypeError):code=503;mime='text/plain';body=b'not ready'
            else:code=404
            self.send_response(code);self.send_header('Content-Type',mime)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass

    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--mission',type=Path,default=None,help='벽 거리 표시용 임무 폴더(없으면 output/mission_bundle을 찾는다)')
    args=parser.parse_args()
    mission=args.mission if args.mission else (args.output/'mission_bundle' if (args.output/'mission_bundle'/'mission.json').is_file() else None)
    server=serve(args.output,args.port,mission);print(f'http://127.0.0.1:{server.server_port}',flush=True)
    try:
        while True:time.sleep(.5)
    except KeyboardInterrupt:pass
    finally:server.shutdown();server.server_close()


if __name__=='__main__':main()
