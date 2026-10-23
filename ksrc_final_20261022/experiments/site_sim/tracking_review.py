# ================================================================
# tracking_review.py
# 설명: 저장된 전방 PNG의 인접 영상 ORB seed를 LK 왕복 추적해 표면별 유지율을 비교한다.
# 사용법: .venv-mapping/bin/python -m experiments.site_sim.tracking_review ROOT
# ================================================================
from pathlib import Path
import argparse
import csv
import json
import cv2
import numpy as np


def review(folder: Path, lower_half: bool) -> dict:
    """매 인접쌍에서 재검출한 seed의 왕복 1px 이내 유지율을 집계한다. VO 정확도가 아니다."""
    files=sorted((folder/'front_frames').glob('*.png'),key=lambda p:float(p.stem))
    if len(files)<2:raise ValueError('at least two frames required')
    rows=[];previous=cv2.imread(str(files[0]),0);orb=cv2.ORB_create(nfeatures=2000)
    for a,b in zip(files,files[1:]):
        current=cv2.imread(str(b),0)
        if current is None or previous is None or current.shape!=previous.shape:raise ValueError('invalid image sequence')
        mask=np.zeros_like(previous);h,w=mask.shape;mask[h//2 if lower_half else 10:h-10,10:w-10]=255
        keypoints=orb.detect(previous,mask);n=len(keypoints);kept=0
        if n:
            p=np.float32([k.pt for k in keypoints]).reshape(-1,1,2)
            q,s,_=cv2.calcOpticalFlowPyrLK(previous,current,p,None,winSize=(21,21),maxLevel=3)
            if q is not None:
                back,sb,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None,winSize=(21,21),maxLevel=3)
                if back is not None:
                    xy=q[:,0];valid=(s[:,0]>0)&(sb[:,0]>0)&np.isfinite(xy).all(1)
                    valid&=np.linalg.norm(back[:,0]-p[:,0],axis=1)<=1.
                    valid&=(xy[:,0]>=10)&(xy[:,0]<w-10)&(xy[:,1]>= (h//2 if lower_half else 10))&(xy[:,1]<h-10)
                    kept=int(valid.sum())
        rows.append({'t0_s':float(a.stem),'t1_s':float(b.stem),'dt_s':float(b.stem)-float(a.stem),
                     'seeds':n,'tracked_fb_1px':kept,'retention':kept/n if n else None})
        previous=current
    rates=[r['retention'] for r in rows if r['retention'] is not None]
    return {'frames':len(files),'pairs':len(rows),'pairs_without_features':sum(r['seeds']==0 for r in rows),
            'median_seeds':float(np.median([r['seeds'] for r in rows])),
            'median_retention':float(np.median(rates)) if rates else None,
            'weighted_retention':sum(r['tracked_fb_1px'] for r in rows)/max(1,sum(r['seeds'] for r in rows)),
            'median_dt_s':float(np.median([r['dt_s'] for r in rows])),'rows':rows}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result={'method':'ORB 2000 seeds per previous image; LK forward/backward <=1px; re-detected each pair; no VO pose',
            'limitations':'~1s sparse frames; independent GT runs; lower-half ROI may contain walls; no stereo/no pose accuracy', 'conditions':{}}
    for level in ('none','low','mid','high'):
        result['conditions'][level]={}
        for roi in ('full','lower_half'):
            data=review(args.root/level,roi=='lower_half');rows=data.pop('rows')
            with (args.root/f'{level}_{roi}.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            result['conditions'][level][roi]=data
    (args.root/'tracking_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
