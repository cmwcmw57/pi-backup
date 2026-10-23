# ================================================================
# texture_quilt.py
# 설명: 작은 실사 모래 사진 한 장에서 반복 주기가 없는 큰 텍스처를 image quilting으로 만들고,
#       평균 밝기를 유지한 채 대비만 줄인 단계별 사본을 저장한다.
# 사용법:
#   python -m experiments.site_sim.texture_quilt --source color.jpg --output out_dir \
#       --size 8192 6144 --source-px 1024 --levels 1.0 0.5 0.3
# ================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def quilt(source: np.ndarray, width: int, height: int, block: int=128, overlap: int=32,
          tolerance: float=0.1, seed: int=0) -> np.ndarray:
    """source 조각을 겹침 SSD가 비슷한 후보 중 무작위로 골라 width x height로 이어 붙인다."""
    rng=np.random.default_rng(seed)
    n=source.shape[0]
    if source.shape[0]!=source.shape[1] or n<block*2:raise ValueError('square source of at least 2*block required')
    stride=block-overlap
    out=np.zeros((height+block,width+block,3),np.float32)
    luma=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY).astype(np.float32)
    src=source.astype(np.float32);limit=n-block+1
    ramp=np.linspace(0,1,overlap,dtype=np.float32)
    for y0 in range(0,height,stride):
        for x0 in range(0,width,stride):
            if x0==0 and y0==0:
                sy,sx=rng.integers(0,limit,2)
            else:
                cost=np.zeros((limit,limit),np.float32)
                out_luma=cv2.cvtColor(out[y0:y0+block,x0:x0+block],cv2.COLOR_BGR2GRAY)
                if x0>0:
                    cost+=cv2.matchTemplate(luma,np.ascontiguousarray(out_luma[:,:overlap]),cv2.TM_SQDIFF)[:limit,:limit]
                if y0>0:
                    cost+=cv2.matchTemplate(luma,np.ascontiguousarray(out_luma[:overlap,:]),cv2.TM_SQDIFF)[:limit,:limit]
                good=np.argwhere(cost<=cost.min()*(1+tolerance)+1e-6)
                sy,sx=good[rng.integers(len(good))]
            patch=src[sy:sy+block,sx:sx+block].copy()
            if x0>0:
                alpha=ramp[None,:,None];patch[:,:overlap]=alpha*patch[:,:overlap]+(1-alpha)*out[y0:y0+block,x0:x0+overlap]
            if y0>0:
                alpha=ramp[:,None,None];patch[:overlap]=alpha*patch[:overlap]+(1-alpha)*out[y0:y0+overlap,x0:x0+block]
            out[y0:y0+block,x0:x0+block]=patch
    return np.clip(out[:height,:width],0,255).astype(np.uint8)


def set_contrast(image: np.ndarray, level: float) -> np.ndarray:
    """전체 평균을 유지하면서 평균 주변 편차만 level배 한다."""
    mean=image.reshape(-1,3).mean(0)
    return np.clip(mean+level*(image.astype(np.float32)-mean),0,255).astype(np.uint8)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,nargs=2,default=(8192,6144));parser.add_argument('--source-px',type=int,default=1024)
    parser.add_argument('--levels',type=float,nargs='+',default=(1.,.5,.3));parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    image=cv2.imread(str(args.source))
    if image is None:raise SystemExit('cannot read source')
    image=cv2.resize(image,(args.source_px,args.source_px),interpolation=cv2.INTER_AREA)
    args.output.mkdir(parents=True,exist_ok=True)
    big=quilt(image,*args.size,seed=args.seed)
    for level in args.levels:
        cv2.imwrite(str(args.output/f'sand_contrast_{level:g}.jpg'),set_contrast(big,level),[cv2.IMWRITE_JPEG_QUALITY,92])
    print('wrote',len(args.levels),'textures',big.shape)


if __name__=='__main__':main()
