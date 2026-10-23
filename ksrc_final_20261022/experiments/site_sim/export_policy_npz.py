# ================================================================
# export_policy_npz.py
# 설명: KSRC Res8 student TorchScript(<p>.pt)의 정규화기와 MLP 가중치를 numpy .npz로 꺼낸다. Isaac 컨테이너에는 torch가 없어서 시뮬은 이 파일을 numpy로 추론한다.
# 사용법: python -m experiments.site_sim.export_policy_npz /path/to/res8_mlp_v1_rc4000   (torch 필요, <p>.npz를 같은 폴더에 쓴다)
# ================================================================
import sys
from pathlib import Path

import numpy as np


def export(policy: Path, samples: int = 200, seed: int = 0) -> float:
    """가중치를 저장하고 torch 출력과 numpy 출력의 최대 절대 차이를 돌려준다."""
    import torch
    from experiments.site_sim.rl_residual import NumpyMlpPolicy
    module = torch.jit.load(str(policy)+'.pt', map_location='cpu').eval()
    state = {k: v.numpy() for k, v in module.state_dict().items()}
    layers = sorted({int(k.split('.')[1]) for k in state if k.startswith('student.') and k.endswith('.weight')})
    eps = float(module.norm.code_with_constants[1].const_mapping['c0'])
    arrays = {'mean': state['norm._mean'][0], 'std': state['norm._std'][0], 'eps': np.array(eps)}
    for n, index in enumerate(layers):
        arrays[f'W{n}'], arrays[f'b{n}'] = state[f'student.{index}.weight'], state[f'student.{index}.bias']
    np.savez(str(policy)+'.npz', **arrays)
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((samples, arrays['mean'].size)).astype(np.float32)
    numpy_policy = NumpyMlpPolicy.load(Path(str(policy)+'.npz'))
    with torch.no_grad():
        expected = module(torch.as_tensor(obs)).numpy()
    return float(max(np.max(np.abs(numpy_policy(obs[i:i+1])-expected[i])) for i in range(samples)))


if __name__ == '__main__':
    print('max |numpy - torch| =', export(Path(sys.argv[1])))
