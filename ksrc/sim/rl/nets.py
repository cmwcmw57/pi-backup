"""Teacher-student (RMA 식) 신경망.

왜 행동 증류가 아니라 잠재변수 증류인가 --
student 가 teacher 의 행동을 직접 흉내내게 하면 "무엇을 추론해야 하는지" 에 대한
감독 신호가 없다.  여기 특권정보의 주축은 **흙 상태**(mu_max, 두께, 침하계수)이고
에피소드 안에서 거의 변하지 않는 환경 파라미터다.  그걸 저차원 z 로 압축해서
student 가 z 를 맞히게 하면 회귀 문제가 훨씬 잘 조건화된다.  덤으로 z 를 찍어보면
정책이 흙을 어떻게 보고 있는지 눈으로 확인할 수 있다.

구조:
    teacher :  encoder(특권 43) -> z(8)
               정책( 고유수용감각 , z ) -> 행동 8
    student :  adapt(고유수용감각) -> z_hat(8)
               정책( 고유수용감각 , z_hat ) -> 행동 8      <- 정책은 그대로 재사용

student 는 정책 가중치를 건드리지 않는다.  adapt 만 학습하면 된다.
"""
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import OBS_PER_FRAME, HISTORY, PRIV_DIM

PROPRIO_DIM = OBS_PER_FRAME * HISTORY
LATENT_DIM = 8


def _mlp(sizes, act=nn.ELU, out_act=None):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    layers = layers[:-1]
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class TeacherExtractor(BaseFeaturesExtractor):
    """관측 [고유수용감각 | 특권] 을 [고유수용감각 | z] 로 바꾼다."""

    def __init__(self, observation_space: spaces.Box, latent_dim: int = LATENT_DIM):
        proprio_dim = observation_space.shape[0] - PRIV_DIM
        assert proprio_dim > 0, \
            f"teacher 관측은 고유수용감각 + {PRIV_DIM}D 특권정보여야 한다"
        super().__init__(observation_space, proprio_dim + latent_dim)
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim
        # tanh 로 z 를 [-1,1] 에 묶는다. student 회귀가 안정된다.
        self.encoder = _mlp([PRIV_DIM, 64, 64, latent_dim], out_act=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio, priv = obs[:, :self.proprio_dim], obs[:, self.proprio_dim:]
        return torch.cat([proprio, self.encoder(priv)], dim=1)


class AdaptationModule(nn.Module):
    """고유수용감각 이력 -> z_hat.

    이력을 (프레임, 채널) 로 놓고 시간축 1D 합성곱을 먼저 태운다.  슬립은 한
    프레임으로는 안 보이고 "명령은 큰데 바퀴 속도가 안 붙는" **시간 패턴**이라,
    채널별 시간 필터가 MLP 보다 사전지식에 맞는다.
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(OBS_PER_FRAME, 48, kernel_size=4), nn.ELU(),
            nn.Conv1d(48, 48, kernel_size=3), nn.ELU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, OBS_PER_FRAME, HISTORY)).numel()
        self.head = _mlp([n, 96, latent_dim], out_act=nn.Tanh)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        x = proprio.view(-1, HISTORY, OBS_PER_FRAME).transpose(1, 2)
        return self.head(self.conv(x).flatten(1))


class StudentPolicy(nn.Module):
    """adapt + (동결된) teacher 정책 머리.  이게 실기체에 올라가는 것."""

    def __init__(self, teacher_policy, adapt: AdaptationModule = None):
        super().__init__()
        self.tp = teacher_policy
        self.adapt = adapt or AdaptationModule(teacher_policy.features_extractor.latent_dim)
        for p in self.tp.parameters():
            p.requires_grad_(False)

    def act(self, proprio: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """z 를 주면 그걸 쓰고(teacher), 없으면 adapt 로 추정한다(student)."""
        if z is None:
            z = self.adapt(proprio)
        feat = torch.cat([proprio, z], dim=1)
        latent_pi = self.tp.mlp_extractor.forward_actor(feat)
        return self.tp.action_net(latent_pi)      # 결정론적 행동 = 평균

    @torch.no_grad()
    def predict(self, proprio_np: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(proprio_np, dtype=torch.float32).reshape(1, -1)
        return self.act(x).squeeze(0).cpu().numpy()
