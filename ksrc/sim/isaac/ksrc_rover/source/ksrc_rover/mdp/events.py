"""Episode-reset domain randomization: asymmetric left/right wheel friction.

Wires the teammate's ``apply_wheel_materials`` (the granular-μ surrogate) so the RL
layer faces the target phenomenon -- a left/right traction imbalance the symmetric
firmware IK cannot model. The realized per-wheel μ is stashed on the env for later
logging; it is NEVER exposed as an observation (the real robot cannot measure μ, and
an actor that saw it would be privileged and would not transfer).
"""

from __future__ import annotations

import torch

# Steering-axis y>0 vs y<0 (contract MODULE_XY_M): fl,rl are the left pair; fr,rr the right.
LEFT_WHEEL_BODIES = ["fl_wheel_link", "rl_wheel_link"]
RIGHT_WHEEL_BODIES = ["fr_wheel_link", "rr_wheel_link"]


def randomize_wheel_friction(env, env_ids, mode: str = "wide", master_seed: int = 0, num_buckets: int = 64):
    """Apply per-episode asymmetric left/right wheel μ; record the realized values.

    Deterministic in ``(master_seed, env_id, episode_id)`` so an RL run and a pure-IK
    baseline sharing the same seed see the *same* μ sequence -- that is what makes the
    paired RL-vs-baseline comparison valid.
    """
    from experiments.rover_training_terrains.materials import apply_wheel_materials

    n = env.scene.num_envs
    if not hasattr(env, "_wheel_mu_episode"):
        env._wheel_mu_episode = [0] * n
        env._wheel_mu_realized = [None] * n  # env_id -> {"left": {...}, "right": {...}}

    if env_ids is None:
        ids = list(range(n))
    elif torch.is_tensor(env_ids):
        ids = [int(i) for i in env_ids.tolist()]
    else:
        ids = [int(i) for i in env_ids]

    episode_ids = [env._wheel_mu_episode[i] for i in ids]
    applied = apply_wheel_materials(
        env, ids, asset_name="robot",
        left_body_names=LEFT_WHEEL_BODIES, right_body_names=RIGHT_WHEEL_BODIES,
        mode=mode, master_seed=master_seed, episode_ids=episode_ids, num_buckets=num_buckets,
    )
    for rec in applied:
        env._wheel_mu_realized[rec["env_id"]] = rec["materials"]
        env._wheel_mu_episode[rec["env_id"]] += 1
    return applied


# --- Our own wheel-μ randomizer (stress / curriculum) -----------------------------
# The teammate helper is pair-only (fl=rl, fr=rr), μ∈[0.2,0.6] clipped, reset-only.
# This one adds a configurable range, per-wheel independence and mid-episode resampling
# (mode="interval"), still deterministic in (master_seed, env_id, draw counter) so an
# IK baseline and a policy evaluated with the same seed see the same μ sequence.
WHEEL_BODIES = ["fl_wheel_link", "fr_wheel_link", "rl_wheel_link", "rr_wheel_link"]  # module order


def _wheel_shape_slices(asset):
    view = asset.root_physx_view
    counts = [asset._physics_sim_view.create_rigid_body_view(p).max_shapes for p in view.link_paths[0]]
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    names = list(asset.body_names)
    return [slice(offsets[names.index(b)], offsets[names.index(b) + 1]) for b in WHEEL_BODIES]


def randomize_wheel_mu(env, env_ids, mu_range=(0.2, 0.6), per="pair", dyn_ratio_range=(0.8, 1.0),
                       master_seed: int = 0):
    """Sample static μ per wheel and write it to the wheel collision shapes.

    per="global": one μ for all four wheels; "pair": left/right independent;
    "wheel": all four independent. dynamic μ = static × U(dyn_ratio_range).
    Realized values go to env._wheel_mu4 (N,4 static) and env._wheel_mu_realized
    (left/right means, same layout eval_log already reads). Never an observation.
    """
    import numpy as np

    asset = env.scene["robot"]
    n = env.scene.num_envs
    if not hasattr(env, "_wheel_mu_draws"):
        env._wheel_mu_draws = [0] * n
        env._wheel_mu4 = torch.full((n, 4), float("nan"))
        env._wheel_mu_realized = [None] * n
        env._wheel_shape_slices = _wheel_shape_slices(asset)
    if env_ids is None:
        ids = list(range(n))
    elif torch.is_tensor(env_ids):
        ids = [int(i) for i in env_ids.tolist()]
    else:
        ids = [int(i) for i in env_ids]
    props = asset.root_physx_view.get_material_properties().clone().cpu()
    lo, hi = mu_range
    for i in ids:
        rng = np.random.default_rng([master_seed, i, env._wheel_mu_draws[i]])
        env._wheel_mu_draws[i] += 1
        if per == "global":
            mu = np.full(4, rng.uniform(lo, hi))
        elif per == "pair":
            l, r = rng.uniform(lo, hi, 2)
            mu = np.array([l, r, l, r])          # fl fr rl rr
        else:
            mu = rng.uniform(lo, hi, 4)
        ratio = rng.uniform(*dyn_ratio_range, 4)
        for k, sl in enumerate(env._wheel_shape_slices):
            props[i, sl, 0] = float(mu[k])
            props[i, sl, 1] = float(mu[k] * ratio[k])
            props[i, sl, 2] = 0.0
        env._wheel_mu4[i] = torch.as_tensor(mu, dtype=torch.float32)
        env._wheel_mu_realized[i] = {"left": {"static": float(mu[[0, 2]].mean())},
                                     "right": {"static": float(mu[[1, 3]].mean())}}
    asset.root_physx_view.set_material_properties(props, torch.as_tensor(ids, dtype=torch.int32))


# --- Main-training μ model: terrain base × per-wheel factor + slick patches -------
# Real sand is one material under all four wheels with local variation, plus the
# occasional loose/slick patch under one wheel or one side -- not four unrelated
# draws. So: a per-episode base μ (terrain type), per-wheel factors around it, and
# at each interval (per-env random 1-5 s) the factors are redrawn and, with some
# probability, one wheel or one side drops to a slick μ. The base stays fixed until
# reset. Deterministic in (master_seed, env_id, draw counter) like randomize_wheel_mu.
def randomize_wheel_mu_patch(env, env_ids, base_range=(0.25, 0.8), factor_range=(0.75, 1.25),
                             uniform_prob=0.2, slick_prob=0.25, slick_range=(0.1, 0.2),
                             mu_clip=(0.08, 1.0), dyn_ratio_range=(0.8, 1.0),
                             new_base: bool = True, master_seed: int = 0):
    import numpy as np

    asset = env.scene["robot"]
    n = env.scene.num_envs
    if not hasattr(env, "_wheel_mu_draws"):
        env._wheel_mu_draws = [0] * n
        env._wheel_mu4 = torch.full((n, 4), float("nan"))
        env._wheel_mu_realized = [None] * n
        env._wheel_shape_slices = _wheel_shape_slices(asset)
    if not hasattr(env, "_wheel_mu_base"):
        env._wheel_mu_base = [None] * n            # (base, uniform?) per env
    if env_ids is None:
        ids = list(range(n))
    elif torch.is_tensor(env_ids):
        ids = [int(i) for i in env_ids.tolist()]
    else:
        ids = [int(i) for i in env_ids]
    props = asset.root_physx_view.get_material_properties().clone().cpu()
    for i in ids:
        rng = np.random.default_rng([master_seed, i, env._wheel_mu_draws[i]])
        env._wheel_mu_draws[i] += 1
        if new_base or env._wheel_mu_base[i] is None:
            env._wheel_mu_base[i] = (rng.uniform(*base_range), rng.random() < uniform_prob)
        base, uniform = env._wheel_mu_base[i]
        mu = np.full(4, base) if uniform else base * rng.uniform(*factor_range, 4)
        if rng.random() < slick_prob:
            # wheels fl, fr, rl, rr = 0..3. one wheel 40%, one side 30%, diagonal 10%,
            # front/rear pair 10%, all four 10% (traction limit)
            u = rng.random()
            if u < 0.4:
                idx = [int(rng.integers(4))]
            else:
                pairs = ([0, 2], [1, 3]) if u < 0.7 else ([0, 3], [1, 2]) if u < 0.8 else ([0, 1], [2, 3]) if u < 0.9 else None
                idx = [0, 1, 2, 3] if pairs is None else pairs[int(rng.integers(2))]
            mu[idx] = rng.uniform(*slick_range, len(idx))
        mu = np.clip(mu, *mu_clip)
        ratio = rng.uniform(*dyn_ratio_range, 4)
        for k, sl in enumerate(env._wheel_shape_slices):
            props[i, sl, 0] = float(mu[k])
            props[i, sl, 1] = float(mu[k] * ratio[k])
            props[i, sl, 2] = 0.0
        env._wheel_mu4[i] = torch.as_tensor(mu, dtype=torch.float32)
        env._wheel_mu_realized[i] = {"left": {"static": float(mu[[0, 2]].mean())},
                                     "right": {"static": float(mu[[1, 3]].mean())}}
    asset.root_physx_view.set_material_properties(props, torch.as_tensor(ids, dtype=torch.int32))


def randomize_base_mass_com(env, env_ids, mass_scale=(1.0, 1.0), com_range_m=(0.0, 0.0, 0.0)):
    """Startup: scale base_link mass (inertia scaled alike) and shift its CoM; record both.

    Stores env._priv_mass_com = [mass/nominal, dCoM x,y,z (m)] so the teacher's
    privileged observation reports exactly what was applied.
    """
    robot = env.scene["robot"]
    view = robot.root_physx_view
    n = env.scene.num_envs
    ids = torch.arange(n) if env_ids is None else env_ids.cpu()
    b = robot.body_names.index("base_link")
    scale = torch.empty(len(ids)).uniform_(*mass_scale)
    masses = view.get_masses().clone()
    inertias = view.get_inertias().clone()
    masses[ids, b] = robot.data.default_mass[ids, b].cpu() * scale
    inertias[ids, b] = robot.data.default_inertia[ids, b].cpu() * scale.unsqueeze(1)
    view.set_masses(masses, ids)
    view.set_inertias(inertias, ids)
    coms = view.get_coms().clone()                      # (N, bodies, 7) pos + quat
    d = torch.stack([torch.empty(len(ids)).uniform_(-r, r) for r in com_range_m], dim=1)
    coms[ids, b, :3] += d
    view.set_coms(coms, ids)
    v = torch.zeros(n, 4)
    v[:, 0] = 1.0
    v[ids, 0] = scale
    v[ids, 1:] = d
    env._priv_mass_com = v.to(env.device)
