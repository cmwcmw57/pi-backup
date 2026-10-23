"""Shared PPO runner settings for the three action-space comparisons."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg,
                               RslRlDistillationStudentTeacherCfg,
                               RslRlDistillationStudentTeacherRecurrentCfg, RslRlOnPolicyRunnerCfg,
                               RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)


@configclass
class RoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "ksrc_rover"
    # rsl-rl-lib 5.x needs an explicit obs-group -> algo-set map. Single
    # "policy" group feeds both actor and critic (symmetric; critic not yet privileged).
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RoverResidual3PPORunnerCfg(RoverPPORunnerCfg):
    experiment_name = "ksrc_rover_residual3"


@configclass
class RoverRes8PPORunnerCfg(RoverPPORunnerCfg):
    experiment_name = "ksrc_rover_res8"


@configclass
class RoverDirect8PPORunnerCfg(RoverPPORunnerCfg):
    experiment_name = "ksrc_rover_direct8"


# --- Teacher (main training, stage 1) ---------------------------------------------
# Actor and critic both read the proprioceptive "policy" group plus the "privileged"
# group (true velocity, per-wheel μ and slip, mass/CoM). Plain MLP, no recurrence:
# the teacher is told the hidden state, so it has nothing to infer. A student is
# distilled from it later. Lower entropy than the single-stage runs: in the S6 run
# Res8's clipped residual got almost no reward signal and the entropy bonus alone
# pushed its noise std 0.4 -> 8.6; watch Policy/mean_noise_std in TensorBoard.
@configclass
class RoverTeacherPPORunnerCfg(RoverPPORunnerCfg):
    obs_groups = {"policy": ["policy", "privileged"], "critic": ["policy", "privileged"]}
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.002,                       # only change vs RoverPPORunnerCfg (0.005)
        num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class RoverTeacherDirect8PPORunnerCfg(RoverTeacherPPORunnerCfg):
    experiment_name = "ksrc_rover_teacher_direct8"


@configclass
class RoverTeacherRes8PPORunnerCfg(RoverTeacherPPORunnerCfg):
    experiment_name = "ksrc_rover_teacher_res8"
    # Larger initial exploration for the residual only (2026-10-11): at 0.3 a trial moved a
    # wheel ~3°/9% even at the widened ±10°/±30% range. Direct8 keeps 0.3 (its action is
    # the full ±90° steering angle, 0.5 would mean ±45° jitter early on).
    # 2026-10-12 wide Res8 (±30°/±60%): 0.5 would be ±15°/30% jitter, 0.3 keeps it at ±9°/18%.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3, actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[256, 128, 128], activation="elu",
    )


@configclass
class RoverTeacherResidual3PPORunnerCfg(RoverTeacherPPORunnerCfg):
    experiment_name = "ksrc_rover_teacher_residual3"


# --- Student (distillation from a trained teacher) ------------------------------------
# DAgger-style: the student drives the robot, the frozen teacher labels each state with
# its action from privileged obs, and the student regresses onto it (MSE). Student sees
# only the "student" group (deployable sensors, STUDENT_HISTORY steps). The teacher net
# must match the teacher run exactly (hidden dims, obs groups, obs normalization).
# experiment_name = the teacher's folder, because the stock train.py looks up the teacher
# checkpoint there (--load_run <teacher run> --checkpoint model_XXXX.pt); run_name tags
# the student runs inside it.
_TEACHER_NET = dict(teacher_hidden_dims=[256, 128, 128], teacher_obs_normalization=True, activation="elu")
_DISTILL_ALGO = dict(num_learning_epochs=2, learning_rate=1.0e-3, gradient_length=15,
                     max_grad_norm=1.0, loss_type="mse")


@configclass
class RoverStudentMLPRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    obs_groups = {"policy": ["student"], "teacher": ["policy", "privileged"]}
    run_name = "student_mlp"
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1, student_obs_normalization=True,
        student_hidden_dims=[256, 128, 128], **_TEACHER_NET)
    algorithm = RslRlDistillationAlgorithmCfg(**_DISTILL_ALGO)


@configclass
class RoverStudentGRURunnerCfg(RoverStudentMLPRunnerCfg):
    # Run with STUDENT_HISTORY=1 (train.py --set): the GRU is the memory.
    run_name = "student_gru"
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1, student_obs_normalization=True,
        student_hidden_dims=[128, 128], **_TEACHER_NET,
        rnn_type="gru", rnn_hidden_dim=128, rnn_num_layers=1, teacher_recurrent=False)


@configclass
class RoverStudentMLPDirect8RunnerCfg(RoverStudentMLPRunnerCfg):
    experiment_name = "ksrc_rover_teacher_direct8"


@configclass
class RoverStudentGRUDirect8RunnerCfg(RoverStudentGRURunnerCfg):
    experiment_name = "ksrc_rover_teacher_direct8"


@configclass
class RoverStudentMLPRes8RunnerCfg(RoverStudentMLPRunnerCfg):
    experiment_name = "ksrc_rover_teacher_res8"


@configclass
class RoverStudentGRURes8RunnerCfg(RoverStudentGRURunnerCfg):
    experiment_name = "ksrc_rover_teacher_res8"


@configclass
class RoverStudentMLPResidual3RunnerCfg(RoverStudentMLPRunnerCfg):
    experiment_name = "ksrc_rover_teacher_residual3"


@configclass
class RoverStudentGRUResidual3RunnerCfg(RoverStudentGRURunnerCfg):
    experiment_name = "ksrc_rover_teacher_residual3"
