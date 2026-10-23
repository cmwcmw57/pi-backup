"""In-process fixes for rsl-rl-lib 3.1.2 (the image's version); applied on import.

StudentTeacherRecurrent builds its teacher MLP with input size ``rnn_hidden_dim`` even
when ``teacher_recurrent=False`` -- then the teacher has no RNN in front of it and must
take the raw teacher observation (here policy 58 + privileged 22 = 80). As shipped, a
plain-MLP PPO teacher can never be loaded into a GRU student ("size mismatch for
0.weight: [256, 80] vs [256, 128]"). evaluate() already skips memory_t in that case, so
rebuilding the MLP with the right input size is the whole fix. The class object is
patched in place, so the DistillationRunner's own reference picks it up.
"""

import inspect


def _patch_student_teacher_recurrent():
    from rsl_rl.modules import StudentTeacherRecurrent
    from rsl_rl.networks import MLP

    if getattr(StudentTeacherRecurrent, "_ksrc_patched", False):
        return
    orig = StudentTeacherRecurrent.__init__
    sig = inspect.signature(orig)

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if self.teacher_recurrent:
            return
        b = sig.bind(self, *args, **kwargs)
        b.apply_defaults()
        obs, groups = b.arguments["obs"], b.arguments["obs_groups"]
        n_teacher = sum(obs[g].shape[-1] for g in groups["teacher"])
        self.teacher = MLP(n_teacher, b.arguments["num_actions"], b.arguments["teacher_hidden_dims"],
                           b.arguments["activation"])
        self.teacher.eval()
        print(f"[ksrc patch] non-recurrent teacher MLP input = {n_teacher}: {self.teacher}")

    StudentTeacherRecurrent.__init__ = __init__
    StudentTeacherRecurrent._ksrc_patched = True


try:
    _patch_student_teacher_recurrent()
except ImportError:   # e.g. laptop tools without rsl_rl
    pass
