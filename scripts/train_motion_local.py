"""Run the standard trainer, failing before an optimizer step on nonfinite gradients.

Uses the normal sm-train CLI; launch with --config configs/motion_tracking_5070ti.yaml.
"""

from functools import wraps

import torch

_clip_grad_norm = torch.nn.utils.clip_grad_norm_


@wraps(_clip_grad_norm)
def _checked_clip_grad_norm(*args, **kwargs):
    kwargs["error_if_nonfinite"] = True
    return _clip_grad_norm(*args, **kwargs)


torch.nn.utils.clip_grad_norm_ = _checked_clip_grad_norm

from supermimic.scripts.train import main  # noqa: E402

if __name__ == "__main__":
    main()
