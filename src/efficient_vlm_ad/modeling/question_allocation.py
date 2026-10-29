"""Question-conditioned token scoring and differentiable prefix allocation.

The DP backend preserves float64 for numerical verification; callers use float32
in production. All budget/slot axes are vectorized; only cameras are looped.
"""
from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass
class QuestionScores:
    scores: Tensor
    alpha: Tensor
    log_camera_shares: Tensor
    camera_shares: Tensor
    counts: Tensor


@dataclass
class SoftPrefixResult:
    selection: Tensor
    expected_budgets: Tensor
    soft_row_mass_error: Tensor
    log_partition: Tensor


def _rms(x: Tensor, eps: float) -> Tensor:
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()


class QuestionConditionedScorer(nn.Module):
    def __init__(self, dim=384, num_heads=4, hidden_dim=64, rms_eps=1e-6):
        super().__init__()
        self.rms_eps = rms_eps
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=0., bias=False, batch_first=True)
        self.fc1 = nn.Linear(2 * dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=.02)

    def forward(self, values, question_states, question_mask, camera_ids, num_cameras):
        if values.ndim != 3 or question_states.ndim != 3:
            raise ValueError('Expected batched visual and question states')
        if question_mask.shape != question_states.shape[:2] or camera_ids.shape != values.shape[:2]:
            raise ValueError('Question mask or camera IDs have incompatible shape')
        mask = question_mask.to(torch.bool)
        if not mask.any(-1).all():
            raise ValueError('Each question must contain a valid token')
        if ((camera_ids < 0) | (camera_ids >= num_cameras)).any():
            raise ValueError('Camera IDs outside configured geometry')
        # Explicitly disable autocast: scoring/allocation are FP32 production paths.
        dtype = torch.float64 if self.fc1.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=values.device.type, enabled=False):
            x = _rms(values.to(dtype), self.rms_eps)
            q = _rms(question_states.to(dtype), self.rms_eps)
            attended, _ = self.attention(q, x, x, need_weights=False)
            mean = (attended * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
            mean = _rms(mean, self.rms_eps).unsqueeze(1).expand_as(x)
            scores = self.fc2(torch.relu(self.fc1(torch.cat((x, mean), -1)))).squeeze(-1)
            membership = camera_ids.unsqueeze(1) == torch.arange(num_cameras, device=values.device)[None, :, None]
            counts = membership.sum(-1)
            if (counts == 0).any():
                raise ValueError('Every camera must have visual tokens')
            camera_logits = torch.logsumexp(scores.unsqueeze(1).masked_fill(~membership, -torch.inf), -1) - counts.to(dtype).log()
            log_shares = camera_logits.log_softmax(-1)
            if not torch.isfinite(scores).all() or not torch.isfinite(log_shares).all():
                raise FloatingPointError('Nonfinite question scores or camera shares')
            return QuestionScores(scores, scores.softmax(-1), log_shares, log_shares.exp(), counts)


def build_prefix_utilities(log_shares, capacities, bank_size, beta=0., training=False, noise=None):
    if log_shares.ndim != 2 or capacities.shape != log_shares.shape or bank_size < 1:
        raise ValueError('Invalid allocation geometry')
    if not torch.isfinite(log_shares).all() or (capacities < 1).any():
        raise ValueError('Shares must be finite and capacities positive')
    if beta < 0:
        raise ValueError('beta must be nonnegative')
    k = torch.arange(1, bank_size + 1, device=log_shares.device, dtype=log_shares.dtype)
    utilities = log_shares.unsqueeze(-1) * k - torch.lgamma(k + 1)
    if training and beta > 0:
        if noise is None:
            uniform = torch.rand_like(utilities).clamp_(torch.finfo(utilities.dtype).tiny, 1-torch.finfo(utilities.dtype).eps)
            noise = -(-uniform.log()).log()
        if noise.shape != utilities.shape or not torch.isfinite(noise).all():
            raise ValueError('Noise must be finite and match utilities')
        utilities = utilities + beta * noise
    valid = k <= capacities.unsqueeze(-1)
    utilities = utilities.masked_fill(~valid, -torch.inf)
    return utilities - utilities.amax(-1, keepdim=True).detach()


def _safe_logsumexp(x, dim):
    valid = torch.isfinite(x).any(dim, keepdim=True)
    safe = torch.where(valid, x, torch.zeros_like(x))
    return torch.logsumexp(safe, dim).masked_fill(~valid.squeeze(dim), -torch.inf)


def _prepare(utilities, total_budget, minimum):
    if utilities.ndim != 3 or not utilities.is_floating_point() or minimum < 1:
        raise ValueError('Expected floating [batch,camera,bank] utilities and positive minimum')
    b, c, r = utilities.shape
    if not b or not c or not r or total_budget < c * minimum or total_budget > c * r:
        raise ValueError('Infeasible total query budget')
    if torch.isnan(utilities).any() or torch.isposinf(utilities).any():
        raise ValueError('Utilities contain NaN or positive infinity')
    k = torch.arange(1, r + 1, device=utilities.device)
    u = utilities.masked_fill(k < minimum, -torch.inf)
    if not torch.isfinite(u).any(-1).all():
        raise ValueError('Camera has no valid allocation')
    totals = torch.arange(total_budget + 1, device=u.device)
    previous = totals[:, None] - k[None, :]
    indices = previous.clamp(0, total_budget)
    return u, k, totals, previous, indices


def hard_prefix_dp(utilities, total_budget, minimum=1):
    u, k, totals, previous, indices = _prepare(utilities, total_budget, minimum)
    b, c, _ = u.shape
    table = u.new_full((b, total_budget + 1), -torch.inf)
    table[:, 0] = 0
    pointers = []
    for camera in range(c):
        candidates = table[:, indices] + u[:, camera, None, :]
        candidates = candidates.masked_fill(previous[None] < 0, -torch.inf)
        table, pointer = candidates.max(-1)  # ascending k, first maximum
        pointers.append(pointer + 1)
    if not torch.isfinite(table[:, total_budget]).all():
        raise ValueError('Infeasible camera capacities for total query budget')
    remaining = torch.full((b, 1), total_budget, device=u.device, dtype=torch.long)
    output = []
    for camera in range(c - 1, -1, -1):
        count = pointers[camera].gather(1, remaining)
        output.append(count)
        remaining = remaining - count
    return torch.cat(output[::-1], -1)


def soft_prefix_dp(utilities, total_budget, minimum=1, tau=1.):
    if tau <= 0:
        raise ValueError('tau must be positive')
    u, k, totals, previous, indices = _prepare(utilities, total_budget, minimum)
    u = u / tau
    b, cameras, rows = u.shape
    base = u.new_full((b, total_budget + 1), -torch.inf)
    base[:, 0] = 0
    forward = [base]
    for camera in range(cameras):
        terms = forward[-1][:, indices] + u[:, camera, None, :]
        forward.append(_safe_logsumexp(terms.masked_fill(previous[None] < 0, -torch.inf), -1))
    log_z = forward[-1][:, total_budget]
    if not torch.isfinite(log_z).all():
        raise ValueError('Infeasible camera capacities for total query budget')
    backward = [base]
    for camera in range(cameras - 1, -1, -1):
        terms = backward[-1][:, indices] + u[:, camera, None, :]
        backward.append(_safe_logsumexp(terms.masked_fill(previous[None] < 0, -torch.inf), -1))
    backward = backward[::-1]
    suffix_total = total_budget - totals[:, None] - k[None, :]
    suffix_indices = suffix_total.clamp(0, total_budget)
    positions = torch.arange(total_budget, device=u.device)
    row = torch.arange(rows, device=u.device)
    prefix_total = positions[:, None] - row[None, :]
    selections = []
    expected = []
    for camera in range(cameras):
        posterior_log = (forward[camera].unsqueeze(-1) + u[:, camera, None, :]
                         + backward[camera+1][:, suffix_indices] - log_z[:, None, None])
        posterior = posterior_log.masked_fill(suffix_total[None] < 0, -torch.inf).exp()
        expected.append((posterior.sum(1) * k).sum(-1))
        inclusion = posterior.flip(-1).cumsum(-1).flip(-1)
        selected = inclusion[:, prefix_total.clamp(0, total_budget), row[None, :]]
        selections.append(selected.masked_fill(prefix_total[None] < 0, 0))
    selection = torch.cat(selections, -1)
    mass = selection.sum(-1, keepdim=True)
    error = (mass.squeeze(-1)-1).abs().amax(-1)
    if not torch.isfinite(selection).all() or (selection < 0).any() or (mass <= 0).any() or (error > 1e-3).any():
        raise FloatingPointError('Invalid soft selection row mass (tolerance 1e-3)')
    return SoftPrefixResult(selection/mass, torch.stack(expected, -1), error, log_z)
