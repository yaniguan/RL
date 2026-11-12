# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from nemo_rl.algorithms.loss.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.reward_functions import (
    RewardShapingConfig,
    apply_reward_shaping,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_gspo_sequence_is_geometric_mean_prevents_long_seq_overflow() -> None:
    """Long sequences with a constant +0.2 per-token logprob delta overflow
    exp(sum_t delta) when T=512, whereas geometric mean stays exp(0.2) ~= 1.2214.
    """
    bsz, seq_len = 2, 513  # 512 next-token transitions
    delta = 0.2
    prev_lp = torch.full((bsz, seq_len), -1.0, dtype=torch.float64)
    gen_lp = prev_lp - delta
    curr_lp = prev_lp[:, :-1].clone().requires_grad_(True)
    ref_lp = prev_lp.clone()
    token_mask = torch.ones((bsz, seq_len), dtype=torch.float64)
    token_mask[:, 0] = 0.0
    sample_mask = torch.ones(bsz, dtype=torch.float64)
    advantages = torch.tensor([[1.0], [-1.0]], dtype=torch.float64).expand(bsz, seq_len)

    data = {
        "input_ids": torch.zeros((bsz, seq_len), dtype=torch.int64),
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "advantages": advantages,
        "prev_logprobs": prev_lp,
        "generation_logprobs": gen_lp,
        "reference_policy_logprobs": ref_lp,
        "rewards": torch.tensor([1.0, 0.0], dtype=torch.float64),
    }

    # Legacy sum-exponent mode overflows exp(512 * 0.2) -> inf -> nan_to_num -> 0.0
    legacy_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            token_level_loss=False,
            sequence_level_importance_ratios=True,
            use_importance_sampling_correction=True,
            sequence_level_importance_sampling_geometric_mean=False,
            reference_policy_kl_penalty=0.0,
        )
    )
    legacy_loss, _ = legacy_fn(
        next_token_logprobs=curr_lp,
        data=data,
        global_valid_seqs=sample_mask.sum(),
        global_valid_toks=(token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum(),
    )
    assert legacy_loss.item() == 0.0

    # Geometric-mean mode preserves exact exp(0.2) importance weight without overflow
    curr_lp_geo = prev_lp[:, :-1].clone().requires_grad_(True)
    geo_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            token_level_loss=False,
            sequence_level_importance_ratios=True,
            use_importance_sampling_correction=True,
            sequence_level_importance_sampling_geometric_mean=True,
            reference_policy_kl_penalty=0.0,
        )
    )
    geo_loss, geo_metrics = geo_fn(
        next_token_logprobs=curr_lp_geo,
        data=data,
        global_valid_seqs=sample_mask.sum(),
        global_valid_toks=(token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum(),
    )
    geo_loss.backward()
    assert torch.isfinite(geo_loss)
    assert abs(geo_metrics["sampling_importance_ratio"] - float(torch.exp(torch.tensor(delta)))) < 1e-6
    assert curr_lp_geo.grad is not None and torch.any(curr_lp_geo.grad != 0.0)


def test_dapo_overlong_clamp_and_combined_stop_properly() -> None:
    """Verify clamp_overlong_penalty caps multi-turn excess penalties at
    -overlong_buffer_penalty and combine_stop_properly_and_overlong applies both.
    """
    batch = BatchedDataDict(
        {
            "response_token_lengths": torch.tensor([15, 25, 30, 80]),
            "truncated": torch.tensor([False, False, True, False]),
            "total_reward": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        }
    )
    cfg = RewardShapingConfig(
        enabled=True,
        max_response_length=30,
        overlong_buffer_length=10,
        overlong_buffer_penalty=0.5,
        stop_properly_penalty_coef=0.2,
        clamp_overlong_penalty=True,
        combine_stop_properly_and_overlong=True,
    )
    out = apply_reward_shaping(batch, cfg)
    # expected_response_length = 20
    # idx 0 (len 15, not trunc): 1.0 + 0 = 1.0
    # idx 1 (len 25, not trunc): 1.0 - (5/10)*0.5 = 0.75
    # idx 2 (len 30, trunc): 1.0 * 0.2 - (10/10)*0.5 = -0.3
    # idx 3 (len 80 multi-turn sum, not trunc): capped at -0.5 -> 1.0 - 0.5 = 0.5
    expected = torch.tensor([1.0, 0.75, -0.3, 0.5])
    torch.testing.assert_close(out["total_reward"], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        out["unshaped_total_reward"], torch.tensor([1.0, 1.0, 1.0, 1.0]), rtol=1e-6, atol=1e-6
    )
