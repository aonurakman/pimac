# Findings for `simple_spread_dynamic`

Archived source suite: `combined2` / `simple_spread_dynamic_v3`

These notes summarize patterns from the archived top configs in `configs/`.
They are useful priors for new runs, not universal claims about the full search space.
The archived FiLM-plus-head sweeps that used to be labeled `pimac_v3` are now tracked here as `pimac_v4`.
The current head-only `pimac_v3` has no archived findings for this task yet.

## Main takeaways
This is the main variable-team benchmark from the earlier sweeps.
On combined2, PIMAC_v2 was the best overall method by study objective. PIMAC_v0 and PIMAC_v4 were also competitive, and the PIMAC family clearly improved over the simpler fixed-team task.
This benchmark is still methodologically weaker than the hard version for local context recovery, because the actor observations are rich and team-size-related shortcuts are easier to exploit.
Interpretation: keep this task as a performance and transfer benchmark, but do not overclaim it as clean evidence of decentralized context recovery.

## Best study scores by algorithm
- `mappo`: `-32.147239685058594`
- `iql`: `-33.03367691040039`
- `ippo`: `-32.11944580078125`
- `qmix`: `-31.902170944213864`
- `vdn`: `-31.9642333984375`
- `pimac_v0`: `-31.727611541748047`
- `pimac_v1`: `-32.068980026245114`
- `pimac_v2`: `-30.76941566467285`
- `pimac_v4`: `-31.772278404235838`

## Parameter zones that kept showing up
- `mappo`: the better archived dynamic configs moved away from the fixed-team backbone toward `widths=[96, 96]` or `[96, 128, 96]`, `critic_hidden_sizes=[96, 96, 96]`, `rnn_hidden_dim=96`, `batch_size=32`, `clip_eps=0.25-0.30`, `gamma=0.98-0.985`, and `num_epochs=4-5`.
- `vdn` and `qmix`: the value-based dynamic regime stayed conservative and stable: `rnn_hidden_dim=64`, `tau=0.1`, `temp_init=1.0-2.0`, `temp_min=0.1-0.2`, `temp_decay≈0.9995-0.99985`. For `qmix`, the best retained mixer was small, usually `mixing_embed_dim=32` with `mixing_widths=[64]`.
- `pimac_v0`: the best retained configs already show the main dynamic-task pattern: `include_team_size_feature=True`, `critic_hidden_sizes=[96, 96]`, `set_embed_dim=96-128`, `set_encoder_hidden_sizes=[128, 64]` or `[128, 96]`, with the stronger dynamic PPO backbone inherited from `mappo`.
- `pimac_v1`: the best retained settings used a fairly rich communication budget, most often `num_tokens=10`, with light-to-moderate distillation (`distill_weight` roughly `7e-4` to `5e-3`) and `teacher_ema_tau=0.01-0.02`.
- `pimac_v2`: this was the clearest winner on this task. The retained top configs stayed very consistent: `num_tokens=10`, `distill_weight` roughly `3e-4` to `1.6e-3`, `teacher_ema_tau=0.005-0.01`, and uncertainty bounds tighter than the old defaults, usually around `[-4, 2]`, `[-2, 2]`, or `[-1, 1]`.
- `pimac_v4`: competitive settings existed, but they were still fairly restrained: `hypernet_rank=2-3`, `hypernet_hidden_sizes=[64, 64]` or `[128, 128]`, `hypernet_delta_init_scale=0.02`, and small `hypernet_l2_coef`.

## What did not look worth revisiting first
- `ippo`: even though the archived dynamic configs moved toward larger batches and somewhat stronger recurrent widths, `ippo` still did not establish a clearly competitive zone.
- `pimac` without team-size information: among the retained dynamic PIMAC configs, `include_team_size_feature=True` was the consistent pattern. The dynamic task appears to reward explicit team-size awareness.
- `pimac_v1`: smaller token counts such as `4` or `6` survived into the archived top five, but the strongest retained settings were the `num_tokens=10` ones.
- `pimac_v4`: there is no archived evidence that aggressive hypernets are the right first thing to revisit here. The retained top configs stayed in moderate-rank, modest-delta regimes.
