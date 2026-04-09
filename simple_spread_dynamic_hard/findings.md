# Findings for `simple_spread_dynamic_hard`

Archived source suite: `combined_hard2` / `simple_spread_dynamic_hard`

These notes summarize patterns from the archived top configs in `configs/` and the earlier hard-task sweeps that produced them.
They are useful priors for new runs, not universal claims about the full search space.
The archived FiLM-plus-head sweeps that used to be labeled `pimac_v3` are now tracked here as `pimac_v4`.
The current head-only `pimac_v3` has no archived findings for this task yet.

## Main takeaways
This is the stronger benchmark for decentralized local context recovery.
On combined_hard2, VDN and QMIX were still the strongest methods on return. Among PIMAC variants, PIMAC_v1 had the best return recovery, while PIMAC_v2 had the best student-teacher alignment and the strongest local context recovery metrics.
The hard sweeps showed a clear pattern: simpler token setups, larger but stable critic/set capacity, lighter distillation, and very conservative v4 hypernet updates were more reliable than the more aggressive settings that worked in easier tasks.
Interpretation: use this task when evaluating the tradeoff between raw return competitiveness and genuine local coordination-context recovery.

## Best study scores by algorithm
- `mappo`: `-38.179466247558594`
- `iql`: `-34.14507293701172`
- `ippo`: `-57.2286262512207`
- `qmix`: `-27.73268070220947`
- `vdn`: `-26.694021224975586`
- `pimac_v0`: `-36.68341979980469`
- `pimac_v1`: `-35.3753547668457`
- `pimac_v2`: `-36.13134841918945`
- `pimac_v4`: `-37.82484588623046`

## Parameter zones that kept showing up
- `vdn`: the strongest return regime stayed simple and repeatable: `rnn_hidden_dim=64`, `tau=0.1`, `temp_init=1.0`, `temp_min=0.1-0.2`, `temp_decay≈0.9995-0.99985`, and `gamma` usually `0.97`. The best archived config used `widths=[64, 128, 64]`.
- `qmix`: the best retained hard-task mixers were also small: `mixing_embed_dim=32`, `mixing_widths=[64]`, `tau=0.1`, `temp_init=1.5-2.0`, `temp_decay≈0.9997`, with `gamma=0.97`. Larger or more aggressive mixers did not dominate the archived top configs.
- `mappo`: the best pure PPO baseline on the hard task was actually simpler than the dynamic-task winner: `widths=[64, 64]`, `critic_hidden_sizes=[128, 128, 128]`, `clip_eps=0.2`, `batch_size=32`, `num_epochs=2`, `rnn_hidden_dim=64`.
- `pimac_v0`: the hard task wanted a stronger set critic than the easier tasks: `widths=[96, 128, 96]`, `include_team_size_feature=True`, `critic_hidden_sizes=[128, 128]` or `[160, 160]`, `set_embed_dim=128-160`, `set_encoder_hidden_sizes=[128, 96]` or `[160, 96]`.
- `pimac_v1`: the return-best PIMAC stayed in a narrow and interpretable zone: `num_tokens=3-4`, very light distillation (`~1e-4` to `6e-4`), `teacher_ema_tau=0.005-0.01`, and tight uncertainty bounds such as `[-0.25, 0.75]` or `[-0.5, 1.0]`.
- `pimac_v2`: this remained the best context-recovery variant. The best retained configs used `num_tokens=3-4`, moderate but still fairly light distillation (`~6e-4` to `3e-3`), `teacher_ema_tau=0.0025-0.01`, and uncertainty bounds around `[-0.5, 1.0]`, `[-0.25, 0.5]`, or `[-0.5, 0.5]`.
- `pimac_v4`: the only viable hard-task regime was conservative: `hypernet_rank=1-2`, `hypernet_hidden_sizes=[32, 32]` or `[64, 32]`, `hypernet_delta_init_scale=0.005-0.02`, and relatively stronger `hypernet_l2_coef` than the easier-task winners.

## What did not look worth revisiting first
- `ippo`: performance stayed poor across the archived hard-task configs. There is no retained evidence of a competitive IPPO zone on this benchmark.
- Broad token budgets: on the hard task, the old dynamic-task preference for larger token counts did not survive. Among the retained hard-task top configs, `num_tokens>4` did not remain attractive.
- Heavy distillation: the better hard-task PIMAC variants moved toward lighter distillation, not stronger teacher forcing.
- Wide uncertainty ranges: the broad `ctx_logvar` ranges that were tolerable on the easier dynamic task did not survive into the retained hard-task winners.
- Aggressive `pimac_v4` hypernets: larger ranks, larger hidden stacks, and larger delta scales were not the right direction here. The retained winners only use small residual hypernet updates.
