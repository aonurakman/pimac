# Findings for `simple_spread`

Archived source suite: `combined2` / `simple_spread_v3`

These notes summarize patterns from the archived top configs in `configs/`.
They are useful priors for new runs, not universal claims about the full search space.
The archived FiLM-plus-head sweeps that used to be labeled `pimac_v3` are now tracked here as `pimac_v4`.
The current head-only `pimac_v3` has no archived findings for this task yet.

## Main takeaways
This task is useful as a fixed-team benchmark, but it is not the main stress test for variable team composition.
On the archived combined2 sweep, VDN was the strongest method by study objective, followed by IQL and QMIX.
The richer PIMAC variants did not pay off here. The simple set-critic baseline PIMAC_v0 was the best PIMAC on this task, while v1/v2/v4 underperformed the stronger value-based baselines.
Interpretation: this task is easy enough that extra context machinery is not clearly rewarded, so it should be treated mainly as a sanity and stability benchmark.

## Best study scores by algorithm
- `mappo`: `-18.953520584106442`
- `iql`: `-15.619139480590821`
- `ippo`: `-26.109231948852536`
- `qmix`: `-16.69352989196777`
- `vdn`: `-13.332641601562498`
- `pimac_v0`: `-19.521293449401853`
- `pimac_v1`: `-25.477704238891597`
- `pimac_v2`: `-24.628803253173828`
- `pimac_v4`: `-22.08838596343994`

## Parameter zones that kept showing up
- `vdn`: the strongest retained configs stayed in a simple recurrent regime: `rnn_hidden_dim=64`, `tau=0.5`, `gamma` around `0.97`, `num_epochs=1-2`, and exploration floors around `temp_min=0.05-0.1`. The best archived config used `widths=[64, 128, 64]`.
- `iql`: the best archived config was also simple: `gamma=0.97`, `temp_init=0.5`, `temp_decay=0.995`, `batch_size=128`, `num_epochs=1`, and no recurrent core (`rnn_hidden_dim=0`, `seq_len=4`). More complex recurrent variants survived into the top five, but not at the very top.
- `mappo`: the stable fixed-team PPO backbone was narrow and consistent: `widths=[128, 128, 128]`, `batch_size=32`, `clip_eps=0.15`, `gamma=0.97`, `num_epochs=8`, `rnn_hidden_dim=64`.
- `pimac_v0`: the best PIMAC kept the same stable PPO backbone and used a modest set critic: `critic_hidden_sizes=[64, 64, 64]` or `[64, 64]`, `set_embed_dim=128`, `set_encoder_hidden_sizes=[128, 128]`. Among the archived top configs, `include_team_size_feature=False` showed up more often than `True`.
- `pimac_v1/v2/v4`: all retained top configs sat on the same fixed-team backbone as `pimac_v0`, with `num_tokens=4` throughout. The extra context machinery did not produce a better return regime on this task.

## What did not look worth revisiting first
- `ippo`: performance was poor across the archived configs, and there was no obvious hyperparameter pocket that closed the gap to `vdn`, `iql`, or `qmix`.
- `pimac_v1`: larger distillation weights were present in archived top configs, but even the best retained settings did not beat `pimac_v0`. This task does not appear to reward heavy teacher-student shaping.
- `pimac_v2`: broad uncertainty ranges such as `ctx_logvar_min=-8` and `ctx_logvar_max=2` survived into the archived top configs, but they still did not make `v2` competitive with `v0` here.
- `pimac_v4`: richer hypernet settings, including `hypernet_rank=4-8` and larger delta scales, were explored and archived, but they still did not justify the added complexity on this fixed-team benchmark.
