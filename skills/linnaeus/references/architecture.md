# Linnaeus Architecture Reference

Detailed module map for operators and contributors working in `~/dev/linnaeus/dev` and `~/dev/linnaeus/wt/<branch>`.

## Top-level map

```
linnaeus/
  main.py                        # training lifecycle entry point
  train.py                       # train_one_epoch and step accounting
  validation.py                  # validation passes and metric finalization
  config.py                      # YACS defaults and config hierarchy

  models/
    model_factory.py             # model/head registration and builders
    mFormerV0.py                 # hybrid Conv-Transformer baseline family
    mFormerV1.py                 # ConvNeXt + RoPE family
    blocks/                      # attention, MLP, drop-path, embedding blocks
    heads/                       # linear, hierarchical-softmax, conditional heads

  h5data/
    prefetching_hybrid_dataset.py
    prefetching_h5_dataset.py
    h5dataloader.py
    grouped_batch_sampler.py

  loss/
    basic_loss.py
    hierarchical_loss.py
    taxonomy_label_smoothing.py
    gradnorm.py
    masking.py

  ops_schedule/
    ops_schedule.py              # validation/checkpoint/log timing policy
    training_progress.py

  profiling/
    cli.py                       # linnaeus-prof entry point
    scanner.py
    summary.py
    diff.py
    repair.py
    tensorboard_launcher.py
    concurrent_executor.py
    gpu_pool.py

  tools/profiling/
    run_profiling_trials.py      # linnaeus-prof-run entry point
```

## Config and runtime boundaries

- `linnaeus/config.py` defines defaults and schedule knobs used by `linnaeus/main.py`.
- `--opts` CLI overrides are highest precedence in the config stack.
- Validation cadence and partial-mask/meta validations are scheduled via `ops_schedule`.

## Profiling boundaries

- `linnaeus-prof-run` is for controlled trial execution and receipts.
- `linnaeus-prof` is for post-run analysis (`scan`, `summary`, `diff`, `repair`, `tensorboard`).
- Profiling docs live under `docs/profiling/` and should stay aligned with actual parser flags.

## Storage/layout boundaries

- Public example configs remain in `configs/`.
- Private run configs and templates live in:
  - `~/dev/linnaeus/dev/private/configs/`
  - `~/dev/linnaeus/dev/private/docker/runtime/profiling/`
- Worktree branches should live under `~/dev/linnaeus/wt/<branch>`.

