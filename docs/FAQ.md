# FAQ

This page answers the questions a new reader is most likely to have after
opening the repo. If you only read one other page, make it
[Current State](current_state.md).

## Is there a public model zoo I can use today?

No verified public Linnaeus model registry is live from this repo today.

The old plan to publish a North American `mFormerV1_sm` suite is retired. If
public artifacts ship on the current trajectory, expect bundle-shaped
`DINOv3MultiHead` releases rather than the older mFormer line.

## What architecture is current?

The active line is `DINOv3MultiHead`:

- frozen DINOv3 ViT-B/16 backbone
- hierarchical classification heads
- optional MIL mean pooling for multi-view bags

Older `mFormerV0` and `mFormerV1` code still exists, but those families are
legacy surfaces, not the current release story.

## How am I supposed to run inference?

Start from a local inference bundle and point
`LinnaeusInferenceHandler.load_from_artifacts(...)` at its
`inference_config.yaml`.

Do not assume:

- a bare model ID is enough
- a hosted API is the primary public surface
- every bundle supports multi-view inference

The current contract is documented in [Inference Overview](inference/overview.md)
and [Inference Bundle](inference/inference_bundle.md).

## Is the config system already fully migrated to Pydantic?

No.

YACS still owns runtime config resolution today. Typed and Pydantic-backed
validation is being added incrementally. Treat the newer validation surface as
hardening around the existing runtime system, not as a completed replacement.

## What command should I use for training?

For source training, the current entrypoint is:

```bash
uv run python -m linnaeus.main --cfg path/to/experiment.yaml
```

For public source checkouts, inspect the training entrypoint directly:

```bash
uv run python -m linnaeus.main --help
```

See [Training Overview](training/overview.md) for the current source-training
path.

## Are the older mFormer docs still useful?

Sometimes, but mostly as provenance.

They can still help you understand:

- historical experiment structure
- older loss and head designs
- migration and cutover context

They should not be read as the current product or release plan.

## Which repo should I watch?

Watch [`polli-labs/linnaeus`](https://github.com/polli-labs/linnaeus) if you
want the public release surface.

Day-to-day work still lands in `polli-labs/linnaeus-dev` first and is promoted
outward deliberately. The migration docs explain that split if you need it;
most public readers do not.
