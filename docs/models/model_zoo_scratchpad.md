# mFormerV0

# mFormerV1

**Revised Plan for `mFormerV1` Variants:**

1.  **`mFormerV1_sm`:** `convnext_tiny_22k_1k_384.pth` + `rope_mixed_deit_small_patch16_LS.pth`
2.  **`mFormerV1_md`:** `convnext_small_22k_1k_384.pth` + `rope_mixed_deit_small_patch16_LS.pth`
3.  **`mFormerV1_lg`:** `convnext_large_22k_1k_384.pth` + `rope_mixed_deit_base_patch16_LS.pth`
4.  **`mFormerV1_xl`:** `convnext_xlarge_22k_1k_384_ema.pth` + `rope_mixed_deit_large_patch16_LS.pth`

Now, let's perform the dimensionality check using the inspection reports and update the YAMLs.

**1. Dimensionality Compatibility Check & Table**

| Variant       | ConvNeXt Checkpoint                 | ConvNeXt Stages (DIMS)      | Output Dim @ S2 Juncture | RoPE-ViT Checkpoint                 | RoPE Input Dim | Compatible? |
| :------------ | :---------------------------------- | :-------------------------- | :----------------------- | :---------------------------------- | :------------- | :---------- |
| `mFormerV1_sm` | `convnext_tiny_22k_1k_384.pth`      | `[96, 192, 384, 768]`       | **384**                  | `rope_mixed_deit_small_patch16_LS`  | **384**        | ✅ **Yes**   |
| `mFormerV1_md` | `convnext_small_22k_1k_384.pth`     | `[96, 192, 384, 768]`       | **384**                  | `rope_mixed_deit_small_patch16_LS`  | **384**        | ✅ **Yes**   |
| `mFormerV1_lg` | `convnext_large_22k_1k_384.pth`     | `[192, 384, 768, 1536]`     | **768**                  | `rope_mixed_deit_base_patch16_LS`   | **768**        | ✅ **Yes**   |
| `mFormerV1_xl` | `convnext_xlarge_22k_1k_384_ema.pth`| `[256, 512, 1024, 2048]`    | **1024**                 | `rope_mixed_deit_large_patch16_LS`  | **1024**       | ✅ **Yes**   |

*   **S2 Juncture Dim:** This is the dimension *after* the ConvNeXt stage 1 blocks and the subsequent downsampler (`mFormerV1.downsample_layers[1]`), which corresponds to `CONVNEXT_STAGES.DIMS[2]` in the mFormerV1 config.
*   **RoPE Input Dim:** This is the `embed_dim` of the RoPE-ViT checkpoint, required by the first RoPE block (`mFormerV1.stages[2]`).
