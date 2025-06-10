# Hierarchical Classification Approaches in Linnaeus

This document describes approaches within Linnaeus that leverage taxonomic hierarchies to improve model training and prediction, including Taxonomy-Guided Label Smoothing and specialized hierarchical classification heads. These components utilize the centralized `TaxonomyTree` representation detailed in [Taxonomy Representation](./taxonomy_representation.md).

## 1. Taxonomy-Guided Label Smoothing

### 1.1 Concept

Standard label smoothing distributes a small probability `alpha` uniformly across incorrect classes. Taxonomy-Guided Label Smoothing uses the taxonomic structure (via `TaxonomyTree`) to distribute this `alpha` mass based on taxonomic distance. Mistakes between closely related taxa are penalized less than mistakes between distant taxa.

### 1.2 Implementation (`loss.taxonomy_label_smoothing`, `utils.taxonomy.taxonomy_utils`)

1.  **Matrix Generation (`utils.taxonomy.taxonomy_utils.generate_taxonomy_matrices`):**
    *   Takes the validated `TaxonomyTree` object.
    *   For each `task_key` enabled for this loss:
        *   Computes pairwise distances between classes *at that level* using `taxonomy_tree.build_distance_matrix(task_key)`.
        *   Identifies root nodes *at that level* using `taxonomy_tree.get_root_nodes()`.
        *   Checks config flags (`UNIFORM_ROOTS`, `FALLBACK_TO_UNIFORM`) to determine if uniform smoothing is needed.
        *   Calls `build_taxonomy_smoothing_matrix` with distances, roots, and `alpha` to create the `[N, N]` smoothing matrix `T`.
2.  **Loss Calculation (`loss.taxonomy_label_smoothing.TaxonomyAwareLabelSmoothingCE`):**
    *   Initialized with the pre-computed smoothing matrix `T`.
    *   Expects 1D integer targets `y_true` (shape `[B]`). Handles conversion from 2D one-hot internally.
    *   Gathers the target distributions: `P_smooth = T[y_true]` (shape `[B, N]`).
    *   Computes cross-entropy: `Loss = -sum(P_smooth * log_softmax(Z), dim=1)` (shape `[B]`).
    *   Optionally applies class weighting.

### 1.3 Configuration

Enable via the loss function selection and configure smoothing parameters:

```yaml
LOSS:
  TASK_SPECIFIC:
    TRAIN:
      FUNCS: ['TaxonomyAwareLabelSmoothing', ...] # Enable for specific tasks in train split

  TAXONOMY_SMOOTHING:
    ENABLED: [True, ...] # List matching TASK_KEYS_H5 order
    ALPHA: 0.1           # Smoothing factor (0.0 to 1.0)
    BETA: 1.0            # Distance scaling (higher = sharper)
    UNIFORM_ROOTS: True  # Apply uniform smoothing if all nodes at a level are roots
                         # (Appropriate for highest level in single-root clades like Amphibia)
    FALLBACK_TO_UNIFORM: True # Apply uniform if hierarchy links are missing for a level
    # PARTIAL_SUBTREE_WEIGHTING: False # Future enhancement for metaclades
```

### 1.4 Hyperparameters & Tuning

Key parameters for `TaxonomyAwareLabelSmoothingCE`:

* **`ALPHA`** – Total mass shifted away from the true class. Larger values
  increase smoothing and can improve generalization but may slow convergence.
* **`BETA`** – Temperature controlling how quickly smoothing falls off with
  taxonomic distance. Higher values focus probability on very close relatives,
  while lower values spread mass more evenly across the taxonomy.

When combined with **GradNorm**, consider tuning GradNorm's `ALPHA` as well.
A smaller GradNorm `ALPHA` (e.g., `0.1` or `0.5`) yields more conservative task
weight updates, whereas larger values aggressively prioritize tasks that are
learning slowly.

## 2. Hierarchical Classification Heads

These heads replace standard output layers, incorporating hierarchy directly. They are configured under `MODEL.CLASSIFICATION.HEADS` in the YAML, specifying the `TYPE` for each task key. Internally, they use shared `nn.Linear` layers (one per task level) for efficiency, especially with DDP.

### 2.1 Hierarchical Softmax (`models.heads.HierarchicalSoftmaxHead`)

*   **Concept:** Efficiently approximates hierarchical softmax using matrix-based refinement. It calculates base logits per level and refines lower levels based on parent probabilities propagated via the `TaxonomyTree`.
*   **Implementation:**
    *   Uses shared `nn.Linear` classifiers (one per task level).
    *   Uses pre-computed hierarchy matrices `H` from `TaxonomyTree`.
    *   `forward(x)` computes `Z_base`, then refines top-down: `Z_refined[L+1] = Z_base[L+1] + log(softmax(Z_refined[L]) @ H[L, L+1] + epsilon)`.
    *   Returns only the refined logits for its primary associated `task_key`.
*   **Configuration (`MODEL.CLASSIFICATION.HEADS.<task_key>`):**
    ```yaml
    TYPE: "HierarchicalSoftmax"
    # IN_FEATURES: <int> # Automatically set based on backbone output
    USE_BIAS: True     # Optional: Bias for internal linear layers (default: True)
    ```
*   **Hyperparameters & Tuning:**
    *   `USE_BIAS`: Standard bias term for linear layers. Usually kept `True`.
    *   **Interaction with Loss:** Best used with standard cross-entropy or label smoothing on the *output* logits. The hierarchical structure is baked into the *logit calculation*, not the loss function itself (unlike true path-product HSM).
    *   **GradNorm:** May benefit from GradNorm, as different levels might have varying gradient magnitudes. Experimentation needed.

### 2.2 Conditional Classifier (`models.heads.ConditionalClassifierHead`)

*   **Concept:** Models predictions top-down, conditioning lower levels on parent level predictions using configurable routing strategies.
*   **Implementation:**
    *   Uses shared `nn.Linear` classifiers (one per task level).
    *   Uses pre-computed hierarchy matrices `H` from `TaxonomyTree`.
    *   `forward(x)` computes `Z_base`, then refines top-down:
        *   Calculates parent routing probabilities `P_routing` based on `routing_strategy` (soft, hard, Gumbel) and `temperature`.
        *   Calculates child prior `P_child_prior = P_routing @ H[L, L+1]`.
        *   Refines child logits: `Z_refined[L+1] = Z_base[L+1] + log(P_child_prior + epsilon)`.
    *   Returns only the refined logits for its primary associated `task_key`.
*   **Configuration (`MODEL.CLASSIFICATION.HEADS.<task_key>`):**
    ```yaml
    TYPE: "ConditionalClassifier"
    # IN_FEATURES: <int> # Automatically set
    ROUTING_STRATEGY: "soft" # Optional: 'soft', 'hard', 'gumbel' (default: 'soft')
    TEMPERATURE: 1.0         # Optional: For 'soft'/'gumbel' routing (default: 1.0)
    USE_BIAS: True         # Optional: Bias for internal linear layers (default: True)
    ```
*   **Hyperparameters & Tuning:**
    *   `ROUTING_STRATEGY`:
        *   `'soft'`: Default, fully differentiable, uses softmax probabilities for weighting. Good starting point.
        *   `'gumbel'`: Uses Gumbel-Softmax during training for differentiable sampling of discrete paths, potentially encouraging more focused predictions. Might require tuning `TEMPERATURE`.
        *   `'hard'`: Uses `argmax` during inference (not training). Deterministic path selection.
    *   `TEMPERATURE`: Controls sharpness of softmax/Gumbel-softmax in routing. Lower values approach hard routing, higher values soften probabilities. Default `1.0` is standard. Tuning might be needed, especially for `'gumbel'`.
    *   `USE_BIAS`: Standard bias term. Usually kept `True`.
    *   **Interaction with Loss:** Similar to HSM Head, typically used with standard cross-entropy or label smoothing on the output logits.
    *   **GradNorm:** Likely benefits from GradNorm due to the multi-level structure and potential for varying gradient scales across levels. Using GradNorm (as in the `blade_amphibia_mini_0_conditional.yaml` config) is a sensible default choice.

## 3. Integration & Training Considerations

*   **Model Forward Pass:** The main model (`mFormerV0`) iterates through its `self.head` ModuleDict, calling each head instance. Each hierarchical head instance performs internal multi-level calculations but returns only the logits for its primary task.
*   **Loss Calculation:** `weighted_hierarchical_loss` receives the dictionary of final logits (one entry per task key) and computes the loss using the configured criteria (e.g., `TaxonomyAwareLabelSmoothingCE`, `CrossEntropyLoss`).
*   **GradNorm with Hierarchical Heads:** When using GradNorm, the `EXCLUDE_CONFIG` should exclude parameters *within* the head modules (typically matching `"head."`). Because the internal `level_classifiers` are shared and used by *all* head instances during the model's forward pass, they *should* receive gradients correctly when the total loss is computed, making them part of the GradNorm backbone calculation (unless explicitly excluded by name/type filters).
*   **Taxonomy Smoothing and Hierarchical Heads:** It's valid to combine Taxonomy-Aware Label Smoothing (as the loss criterion) with either `HierarchicalSoftmaxHead` or `ConditionalClassifierHead`. The head produces refined logits, and the loss function then compares these logits against the taxonomically smoothed ground truth distribution.

## 4. Mixup Configuration with Hierarchical Classifications

When using Selective Mixup with hierarchical classification tasks, special configuration is needed to maintain the integrity of taxonomic relationships.

### 4.1 Best Practices for Mixup with Taxonomy-Aware Losses

To ensure targets maintain "hard label" properties after mixup:

1. **Set `MIXUP.GROUP_LEVELS` to the lowest-rank taxonomic level** in your task key hierarchy
   - Usually `['taxa_L10']` for species-level classification
   - This ensures mixed pairs share identical labels for ALL hierarchical levels

2. **Always set `SCHEDULE.MIXUP.EXCLUDE_NULL_SAMPLES=True`**
   - Prevents mixing samples with unknown classifications
   - Maintains certainty in the training targets

3. **Use a taxonomic task key order where `GROUP_LEVELS` contains the leaf node**
   - For example, with task keys `['taxa_L10', 'taxa_L20', 'taxa_L30', 'taxa_L40']`:
   - If `taxa_L10` is the species level, it should be the lowest-rank task
   - Setting `GROUP_LEVELS=['taxa_L10']` ensures hierarchical consistency

### 4.2 Why These Settings Work

Under these specific conditions, when mixup is applied:
1. Only samples with identical species labels are mixed
2. Due to taxonomic hierarchy, these samples must also share the same labels for all higher ranks
3. Therefore, the mixed `targets` dictionary still represents a single ground truth class for each level
4. Loss functions like `TaxonomyAwareLabelSmoothingCE` can safely convert the 2D mixed targets to 1D indices with `argmax()`

### 4.3 Example Configuration

```yaml
SCHEDULE:
  MIXUP:
    GROUP_LEVELS: ['taxa_L10']  # Critical: Set to lowest taxonomic rank
    EXCLUDE_NULL_SAMPLES: True  # Critical: Prevent mixing with unknown labels
    ENABLED: True               # Master switch for mixup
    PROB_SCHEDULE:              # Optional probability scheduling
      TYPE: 'linear'            # Schedule type (constant, linear, cosine, etc.)
      START_PROB: 0.8           # Initial probability
      END_PROB: 0.2             # Final probability
```

### 4.4 Warning Signs of Incorrect Configuration

Be cautious of these configurations:

1. **Setting `GROUP_LEVELS` to a higher taxonomic rank than your lowest task**
   - Example: `GROUP_LEVELS=['taxa_L20']` with task keys including `taxa_L10`
   - WARNING: This allows mixing samples with different species-level labels

2. **Setting `EXCLUDE_NULL_SAMPLES=False`**
   - WARNING: This allows mixing samples with unknown labels, creating problematic "partial" targets

3. **Using multiple levels in `GROUP_LEVELS`**
   - Example: `GROUP_LEVELS=['taxa_L10', 'taxa_L20']`
   - This combines samples that match on EITHER level, breaking hierarchical consistency

## 5. Distributed Training (DDP) and GradNorm with Hierarchical Heads

A critical interaction occurs when using **hierarchical heads with shared internal classifiers** (like `HierarchicalSoftmaxHead` and `ConditionalClassifierHead`) in combination with **Distributed Data Parallel (DDP)** and **GradNorm**.

### 5.1 The Challenge: Unused Parameters during GradNorm Backward

-   **Shared Classifiers:** Both `HierarchicalSoftmaxHead` and `ConditionalClassifierHead` use an internal `nn.ModuleDict` (`level_classifiers` or `task_classifiers`) containing one `nn.Linear` layer per taxonomic level (e.g., for `taxa_L10`, `taxa_L20`, etc.). These linear layers are shared across all head instances responsible for different output tasks.
-   **GradNorm Re-Forward:** The GradNorm algorithm performs separate forward and backward passes for each task to calculate gradients with respect to the shared backbone. For a given task `i`, it computes `loss_i` using only `output[task_i]` and then calls `loss_i.backward()`.
-   **DDP Expectation:** DDP synchronizes gradients during the backward pass. It expects that *all* parameters involved in the forward pass calculation of `loss_i` will receive a gradient during `loss_i.backward()`.
-   **The Conflict:** When calculating `loss_i` (e.g., for `taxa_L20`), the forward pass through the hierarchical head likely uses parameters from the shared `level_classifiers` of *other* levels (e.g., `taxa_L10`, `taxa_L30`) to compute internal probabilities or priors. However, when `loss_i.backward()` is called, only the parameters *directly contributing* to `loss_i` (the backbone and the specific path through the head for `taxa_L20`, including *some* of the shared classifiers) receive gradients. The parameters of the other shared classifiers, although used in the forward pass, do *not* receive gradients in this specific backward step.
-   **DDP Error:** In the *next* forward pass, DDP detects that parameters from unused levels (e.g., `head.taxa_L10.level_classifiers.taxa_L30.weight`) did not have their gradients computed and reduced, leading to a `RuntimeError: Expected to have finished reduction in the prior iteration...`.

### 5.2 Solution: `find_unused_parameters=True`

The standard and recommended solution provided by PyTorch for this scenario is to initialize DDP with `find_unused_parameters=True`.

```python
# Inside main.py

find_unused = config.MODEL.FIND_UNUSED_PARAMETERS # Default from config
# FORCE True if using GradNorm with hierarchical heads
if config.LOSS.GRAD_WEIGHTING.TASK.TYPE == 'gradnorm' and \
   any(h.get("TYPE", "").startswith(("Hierarchical", "Conditional"))
       for h in config.MODEL.CLASSIFICATION.HEADS.values()):
    if not find_unused:
        logger.warning("GradNorm + Hierarchical Heads require find_unused_parameters=True for DDP. Overriding.")
    find_unused = True

model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
```

-   **Effect:** This tells DDP to dynamically detect which parameters were *actually* used in the backward pass and only wait for those gradients, ignoring parameters that didn't receive gradients (like the unused shared classifiers during a specific GradNorm task backward).
-   **Trade-off:** Setting `find_unused_parameters=True` introduces a small performance overhead during the backward pass as DDP needs to analyze the computation graph. However, it ensures correctness when using complex architectures like shared-parameter hierarchical heads with GradNorm.
-   **Current Implementation:** linnaeus automatically forces this setting in `main.py` when `LOSS.GRAD_WEIGHTING.TASK.TYPE` is `'gradnorm'` to prevent the DDP error.

### 5.3 Why Filtering Isn't Enough

While the `LOSS.GRAD_WEIGHTING.TASK.EXCLUDE_CONFIG` correctly identifies *all* head parameters (including shared ones) to exclude them from the *GradNorm L2 norm calculation*, it does *not* prevent DDP from seeing these parameters being used during the forward pass of the GradNorm loop. DDP's error occurs because of the mismatch between parameters used in *forward* versus those receiving gradients in the *selective backward* inherent to GradNorm's per-task gradient calculation.

Therefore, even with perfect filtering for the GradNorm *weight calculation*, `find_unused_parameters=True` is necessary for DDP compatibility when using GradNorm with these specific hierarchical head structures.

## 6. GradNorm Configuration Requirements

GradNorm has specific configuration requirements to work correctly, especially in distributed training environments.

### 6.1 DDP Configuration Requirements

As noted in section 5.2, using GradNorm with DDP requires setting `find_unused_parameters=True` in the DDP initialization. The linnaeus framework automatically forces this setting when GradNorm is enabled to prevent potential NCCL timeouts and synchronization errors:

```python
# In main.py
find_unused = config.MODEL.FIND_UNUSED_PARAMETERS  # Default from config

# Force find_unused_parameters=True for all GradNorm usage
if config.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ENABLED:
    if not find_unused:
        logger.warning("Forcing find_unused_parameters=True for DDP because GradNorm is enabled")
    find_unused = True
    
model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
```

This setting is critical for preventing deadlocks during distributed communication, especially when GradNorm performs separate forward and backward passes for each task.

### 6.2 Accumulation Steps Constraint

GradNorm's update interval must be compatible with the gradient accumulation setting:

```yaml
LOSS:
  GRAD_WEIGHTING:
    TASK:
      GRADNORM_ENABLED: True
      TYPE: 'gradnorm'
      UPDATE_INTERVAL: 50  # Must be >= TRAIN.ACCUMULATION_STEPS
      GRADNORM_WARMUP_STEPS: 2000
      
TRAIN:
  ACCUMULATION_STEPS: 4  # Must be <= GRAD_WEIGHTING.TASK.UPDATE_INTERVAL
```

**Important constraint:** GradNorm's `UPDATE_INTERVAL` must be greater than or equal to `TRAIN.ACCUMULATION_STEPS`. The framework will validate this constraint at startup and raise a configuration error if violated.

**Why this matters:**
- GradNorm updates need to happen at optimizer step boundaries
- Multiple GradNorm updates within a single optimizer step would lead to gradient inconsistency
- The implementation ensures GradNorm only runs once per optimizer step (and only when triggered by its update interval)

### 6.3 Distributed Synchronization

In distributed training, processes must stay synchronized during GradNorm's task-by-task processing:

- The framework inserts explicit distributed barriers between task processing in GradNorm
- This ensures all ranks complete processing one task before any rank moves to the next task
- Without these barriers, ranks could desynchronize during GradNorm's sequential task processing, potentially leading to NCCL timeouts or deadlocks

These constraints and safeguards help ensure GradNorm functions correctly and reliably, especially in more complex distributed training scenarios.

### 6.4 GradNorm Mode for Hierarchical Heads

Hierarchical heads can optionally operate in a special **GradNorm mode** during
GradNorm's internal re-forward steps. When enabled, the heads bypass their
hierarchical refinement and use only their direct linear classifier for the
task being processed. This avoids vanishing gradients for child tasks when
GradNorm measures per-task gradient norms.

Enable this behavior via the configuration flag:

```yaml
LOSS:
  GRAD_WEIGHTING:
    TASK:
      USE_LINEAR_HEADS_FOR_GRADNORM_REFORWARD: True  # Default
```

This setting does not affect the main training forward pass. It is only applied
while GradNorm computes its task weights.
