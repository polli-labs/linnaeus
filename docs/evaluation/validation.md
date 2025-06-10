<WARNING> Possibly out of data, simplified some phase metrics (for various val phases/types) since drafting, unclear if we reviewed/revised this doc </WARNING>

# Validation in linnaeus

## Overview

This document describes the validation system in linnaeus, including the different types of validation, how to configure validation schedules, and best practices for effective model evaluation.

Validation in linnaeus provides critical insight into model performance on unseen data, helping to detect overfitting and assess model generalization capabilities. The system supports several validation modes to comprehensively evaluate model performance under different conditions.

The validation implementation can be found in `linnaeus/validation.py`, which contains the core validation functions that are called from the main training loop.

## Validation Types

linnaeus supports three main types of validation:

1. **Standard Validation**: Evaluates the model on the validation dataset with all metadata available
2. **Mask Meta Validation**: Evaluates the model with all metadata masked (hidden) to assess performance when metadata is unavailable
3. **Partial Mask Meta Validation**: Evaluates the model with specific combinations of metadata components masked to understand the impact of individual metadata elements

### Standard Validation

Standard validation evaluates the model on validation data without modifying metadata availability. This represents the ideal scenario where all inputs are available as expected.

### Mask Meta Validation

Mask meta validation completely masks (hides) all metadata components, forcing the model to rely solely on the image features. This validation type helps assess:

1. How well the model performs when no metadata is available
2. The overall impact of metadata on model performance
3. The model's ability to generalize from visual features alone

### Partial Mask Meta Validation

Partial mask meta validation selectively masks specific metadata components or combinations of components. This allows for detailed analysis of:

1. The impact of individual metadata components on model performance
2. The model's ability to compensate for missing specific metadata elements
3. Critical metadata dependencies in the model

## Validation Configuration

Validation is configured in the experiment configuration file under the `SCHEDULE.VALIDATION` section.

### Standard Validation Configuration

```yaml
SCHEDULE:
  VALIDATION:
    # Step-based validation (every 2% of total steps)
    INTERVAL_FRACTION: 0.02
    INTERVAL_STEPS: 0       # Set to 0 when using INTERVAL_FRACTION
    INTERVAL_EPOCHS: 0      # Set to 0 when using step-based validation
```

### Mask Meta Validation Configuration

```yaml
SCHEDULE:
  VALIDATION:
    # Mask meta validation (every 2% of total steps)
    MASK_META_INTERVAL_FRACTION: 0.02
    MASK_META_INTERVAL_STEPS: 0   # Set to 0 when using MASK_META_INTERVAL_FRACTION
    MASK_META_INTERVAL_EPOCHS: 0  # Set to 0 when using step-based validation
```

### Partial Mask Meta Validation Configuration

Partial mask meta validation can be scheduled in three ways:

1. **At a specific point in training** by setting `INTERVAL_FRACTION` to a value like 0.5 (run once at 50% of training)
2. **At regular intervals** by setting `INTERVAL_STEPS` or `INTERVAL_EPOCHS` to run periodically
3. **At the final epoch only** using the `FINAL_EPOCH` configuration

Here's an example configuration for running partial mask meta validation at the 50% mark of training:

```yaml
SCHEDULE:
  VALIDATION:
    # Partial mask meta validation configuration
    PARTIAL_MASK_META:
      ENABLED: True
      INTERVAL_FRACTION: 0.5   # Run once at 50% of training
      INTERVAL_STEPS: 0        # Set to 0 when using INTERVAL_FRACTION
      INTERVAL_EPOCHS: 0       # Set to 0 when using INTERVAL_FRACTION
      WHITELIST:               # Component combinations to validate with
        - ["TEMPORAL"]
        - ["SPATIAL"] 
        - ["ELEVATION"]
        - ["TEMPORAL", "SPATIAL"]
    
    # Final epoch exhaustive validation (optional)
    FINAL_EPOCH:
      EXHAUSTIVE_PARTIAL_META_VALIDATION: True
      EXHAUSTIVE_META_COMPONENTS:  # All components to generate combinations from
        - "TEMPORAL"
        - "SPATIAL"
        - "ELEVATION"
```

When you set `INTERVAL_FRACTION` to a specific point (e.g., 0.5 for 50% of training), the validation will run only once at that point, not at regular intervals. This is useful for checking model performance with partially masked metadata at a specific training milestone.

For periodic validation, use either `INTERVAL_EPOCHS` or `INTERVAL_STEPS` instead:

```yaml
PARTIAL_MASK_META:
  ENABLED: True
  INTERVAL_EPOCHS: 2           # Run every 2 epochs
  INTERVAL_FRACTION: 0         # Set to 0 when using INTERVAL_EPOCHS
  INTERVAL_STEPS: 0            # Set to 0 when using INTERVAL_EPOCHS
```

## Scheduling Validation Runs

Validation runs can be scheduled based on steps or epochs. However, validations only execute at epoch boundaries (after a complete epoch finishes), even when using step-based scheduling.

### Validation Scheduling Methods

You can schedule validation using one of three methods:

1. **Epoch-Based Scheduling**: Based on completed epochs
2. **Step-Based Scheduling**: Based on absolute step counts
3. **Fraction-Based Scheduling**: Based on fraction of total training steps

#### Epoch-Based Scheduling

```yaml
INTERVAL_EPOCHS: 1  # Validate every epoch
INTERVAL_STEPS: 0   # Must be 0 when using epoch-based scheduling
INTERVAL_FRACTION: None  # Must be None when using epoch-based scheduling
```

**Important Note on Epoch-Based Scheduling Behavior:**

When using epoch-based intervals, the system follows these rules:
- If `INTERVAL_EPOCHS = 1`: Validation runs at epochs 0, 1, 2, 3, ...
- If `INTERVAL_EPOCHS > 1`: Validation runs at epochs N, 2N, 3N, ... (skipping epoch 0)
- This applies to all epoch-based intervals (standard validation, mask meta validation, partial mask meta validation, and checkpointing)

#### Step-Based Scheduling

```yaml
INTERVAL_STEPS: 1000  # Validate every 1000 steps
INTERVAL_EPOCHS: 0    # Must be 0 when using step-based scheduling
INTERVAL_FRACTION: None  # Must be None when using step-based scheduling
```

#### Fraction-Based Scheduling

```yaml
INTERVAL_FRACTION: 0.05  # Validate every 5% of total steps
INTERVAL_STEPS: 0       # Must be 0 when using fraction-based scheduling
INTERVAL_EPOCHS: 0      # Must be 0 when using fraction-based scheduling
```

When using `INTERVAL_FRACTION`, the system calculates the corresponding step interval during initialization. For example, with `INTERVAL_FRACTION: 0.05` and 100,000 total steps, validation would be scheduled every 5,000 steps.

### Important Note on Validation Timing and Execution Guarantee

A key detail to understand about validation scheduling is that actual validation runs occur at epoch boundaries, even when using step-based scheduling. This is because validation requires processing the entire validation dataset, which happens most efficiently at epoch boundaries.

The validation system provides a **single execution guarantee** for each configured step-based trigger. This means each validation trigger happens exactly once at the first epoch boundary after the step threshold is reached.

Here's how this works in practice:

1. **Epoch-based scheduling**: This is straightforward - validation runs after the specified number of epochs
   
2. **Step-based scheduling**: The system tracks the step count, but validation runs are actually triggered at the first epoch boundary after hitting the step threshold
   
   - When a step threshold is reached, validation is triggered at the next epoch boundary
   - Each step threshold triggers exactly one validation run (single execution guarantee)
   - The system keeps track of which thresholds have already triggered validation to avoid duplicates

3. **Fraction-based scheduling**: Similar to step-based, but using a fraction of total_steps
   - The fraction is converted to a specific step count during initialization
   - Validation then follows the step-based rules above with the same single execution guarantee

This is why the "Schedule Summary" displayed at the start of training might show a validation interval in steps, but the actual timing will be aligned with epoch boundaries.

**Example**: With 5,000 steps per epoch and validation intervals at steps 4000 and 8000, validation would actually run after epoch 1 (at step 5000) for the first trigger and after epoch 2 (at step 10000) for the second trigger.

For short training runs (few epochs), you may need to adjust validation fractions to get the desired validation frequency.

### Choosing the Right Scheduling Method

- **Epoch-Based**: Most intuitive when you want validation to occur at dataset boundaries
- **Step-Based**: Useful for precise control with large datasets
- **Fraction-Based**: Best for configurations that should scale with dataset size

Note: You must choose only one scheduling method for each validation type.

## Exhaustive Final Validation

linnaeus supports an exhaustive validation at the end of training that tests all possible combinations of masked metadata components:

```yaml
FINAL_EPOCH:
  EXHAUSTIVE_PARTIAL_META_VALIDATION: True
  EXHAUSTIVE_META_COMPONENTS:  # Components to generate combinations from
    - "TEMPORAL"
    - "SPATIAL"
    - "ELEVATION"
```

This generates and evaluates all possible combinations (except the empty set) of the specified components. For example, with three components, it tests seven combinations (3 single-component masks, 3 two-component masks, and 1 all-component mask).

## Validation Metrics and Masking

### Metrics

Validation results are tracked separately for each validation type and component combination:

- `val_loss`, `val_accuracy`: Standard validation metrics
- `val_mask_meta_loss`, `val_mask_meta_accuracy`: Metrics with all metadata masked
- `val_mask_TEMPORAL_loss`, `val_mask_TEMPORAL_accuracy`: Metrics with only TEMPORAL component masked
- `val_mask_TEMPORAL_SPATIAL_loss`, `val_mask_TEMPORAL_SPATIAL_accuracy`: Metrics with both TEMPORAL and SPATIAL components masked

These metrics allow for comprehensive analysis of how different metadata components impact model performance.

### Null Masking During Validation

It's important to note that **null masking is always disabled during validation**, regardless of the training schedule. This ensures that all validation runs represent realistic deployment conditions where all data contributes to predictions.

While normal training may use scheduled null masking to stabilize learning, validation always includes all data points to provide an accurate performance assessment.

## Best Practices

1. **Standard and Mask Meta Validation**: Always include both standard and mask meta validation to understand metadata impact.

2. **Partial Mask Meta Validation**: Include partial mask meta validation with realistic component combinations that match deployment scenarios.

3. **Validation Frequency**: For short training runs (few epochs), increase validation fractions to avoid validating after every epoch.

4. **Exhaustive Validation**: Use exhaustive validation at the end of training to understand component importance, but be mindful that this generates many validation passes (2^n - 1, where n is the number of components).

5. **Resource Management**: Partial mask meta validations are resource-intensive as they run a full validation for each component combination. Adjust frequency appropriately.

6. **Metrics Analysis**: Compare metrics across validation types to understand the contribution of each metadata component to model performance.

See also:
- [Training Scheduling](../training/scheduling.md) for more details on the scheduling system
- [Meta Masking](../training/meta_masking.md) for information on metadata masking during training