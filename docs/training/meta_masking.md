# Metadata Masking in linnaeus

## Overview

The metadata masking system in linnaeus is designed to improve model robustness when dealing with partial or missing metadata. Real-world deployment scenarios often involve incomplete metadata availability, and our models need to handle these scenarios gracefully.

## Basic Metadata Masking

The basic metadata masking schedule randomly masks all metadata components during training with a configurable probability. This probability can be scheduled to decrease over time, allowing the model to gradually rely more on metadata as training progresses.

```yaml
SCHEDULE:
  META_MASKING:
    ENABLED: True
    START_PROB: 1.0  # Start with 100% probability of masking all metadata
    END_PROB: 0.05   # End with 5% probability of masking all metadata
    END_FRACTION: 0.3  # Reach the END_PROB at 30% of training
```

This approach helps models learn to make predictions when metadata is entirely unavailable. However, it doesn't address cases where only some metadata components are missing.

## Granular Metadata Masking

The granular metadata masking feature extends the basic approach to handle partial metadata masking. Instead of masking all metadata or none, it allows selectively masking specific components or combinations of components during training and validation.

### Configuration

To configure granular metadata masking, add a `PARTIAL` section to the `META_MASKING` configuration:

```yaml
SCHEDULE:
  META_MASKING:
    ENABLED: True
    START_PROB: 1.0
    END_PROB: 0.05
    END_FRACTION: 0.3
    
    # Partial meta masking configuration
    PARTIAL:
      ENABLED: True  # Enable partial meta masking
      START_FRACTION: 0.1  # Start partial masking after 10% of training
      END_FRACTION: 0.9  # Continue until 90% of training
      
      # Probabilistic partial meta masking (new capability)
      START_PROB: 0.01  # Initial probability of applying partial meta masking (1%)
      END_PROB: 0.7     # Final probability of applying partial meta masking (70%)
      PROB_END_FRACTION: 0.5  # Reach END_PROB at 50% of training
      
      WHITELIST:  # Components to selectively mask
        - ["TEMPORAL"]
        - ["SPATIAL"]
        - ["ELEVATION"]
        - ["TEMPORAL", "SPATIAL"]
        - ["TEMPORAL", "ELEVATION"]
        - ["SPATIAL", "ELEVATION"]
      WEIGHTS: [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]  # Optional weights for component combinations
```

The **whitelist** defines which combinations of metadata components should be masked. Each entry is a list of component names. When partial masking is active, a random combination from the whitelist is chosen, and those specific components are masked for each sample. The WEIGHTS parameter (optional) allows you to control the probability distribution for selecting combinations.

The **probability scheduling** parameters (`START_PROB`, `END_PROB`, and `PROB_END_FRACTION` or `PROB_END_STEPS`) control the likelihood of applying partial masking to samples that aren't already fully masked. This ensures that some proportion of samples retain all their metadata during training, which is important for model performance.

### Training Behavior

During training, for each batch:

1. With probability `meta_mask_prob`, all metadata is masked (global meta masking)
2. Otherwise, for each remaining sample:
   - With probability `partial_meta_mask_prob` (scheduled from START_PROB to END_PROB):
     - A random combination from the whitelist is chosen
     - Only those specific metadata components are masked
   - Otherwise (with probability `1 - partial_meta_mask_prob`):
     - All metadata is retained for that sample

This approach ensures the model learns to handle various realistic partial-metadata states during training while still having some samples with full metadata available.

## Validation with Partial Masking

To evaluate model performance with specific partial metadata combinations, you can configure validation passes that mask particular components:

```yaml
SCHEDULE:
  VALIDATION:
    # Partial meta mask validation configuration
    PARTIAL_MASK_META:
      ENABLED: True
      STEP_FRACTION: 0.05  # Run every 5% of total steps
      WHITELIST:  # Component combinations to validate with
        - ["TEMPORAL"]
        - ["SPATIAL"]
        - ["ELEVATION"]
        - ["TEMPORAL", "SPATIAL"]
    
    # Final epoch exhaustive validation
    FINAL_EPOCH:
      EXHAUSTIVE_PARTIAL_META_VALIDATION: True
      EXHAUSTIVE_META_COMPONENTS:  # All components to generate combinations from
        - "TEMPORAL"
        - "SPATIAL"
        - "ELEVATION"
```

This configuration:
1. Runs periodic validation passes with each combination in the whitelist
2. Optionally performs an exhaustive validation at the final epoch, testing all possible combinations of the specified components (except the full set, which is redundant with standard masking)

## Metrics Tracking

Partial mask validation results are tracked with phase names based on the masked components:
- `val_mask_TEMPORAL` for validation with only TEMPORAL masked
- `val_mask_TEMPORAL_SPATIAL` for validation with both TEMPORAL and SPATIAL masked

These metrics help understand how each metadata component impacts model performance.

## Usage Guidelines

### Selecting Component Combinations

Choose whitelist combinations that reflect real-world deployment scenarios. For example:
- If temporal data is often unavailable, include `["TEMPORAL"]`
- If spatial and elevation data tend to be missing together, include `["SPATIAL", "ELEVATION"]`

### Weighting Combinations

Use the WEIGHTS parameter to emphasize more common real-world scenarios. For example, if missing temporal data is twice as common as missing spatial data, use weights like:
```yaml
WHITELIST:
  - ["TEMPORAL"]
  - ["SPATIAL"]
WEIGHTS: [2.0, 1.0]
```

### Validation Strategy

Configure periodic validation with the most important metadata combinations, and use exhaustive validation at the end of training to get a complete understanding of component importance.