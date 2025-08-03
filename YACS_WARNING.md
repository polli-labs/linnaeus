# ⚠️ CRITICAL: YACS Config List Behavior

## The Problem
YACS **replaces entire lists** when merging configs. It does NOT merge element-wise.

## Example
```yaml
# Base config
MODEL:
  DEPTHS: [3, 3, 27, 3]

# Your override
MODEL:  
  DEPTHS: [5, 5]  # ❌ WRONG - This deletes the last 2 elements!

# Correct override
MODEL:
  DEPTHS: [5, 5, 27, 3]  # ✅ Must specify ALL elements
```

## Rules
1. **ALWAYS specify complete lists** in override configs
2. **NEVER use partial lists** 
3. **CHECK the model logs** to verify what config it received

## Affected Parameters
- `MODEL.CONVNEXT_STAGES.DEPTHS` - Must be length 4
- `MODEL.CONVNEXT_STAGES.DIMS` - Must be length 4  
- `MODEL.ROPE_STAGES.DEPTHS` - Must be length 2
- `MODEL.ROPE_STAGES.DIMS` - Must be length 2

## Impact
This behavior can cause:
- Silent architectural misconfigurations
- Accidental addition/removal of layers
- Performance regressions (46.8% observed in one case)
- Wasted compute resources on failed experiments

## Validation
As of August 2024, mFormerV1 includes validation that will raise an error if lists have incorrect lengths:
```
ValueError: CONVNEXT_STAGES.DEPTHS must have exactly 4 elements, got 2.
Current value: [3, 3]
Due to YACS list replacement behavior, you MUST specify all 4 values.
Example for ConvNeXt-S: [3, 3, 27, 3], not just [3, 3].
```

## Further Details
See `work/bugs/inbox/P0/yacs_list_inheritance/` for comprehensive analysis and examples.