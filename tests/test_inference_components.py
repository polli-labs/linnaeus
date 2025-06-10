import torch
import pytest
from datetime import datetime
import logging

from linnaeus.inference.preprocessing import preprocess_metadata_batch
from linnaeus.inference.config import MetaConfig

# Configure logging for capturing warnings
logger = logging.getLogger("linnaeus.inference")

def test_preprocess_metadata(caplog):
    # Mock MetaConfig: Enable all components for comprehensive testing
    meta_cfg = MetaConfig(
        use_geolocation=True,
        use_temporal=True,
        temporal_use_julian_day=False, # Using month_of_year
        temporal_use_hour=True,      # Using hour_of_day
        use_elevation=True,
        elevation_scales=[100.0, 1000.0] # Using two scales
    )
    # Expected length: geo(3) + temporal_month_year(2) + temporal_hour(2) + elev(2*2=4) = 3 + 2 + 2 + 4 = 11
    expected_length = 3 + 2 + 2 + (2 * len(meta_cfg.elevation_scales))

    # Test Case 1: Full Metadata
    metadata_list_full = [{
        "lat": 40.7128,
        "lon": -74.0060,
        "datetime_utc": datetime(2023, 7, 15, 14, 30, 0),
        "elevation_m": 10.0,
    }]
    caplog.clear()
    output_full = preprocess_metadata_batch(metadata_list_full, meta_cfg, expected_length)
    assert isinstance(output_full, torch.Tensor), "Output should be a torch.Tensor"
    assert output_full.shape == (1, expected_length), f"Shape mismatch for full metadata. Expected (1, {expected_length}), got {output_full.shape}"
    assert not torch.all(output_full == 0).item(), "Full metadata output should not be all zeros"
    # Check that no warnings were logged for valid data
    assert not any(record.levelno == logging.WARNING for record in caplog.records), "No warnings should be logged for valid full metadata"


    # Test Case 2: Partial Metadata (missing elevation_m)
    metadata_list_partial = [{
        "lat": 40.7128,
        "lon": -74.0060,
        "datetime_utc": datetime(2023, 7, 15, 14, 30, 0),
        # elevation_m is missing
    }]
    caplog.clear()
    output_partial = preprocess_metadata_batch(metadata_list_partial, meta_cfg, expected_length)
    assert output_partial.shape == (1, expected_length), "Shape mismatch for partial metadata"
    # Elevation features are the last 2 * len(meta_cfg.elevation_scales) = 4 features
    elevation_slice = output_partial[0, - (2 * len(meta_cfg.elevation_scales)):]
    assert torch.all(elevation_slice == 0).item(), "Elevation slice should be all zeros when elevation_m is missing"
    # Check other parts are not all zero
    non_elevation_slice = output_partial[0, :-(2 * len(meta_cfg.elevation_scales))]
    assert not torch.all(non_elevation_slice == 0).item(), "Non-elevation part should not be all zeros for partial metadata"
    assert not any(record.levelno == logging.WARNING for record in caplog.records), "No warnings should be logged for missing (but valid) elevation"


    # Test Case 3: No Metadata (empty dict)
    metadata_list_none = [{}]
    caplog.clear()
    output_none = preprocess_metadata_batch(metadata_list_none, meta_cfg, expected_length)
    assert output_none.shape == (1, expected_length), "Shape mismatch for no metadata"
    assert torch.all(output_none == 0).item(), "Output should be all zeros for empty metadata dict"
    assert not any(record.levelno == logging.WARNING for record in caplog.records), "No warnings should be logged for empty metadata dict"


    # Test Case 4: Invalid Metadata (non-numeric lat)
    metadata_list_invalid_lat = [{
        "lat": "not-a-number", # Invalid
        "lon": -74.0060,
        "datetime_utc": datetime(2023, 7, 15, 14, 30, 0),
        "elevation_m": 10.0,
    }]
    caplog.clear()
    output_invalid_lat = preprocess_metadata_batch(metadata_list_invalid_lat, meta_cfg, expected_length)
    assert output_invalid_lat.shape == (1, expected_length), "Shape mismatch for invalid lat metadata"
    # Geolocation features are the first 3
    geo_slice_invalid = output_invalid_lat[0, :3]
    assert torch.all(geo_slice_invalid == 0).item(), "Geolocation slice should be all zeros for invalid lat"
    # Check that a warning was logged
    assert any("Invalid lat/lon values" in record.message for record in caplog.records if record.levelno == logging.WARNING), "Warning for invalid lat should be logged"


    # Test Case 5: Invalid Metadata (non-numeric elevation_m)
    metadata_list_invalid_elev = [{
        "lat": 40.7128,
        "lon": -74.0060,
        "datetime_utc": datetime(2023, 7, 15, 14, 30, 0),
        "elevation_m": "not-a-number", # Invalid
    }]
    caplog.clear()
    output_invalid_elev = preprocess_metadata_batch(metadata_list_invalid_elev, meta_cfg, expected_length)
    assert output_invalid_elev.shape == (1, expected_length), "Shape mismatch for invalid elevation metadata"
    # Elevation features are the last 4
    elevation_slice_invalid = output_invalid_elev[0, - (2 * len(meta_cfg.elevation_scales)):]
    assert torch.all(elevation_slice_invalid == 0).item(), "Elevation slice should be all zeros for invalid elevation_m"
    # Check that a warning was logged
    assert any("Invalid elevation value" in record.message for record in caplog.records if record.levelno == logging.WARNING), "Warning for invalid elevation should be logged"

    # Test Case 6: Invalid datetime string
    metadata_list_invalid_dt_str = [{
        "lat": 40.7128,
        "lon": -74.0060,
        "datetime_utc": "not-a-valid-iso-datetime-string",
        "elevation_m": 10.0,
    }]
    caplog.clear()
    output_invalid_dt_str = preprocess_metadata_batch(metadata_list_invalid_dt_str, meta_cfg, expected_length)
    assert output_invalid_dt_str.shape == (1, expected_length), "Shape mismatch for invalid datetime string"
    # Temporal features are after geo (3) and before elevation (4), so indices 3, 4, 5, 6 (total 4 features)
    temporal_slice_invalid_dt = output_invalid_dt_str[0, 3:3+4]
    assert torch.all(temporal_slice_invalid_dt == 0).item(), "Temporal slice should be all zeros for invalid datetime string"
    assert any("Invalid datetime string" in record.message for record in caplog.records if record.levelno == logging.WARNING), "Warning for invalid datetime string should be logged"

    # Test Case 7: Batch of multiple metadata entries
    metadata_list_batch = [
        { # Full
            "lat": 40.7128, "lon": -74.0060, "datetime_utc": datetime(2023, 7, 15, 14, 30, 0), "elevation_m": 10.0,
        },
        { # Missing elevation
            "lat": 34.0522, "lon": -118.2437, "datetime_utc": datetime(2023, 8, 1, 10, 0, 0),
        },
        {} # Empty
    ]
    caplog.clear()
    output_batch = preprocess_metadata_batch(metadata_list_batch, meta_cfg, expected_length)
    assert output_batch.shape == (len(metadata_list_batch), expected_length), "Shape mismatch for batch metadata"
    # Check first entry (full)
    assert not torch.all(output_batch[0] == 0).item(), "First entry in batch (full) should not be all zeros"
    # Check second entry (partial) - elevation part should be zero
    assert torch.all(output_batch[1, -(2*len(meta_cfg.elevation_scales)):] == 0).item(), "Elevation slice for second entry (partial) should be zero"
    assert not torch.all(output_batch[1, :-(2*len(meta_cfg.elevation_scales))] == 0).item(), "Non-elevation slice for second entry (partial) should not be zero"
    # Check third entry (empty)
    assert torch.all(output_batch[2] == 0).item(), "Third entry in batch (empty) should be all zeros"
    assert not any(record.levelno == logging.WARNING for record in caplog.records), "No warnings for valid batch operations"

    # Test Case 8: MetaConfig with no components enabled
    meta_cfg_none_enabled = MetaConfig(
        use_geolocation=False,
        use_temporal=False,
        use_elevation=False
    )
    expected_length_none_enabled = 0
    # Provide some data, it should be ignored
    metadata_list_some_data = [{
        "lat": 40.7128, "lon": -74.0060, "datetime_utc": datetime(2023, 7, 15, 14, 30, 0), "elevation_m": 10.0,
    }]
    caplog.clear()
    output_none_enabled = preprocess_metadata_batch(metadata_list_some_data, meta_cfg_none_enabled, expected_length_none_enabled)
    assert isinstance(output_none_enabled, torch.Tensor), "Output should be a torch.Tensor even with no components"
    assert output_none_enabled.shape == (1, expected_length_none_enabled), f"Shape should be (1,0) if no components enabled, got {output_none_enabled.shape}"
    assert not any(record.levelno == logging.WARNING for record in caplog.records)

    # Test Case 9: Empty input list
    caplog.clear()
    output_empty_list = preprocess_metadata_batch([], meta_cfg, expected_length)
    assert isinstance(output_empty_list, torch.Tensor)
    assert output_empty_list.shape == (0, expected_length), f"Shape should be (0, {expected_length}) for empty input list, got {output_empty_list.shape}"

    # Test Case 10: Empty input list with no components enabled
    caplog.clear()
    output_empty_list_none_enabled = preprocess_metadata_batch([], meta_cfg_none_enabled, expected_length_none_enabled)
    assert isinstance(output_empty_list_none_enabled, torch.Tensor)
    assert output_empty_list_none_enabled.shape == (0, 0), f"Shape should be (0,0) for empty list and no components, got {output_empty_list_none_enabled.shape}"


from typus.models.classification import HierarchicalClassificationResult, TaskPrediction, TaxonomyContext
from typus.constants import RankLevel

from linnaeus.inference.postprocessing import enforce_hierarchical_consistency
from linnaeus.inference.artifacts import TaxonomyData, ClassIndexMapData
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree

# Helper function to create mock TaxonomyData
def create_mock_taxonomy_data(hierarchy_map, task_keys, num_classes_per_task, root_id=None, source="test_source", version="1.0"):
    # Ensure num_classes_per_task aligns with what TaxonomyTree expects for task_keys
    num_classes_dict = {task_keys[i]: num_classes_per_task[i] for i in range(len(task_keys))}

    # The hierarchy_map for TaxonomyTree should map child_task_key to {child_idx: parent_idx}
    # For this test, we are providing it directly.
    # Example: {"taxa_L10": {0:0, 1:0, 2:1}, "taxa_L20": {0:0, 1:0}}
    # task_keys would be ["taxa_L10", "taxa_L20", "taxa_L30"] (lowest to highest for map keys)
    # num_classes_per_task [3, 2, 1] (for L10, L20, L30 respectively)

    # TaxonomyTree expects task_keys to be ordered from lowest rank to highest rank
    # for the hierarchy_map processing.
    # However, the input `task_keys` here might be in a different order for other uses.
    # Let's ensure the order for TaxonomyTree is correct.
    # The `hierarchy_map` keys define the child levels. The values define parent levels.
    # We need `task_keys` for TaxonomyTree to be lowest to highest.
    # The `ClassIndexMapData` and `HierarchicalClassificationResult` use RankLevel,
    # which is typically highest value = highest rank.

    # For TaxonomyTree, let's assume hierarchy_map keys are child_task_keys
    # and these correspond to the *earlier* elements in the `task_keys` list
    # when it's sorted from lowest to highest semantic rank.
    # E.g. task_keys = ["species", "genus", "family"]
    # hierarchy_map = {"species": {child_sp_idx: parent_genus_idx}, "genus": {child_gen_idx: parent_fam_idx}}

    tree = TaxonomyTree(
        hierarchy_map=hierarchy_map, # child_task_key -> {child_idx: parent_idx}
        task_keys=task_keys,         # Ordered list of task keys (e.g. L10, L20, L30)
        num_classes=num_classes_dict # map of task_key -> num_classes
    )
    return TaxonomyData(
        taxonomy_tree=tree,
        source=source,
        version=version,
        root_id=root_id,
        linnaeus_task_keys=task_keys # Store the original task keys if needed
    )

# Helper function to create mock ClassIndexMapData
def create_mock_class_index_map_data(rank_levels, idx_to_taxon_id_maps, taxon_id_to_idx_maps, null_taxon_ids, num_classes_per_rank):
    return ClassIndexMapData(
        idx_to_taxon_id=idx_to_taxon_id_maps,    # Dict[RankLevel, Dict[int, int]]
        taxon_id_to_idx=taxon_id_to_idx_maps,    # Dict[RankLevel, Dict[int, int]]
        null_taxon_ids=null_taxon_ids,          # Dict[RankLevel, int]
        num_classes_per_rank=num_classes_per_rank # Dict[RankLevel, int]
    )

def test_hierarchical_consistency():
    # --- Setup ---
    # Define ranks (typus RankLevel, higher value = higher rank)
    # Linnaeus task keys usually map L10=species, L20=genus, etc.
    # So RankLevel.L10 (10) < RankLevel.L20 (20)
    rank_species = RankLevel.L10   # Value 10 (species)
    rank_genus = RankLevel.L20       # Value 20 (genus)
    rank_family = RankLevel.L30     # Value 30 (family)

    # Linnaeus task keys (consistent with typical Linnaeus usage, e.g. from config)
    # Order might be high-to-low or low-to-high depending on context.
    # For TaxonomyTree, we need low-to-high for map processing.
    # For ClassIndexMapData, we use RankLevel keys.
    task_key_species = "taxa_L10" # Corresponds to RankLevel.L10 (species)
    task_key_genus = "taxa_L20"   # Corresponds to RankLevel.L20 (genus)
    task_key_family = "taxa_L30"  # Corresponds to RankLevel.L30 (family)

    # TaxonomyTree setup: task_keys ordered lowest to highest rank for hierarchy_map parsing
    # hierarchy_map: child_task_key -> {child_model_idx: parent_model_idx}
    # Example: Species S1 (idx 0 at L10) is child of Genus G1 (idx 0 at L20)
    #          Species S2 (idx 1 at L10) is child of Genus G2 (idx 1 at L20)
    #          Genus   G1 (idx 0 at L20) is child of Family F1 (idx 0 at L30)
    #          Genus   G2 (idx 1 at L20) is child of Family F1 (idx 0 at L30) (Mistake here for test)
    #          Genus   G2 (idx 1 at L20) should be child of Family F1 (idx 0 at L30) to be consistent for S2->G2->F1
    # Let's make S2 (idx 1, L10) child of G2 (idx 1, L20)
    # And G1 (idx 0, L20) child of F1 (idx 0, L30)
    # And G2 (idx 1, L20) child of F1 (idx 0, L30) -- CORRECTION: G2 is child of F1.

    # Define hierarchy for TaxonomyTree:
    # F1 (idx 0, task_key_family)
    #  |- G1 (idx 0, task_key_genus)
    #  |  |- S1 (idx 0, task_key_species, TID 101)
    #  |  |- S_other (idx 2, task_key_species, TID 103) -> child of G1
    #  |- G2 (idx 1, task_key_genus)
    #     |- S2 (idx 1, task_key_species, TID 201)
    # Null taxa: S_null (idx 3, TID 0), G_null (idx 2, TID 1), F_null (idx 1, TID 2)

    tt_task_keys = [task_key_species, task_key_genus, task_key_family] # Lowest to highest for TaxonomyTree
    tt_num_classes = [4, 3, 2] # Num classes for L10, L20, L30 (incl null)

    # child_task_key: {child_model_idx: parent_model_idx}
    hierarchy_map_for_tree = {
        task_key_species: {
            0: 0, # S1 (idx 0) -> G1 (idx 0)
            1: 1, # S2 (idx 1) -> G2 (idx 1)
            2: 0, # S_other (idx 2) -> G1 (idx 0)
            # S_null (idx 3) has no parent link in map, will be root-like at its level for map
        },
        task_key_genus: {
            0: 0, # G1 (idx 0) -> F1 (idx 0)
            1: 0, # G2 (idx 1) -> F1 (idx 0)
            # G_null (idx 2) has no parent link
        }
        # Family F1 (idx0) and F_null (idx1) are roots in this map context
    }

    taxonomy_data = create_mock_taxonomy_data(
        hierarchy_map=hierarchy_map_for_tree,
        task_keys=tt_task_keys,
        num_classes_per_task=tt_num_classes, # Num classes for Species, Genus, Family
        root_id=1000 # Dummy overall root, not used by enforce_hierarchical_consistency directly
    )

    # ClassIndexMapData setup (uses typus.RankLevel)
    # Taxon IDs: S1=101, S2=201, S_other=103, S_null=0
    #            G1=10,  G2=20, G_null=1
    #            F1=100, F_null=2
    idx_to_tid = {
        rank_species: {0: 101, 1: 201, 2: 103, 3: 0}, # model_idx -> taxon_id
        rank_genus:   {0: 10,  1: 20,  2: 1},
        rank_family:  {0: 100, 1: 2},
    }
    tid_to_idx = {
        rank_species: {101: 0, 201: 1, 103: 2, 0: 3}, # taxon_id -> model_idx
        rank_genus:   {10: 0,  20: 1,  1: 2},
        rank_family:  {100: 0, 2: 1},
    }
    null_taxon_ids = {
        rank_species: 0,
        rank_genus: 1,
        rank_family: 2
    }
    num_classes_map = {
        rank_species: tt_num_classes[0], # 4
        rank_genus:   tt_num_classes[1], # 3
        rank_family:  tt_num_classes[2]  # 2
    }
    class_maps = create_mock_class_index_map_data(
        rank_levels=[rank_species, rank_genus, rank_family],
        idx_to_taxon_id_maps=idx_to_tid,
        taxon_id_to_idx_maps=tid_to_idx,
        null_taxon_ids=null_taxon_ids,
        num_classes_per_rank=num_classes_map
    )

    # --- Test Case 1: Inconsistent species prediction ---
    # Prediction: Family F1 (TID 100), Genus G1 (TID 10), Species S2 (TID 201)
    # S2 (TID 201) is child of G2 (TID 20), not G1 (TID 10). Species should be nulled.
    inconsistent_result1 = HierarchicalClassificationResult(
        taxonomy_context=TaxonomyContext(source="test", version="1.0"),
        tasks=[
            TaskPrediction(rank_level=rank_family, temperature=1.0, predictions=[(100, 0.9)]), # F1
            TaskPrediction(rank_level=rank_genus,  temperature=1.0, predictions=[(10, 0.8)]),  # G1
            TaskPrediction(rank_level=rank_species,temperature=1.0, predictions=[(201, 0.7)]) # S2 (Inconsistent)
        ]
    )
    consistent_result1 = enforce_hierarchical_consistency(inconsistent_result1, taxonomy_data, class_maps)

    assert isinstance(consistent_result1, HierarchicalClassificationResult)
    sorted_tasks1 = sorted(consistent_result1.tasks, key=lambda t: t.rank_level.value, reverse=True)

    # Family prediction should remain F1 (TID 100)
    assert sorted_tasks1[0].rank_level == rank_family
    assert sorted_tasks1[0].predictions[0][0] == 100
    # Genus prediction should remain G1 (TID 10)
    assert sorted_tasks1[1].rank_level == rank_genus
    assert sorted_tasks1[1].predictions[0][0] == 10
    # Species prediction should be changed to null_taxon_id for species (TID 0)
    assert sorted_tasks1[2].rank_level == rank_species
    assert sorted_tasks1[2].predictions[0][0] == null_taxon_ids[rank_species] # Expected TID 0

    # --- Test Case 2: Parent is null, child should become null ---
    # Prediction: Family F1 (TID 100), Genus G_null (TID 1), Species S1 (TID 101)
    # Genus is null, so Species should be nulled.
    inconsistent_result2 = HierarchicalClassificationResult(
        taxonomy_context=TaxonomyContext(source="test", version="1.0"),
        tasks=[
            TaskPrediction(rank_level=rank_family, temperature=1.0, predictions=[(100, 0.9)]), # F1
            TaskPrediction(rank_level=rank_genus,  temperature=1.0, predictions=[(null_taxon_ids[rank_genus], 0.8)]), # G_null (TID 1)
            TaskPrediction(rank_level=rank_species,temperature=1.0, predictions=[(101, 0.7)])  # S1
        ]
    )
    consistent_result2 = enforce_hierarchical_consistency(inconsistent_result2, taxonomy_data, class_maps)
    sorted_tasks2 = sorted(consistent_result2.tasks, key=lambda t: t.rank_level.value, reverse=True)

    assert sorted_tasks2[0].predictions[0][0] == 100 # Family F1
    assert sorted_tasks2[1].predictions[0][0] == null_taxon_ids[rank_genus] # Genus G_null
    assert sorted_tasks2[2].predictions[0][0] == null_taxon_ids[rank_species] # Species S_null (TID 0)

    # --- Test Case 3: Fully consistent predictions ---
    # Prediction: Family F1 (TID 100), Genus G2 (TID 20), Species S2 (TID 201)
    # This is consistent with the defined hierarchy.
    consistent_input = HierarchicalClassificationResult(
        taxonomy_context=TaxonomyContext(source="test", version="1.0"),
        tasks=[
            TaskPrediction(rank_level=rank_family, temperature=1.0, predictions=[(100, 0.9)]), # F1
            TaskPrediction(rank_level=rank_genus,  temperature=1.0, predictions=[(20, 0.8)]),  # G2
            TaskPrediction(rank_level=rank_species,temperature=1.0, predictions=[(201, 0.7)]) # S2
        ]
    )
    processed_result3 = enforce_hierarchical_consistency(consistent_input, taxonomy_data, class_maps)
    sorted_tasks3 = sorted(processed_result3.tasks, key=lambda t: t.rank_level.value, reverse=True)

    assert sorted_tasks3[0].predictions[0][0] == 100 # Family F1
    assert sorted_tasks3[1].predictions[0][0] == 20  # Genus G2
    assert sorted_tasks3[2].predictions[0][0] == 201 # Species S2

    # --- Test Case 4: Prediction for a rank not in class_maps (should ideally not happen with good config) ---
    # This test checks robustness. The function should ideally log a warning and skip consistency for that rank.
    # Let's say we have a prediction for ORDER, but it's not in our class_maps.
    rank_order = RankLevel.L40 # Value 40 (order)
    inconsistent_result4 = HierarchicalClassificationResult(
        taxonomy_context=TaxonomyContext(source="test", version="1.0"),
        tasks=[
            TaskPrediction(rank_level=rank_family, temperature=1.0, predictions=[(100, 0.9)]),
            TaskPrediction(rank_level=rank_order, temperature=1.0, predictions=[(999, 0.95)]), # Order, TID 999
            TaskPrediction(rank_level=rank_genus,  temperature=1.0, predictions=[(10, 0.8)]),
            TaskPrediction(rank_level=rank_species,temperature=1.0, predictions=[(101, 0.7)])
        ]
    )
    # We expect a log warning about the unmapped rank (ORDER)
    # The behavior for other ranks might depend on how it handles the unknown rank in sorting.
    # The current enforce_hierarchical_consistency sorts by rank_level.value.
    # If a rank_level cannot be mapped to a linnaeus_task_key in the TaxonomyTree, it's skipped.
    # If a taxon_id within a known rank_level cannot be mapped, it's treated as null.

    # To test this properly, we might need to capture logs or ensure it doesn't crash.
    # For this specific case, if 'taxa_L40' (for ORDER) is not in tt_task_keys, it will be skipped.
    # The remaining hierarchy (Family -> Genus -> Species) should still be processed.
    # The function logs warnings but doesn't emit pytest warnings, so we just call it directly
    consistent_result4 = enforce_hierarchical_consistency(inconsistent_result4, taxonomy_data, class_maps)

    sorted_tasks4 = sorted(consistent_result4.tasks, key=lambda t: t.rank_level.value, reverse=True)

    # Assuming ORDER is skipped, the hierarchy F->G->S is (100, 10, 101) which is consistent.
    # Find the tasks by rank level
    pred_family = next(t for t in sorted_tasks4 if t.rank_level == rank_family)
    pred_order = next(t for t in sorted_tasks4 if t.rank_level == rank_order)
    pred_genus = next(t for t in sorted_tasks4 if t.rank_level == rank_genus)
    pred_species = next(t for t in sorted_tasks4 if t.rank_level == rank_species)

    assert pred_family.predictions[0][0] == 100
    assert pred_order.predictions[0][0] == 999 # Order prediction remains as it was skipped
    assert pred_genus.predictions[0][0] == 10
    assert pred_species.predictions[0][0] == 101


    # --- Test Case 5: Inconsistency at a higher level nullifies multiple lower levels ---
    # Prediction: Family F1 (TID 100), Genus G2 (TID 20), Species S1 (TID 101) -> Inconsistent G2 -> S1
    # Correction: Family F_null (TID 2), Genus G1 (TID 10), Species S1 (TID 101)
    # Family is F_null, so Genus G1 should become G_null, and Species S1 should become S_null.
    inconsistent_result5 = HierarchicalClassificationResult(
        taxonomy_context=TaxonomyContext(source="test", version="1.0"),
        tasks=[
            TaskPrediction(rank_level=rank_family, temperature=1.0, predictions=[(null_taxon_ids[rank_family], 0.9)]), # F_null
            TaskPrediction(rank_level=rank_genus,  temperature=1.0, predictions=[(10, 0.8)]),  # G1
            TaskPrediction(rank_level=rank_species,temperature=1.0, predictions=[(101, 0.7)]) # S1
        ]
    )
    consistent_result5 = enforce_hierarchical_consistency(inconsistent_result5, taxonomy_data, class_maps)
    sorted_tasks5 = sorted(consistent_result5.tasks, key=lambda t: t.rank_level.value, reverse=True)

    assert sorted_tasks5[0].predictions[0][0] == null_taxon_ids[rank_family]
    assert sorted_tasks5[1].predictions[0][0] == null_taxon_ids[rank_genus]
    assert sorted_tasks5[2].predictions[0][0] == null_taxon_ids[rank_species]
