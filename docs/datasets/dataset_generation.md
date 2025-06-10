# Official Dataset Provenance (ibrida-v0-r1)

This document details the generation process for the initial set of official Polli Linnaeus pre-trained models. These models (covering North American Mammalia, Amphibia, Reptilia, Aves, Primary Terrestrial Arthropoda, and Angiospermae) were trained on datasets derived from the `ibrida-v0-r1` database. Understanding this provenance is crucial for interpreting model performance, taxonomic scope, and applicability.

For guidance on preparing your own custom datasets for Polli Linnaeus, please refer to [Data Loading for Training](../training/data_loading.md).

## Dataset Generation Tool: `ibrida.generator`

The datasets were generated using an internal tool called `ibrida.generator`.

> **Note on `ibrida.generator` Availability:** The `ibrida.generator` tool is currently internal. We plan to open-source it in late summer 2025. If you have a pressing need for this tool before then, please open an issue on the [Polli Linnaeus GitHub repository](https://github.com/polli-labs/linnaeus), and we may be able to prioritize its release based on demand.

## Executive Summary of Dataset Provenance

The six datasets (`Mammalia`, `Amphibia`, `Reptilia`, `Aves`, `PTA`, `Angiospermae`) were generated from the `ibrida-v0-r1` database, which is built from the **iNaturalist Open Data dump of December 2024**. The export process follows a sophisticated, multi-stage filtering pipeline designed to produce datasets suitable for training hierarchical taxonomic classifiers. All datasets share a common core logic, with minor variations in observation thresholds.

The key stages of the data generation process are as follows:

1.  **Source Data**: All exports originate from the `ibrida-v0-r1` database, which includes elevation data for each observation.

2.  **Regional Species Selection**: The pipeline first identifies a set of "in-threshold" species. For all six datasets, this is defined as species having at least **50 research-grade observations** (60 for PTA and Angiospermae) within the **North America (`NAfull`)** bounding box.

3.  **Ancestor-Aware Expansion**: This is a critical step for hierarchical modeling. For each "in-threshold" species, the pipeline traverses its entire taxonomic lineage (e.g., genus, family, order, etc.) using the pre-computed `expanded_taxa` table. The union of the initial species set and all their unique ancestors forms the final set of taxa for the dataset.
    *   For all datasets except Amphibia, this "full ancestor search" (`fas`) includes both major (e.g., family, order) and minor (e.g., subfamily, tribe) taxonomic ranks.
    *   For the Amphibia dataset, the ancestor search was configured to include **only major ranks**.

4.  **Global Observation Inclusion**: Once the final taxon list (species + ancestors) is defined, the pipeline gathers **all observations** for these taxa globally, not just those within North America. A boolean flag `in_region` is computed for each observation to indicate if it originated within the `NAfull` bounding box. This `INCLUDE_OUT_OF_REGION_OBS=true` setting ensures the model is exposed to a wider geographic distribution for the target taxa.

5.  **Cladistic and Quality Filtering**: The globally gathered observations are then filtered to the specific taxonomic group (`CLADE` or `METACLADE`). The following quality filters are applied:
    *   **Quality Grade Filtering (`RG_FILTER_MODE`)**: A mode of `ALL_EXCLUDE_SPECIES_NON_RESEARCH` is used. This means:
        *   All `research` grade observations are kept.
        *   `casual` or `needs_id` grade observations are only kept if they are *not* identified to the species level (i.e., they are coarse-level labels like genus or family). This preserves valuable partial labels while discarding lower-quality species-level data.
    *   **Partial Rank Wiping (`MIN_OCCURRENCES_PER_RANK`)**: To prevent the model from learning from insufficient data, any taxonomic rank (e.g., genus, family) with fewer than **50 total occurrences** (60 for PTA and Angiospermae) in the final dataset has its label "wiped" (set to NULL) for the relevant observations. This effectively pushes rare taxa up to their next-most-common ancestor.

6.  **Final Sampling and Export**:
    *   Only the primary photo (`position=0`) for each observation is included (`PRIMARY_ONLY=true`).
    *   To prevent class imbalance from hyper-abundant species, the number of research-grade, species-level observations is capped via random sampling (`MAX_RN`).
    *   The final export includes all `expanded_taxa` columns (providing full ancestry for each observation) and the `elevation_meters` column.

This comprehensive process yields six large-scale, hierarchically-consistent datasets that are optimized for training robust, taxonomy-aware computer vision models.

## Exhaustive Filtering Parameters Table

This table details the specific parameters from the `ibrida-v0-r1` export wrappers used to generate each of the six datasets.

| Parameter | Description | Mammalia | Amphibia | Reptilia | Aves | PTA (Arthropoda) | Angiospermae |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Database & Release** |
| `VERSION_VALUE` | Database schema version. | `v0` | `v0` | `v0` | `v0` | `v0` | `v0` |
| `RELEASE_VALUE` | iNaturalist data release. | `r1` | `r1` | `r1` | `r1` | `r1` | `r1` |
| `DB_NAME` | Source database name. | `ibrida-v0-r1` | `ibrida-v0-r1` | `ibrida-v0-r1` | `ibrida-v0-r1` | `ibrida-v0-r1` | `ibrida-v0-r1` |
| **Regional Filtering** |
| `REGION_TAG` | Bounding box for species selection. | `NAfull` | `NAfull` | `NAfull` | `NAfull` | `NAfull` | `NAfull` |
| `MIN_OBS` | Min research-grade obs in-region for a species to be included. | `50` | `50` | `50` | `50` | `50` | `60` |
| `INCLUDE_OUT_OF_REGION_OBS` | If true, fetch all global obs for selected taxa. | `true` | `true` | `true` | `true` | `true` | `true` |
| **Cladistic & Ancestor Filtering** |
| `CLADE` / `METACLADE` | The root taxonomic group for the dataset. | `mammalia` | `amphibia` | `reptilia` | `aves` | `pta` | `angiospermae` |
| `INCLUDE_MINOR_RANKS_IN_ANCESTORS` | If true, includes minor ranks (e.g., subfamily) in ancestor traversal. | `true` | `false` | `true` | `true` | `true` | `true` |
| **Quality & Observation Filtering** |
| `RG_FILTER_MODE` | Logic for filtering by `quality_grade`. | `ALL_EXCLUDE_SPECIES_NON_RESEARCH` | `ALL_EXCLUDE_SPECIES_NON_RESEARCH` | `ALL_EXCLUDE_SPECIES_NON_RESEARCH` | `ALL_EXCLUDE_SPECIES_NON_RESEARCH` | `ALL_EXCLUDE_SPECIES_NON_RESEARCH` | `ALL_EXCLUDE_SPECIES_NON_RESEARCH` |
| `MIN_OCCURRENCES_PER_RANK` | If a rank has fewer occurrences than this, its label is wiped. | `50` | `50` | `50` | `50` | `50` | `60` |
| **Export & Sampling** |
| `MAX_RN` | Max random sample of research-grade observations per species. | `2500` | `2500` | `2500` | `2500` | `2750` | `1750` |
| `PRIMARY_ONLY` | If true, only export the primary photo for an observation. | `true` | `true` | `true` | `true` | `true` | `true` |
| `INCLUDE_ELEVATION_EXPORT` | If true, include `elevation_meters` column in the final CSV. | `true` | `true` | `true` | `true` | `true` | `true` |
| **Source Wrapper Script** | Script used for generation. | `wrapper_mammalia_..._fas_elev.sh` | `wrapper_amphibia_..._fas_elev.sh` | `wrapper_reptilia_..._fas_elev.sh` | `wrapper_aves_..._fas_elev.sh` | `wrapper_pta_..._fas_elev.sh` | `wrapper_angiospermae_..._fas_elev.sh` |

## Understanding Taxonomic Scope and Root Taxa

The filtering logic described above (especially Regional Species Selection and Ancestor-Aware Expansion) defines the taxonomic scope of each model. While a full list of included taxa for each dataset is extensive, the parameters provide insight into how this scope was determined. The `CLADE` or `METACLADE` parameter in the [Exhaustive Filtering Parameters Table](#exhaustive-filtering-parameters-table) specifies the highest-level taxonomic group(s) from which all other taxa in the dataset descend.

Here are the root taxa definitions for the six initial Polli Linnaeus models:

1.  **Mammalia Model:**
    *   **Root:** Class Mammalia
    *   **Definition:** `("L50_taxonID" = 40151)` (Taxon ID 40151 at taxonomic rank L50)

2.  **Amphibia Model:**
    *   **Root:** Class Amphibia
    *   **Definition:** `("L50_taxonID" = 20978)` (Taxon ID 20978 at taxonomic rank L50)

3.  **Reptilia Model:**
    *   **Root:** Class Reptilia
    *   **Definition:** `("L50_taxonID" = 26036)` (Taxon ID 26036 at taxonomic rank L50)

4.  **Aves Model:**
    *   **Root:** Class Aves
    *   **Definition:** `("L50_taxonID" = 3)` (Taxon ID 3 at taxonomic rank L50)

5.  **PTA (Primary Terrestrial Arthropoda) Model:**
    *   **Type:** Metaclade (derived from multiple roots)
    *   **Roots:** Class Insecta OR Class Arachnida
    *   **Definition:** `("L50_taxonID" = 47158 OR "L50_taxonID" = 47119)` (Taxon ID 47158 [Insecta] or Taxon ID 47119 [Arachnida] at taxonomic rank L50)

6.  **Angiospermae Model:**
    *   **Root:** Subphylum Angiospermae (Flowering Plants)
    *   **Definition:** `("L57_taxonID" = 47125)` (Taxon ID 47125 at taxonomic rank L57)

This documentation aims to provide transparency into how the official Polli Linnaeus datasets were constructed, enabling users to better understand model behavior and make informed decisions about their application.
