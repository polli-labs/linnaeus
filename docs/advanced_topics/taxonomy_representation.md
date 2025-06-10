# Taxonomy Representation in linnaeus

This document details how taxonomic hierarchies are represented and processed within the linnaeus framework, focusing on the transition from the raw `hierarchy_map` to the centralized `TaxonomyTree` class.

## 1. Biological Taxonomy Structure

Biological classification follows a hierarchical tree structure, organizing taxa into nested ranks of increasing specificity (e.g., Species < Genus < Family < Order < Class). Each taxon (except the ultimate root) has exactly one parent at the next higher rank but can have multiple children at the next lower rank. linnaeus typically works with ranks like L10 (Species), L20 (Genus), L30 (Family), L40 (Order).

## 2. The Raw `hierarchy_map`

### 2.1 Generation

During dataset preparation (`vectorized_dataset_processor.py`), a data structure named `hierarchy_map` is generated based on co-occurring labels observed across adjacent taxonomic levels in the training and validation data.

### 2.2 Structure

The `hierarchy_map` has the following Python structure:

```python
hierarchy_map: Dict[str, Dict[int, int]]
# Example: {
#    "taxa_L10": { child_idx_L10: parent_idx_L20, ... },
#    "taxa_L20": { child_idx_L20: parent_idx_L30, ... },
#    ...
# }
```

-   The **outer keys** are the *child* task level strings (e.g., `"taxa_L10"`).
-   The **inner dictionary** maps the *class index* of a taxon at the child level (e.g., `child_idx_L10`) to the *class index* of its parent at the next higher level (e.g., `parent_idx_L20`).
-   Crucially, it encodes the **cross-level parent-child relationship** using the *class indices* defined for each level.

### 2.3 Limitations

While containing the necessary information, this raw structure presented challenges:

-   **Interpretation:** Its structure (`child_task -> {child_idx: parent_idx}`) required careful interpretation by consuming components.
-   **Redundancy:** Different components (label smoothing, hierarchical heads) independently parsed and processed this map, risking inconsistency.
-   **Lack of Validation:** The raw map generation didn't inherently validate the overall tree structure for consistency (e.g., cycles, multiple parents).

## 3. The Centralized `TaxonomyTree` Class

To address these limitations, linnaeus uses the `TaxonomyTree` class (`utils/taxonomy/taxonomy_tree.py`) as the central, validated representation of the hierarchy.

### 3.1 Purpose and Design

-   **Single Source of Truth:** Provides a unified, validated representation of the taxonomy derived from the raw `hierarchy_map`.
-   **Abstraction:** Hides the raw map structure, offering a clean API for querying relationships.
-   **Validation:** Performs structural validation (single parent, no cycles, index bounds) during initialization.
-   **Efficiency:** Uses an internal bidirectional graph for efficient traversal.
-   **Extensibility:** Centralizes taxonomy-related computations (e.g., distance calculation).

### 3.2 Initialization

The `TaxonomyTree` is instantiated once during dataset processing (`vectorized_dataset_processor.py`) after the raw `hierarchy_map` and `num_classes` are determined.

```python
# Inside vectorized_dataset_processor.py
raw_hierarchy_map = self._generate_hierarchy_map()
num_classes = {task: len(mapping) for task, mapping in self.class_to_idx.items()}
taxonomy_tree = TaxonomyTree(raw_hierarchy_map, self.task_keys, num_classes)
```

### 3.3 Key Concepts and API

The `TaxonomyTree` operates on **Nodes**, represented as `Tuple[str, int]`, e.g., `("taxa_L10", 5)`.

-   **Traversal:**
    -   `get_parent(node)`: Returns the parent `Node` or `None`.
    -   `get_children(node)`: Returns a `List[Node]` of direct children.
    -   `get_ancestors(node)`: Returns the path `List[Node]` from the node up to its root.
    -   `get_descendants(node)`: Returns a `List[Node]` of all nodes in the subtree below the node (including itself).
-   **Structure Query:**
    -   `get_nodes_at_level(task_key)`: Returns all `Node`s at a specific level.
    -   `get_root_nodes()`: Returns `List[Node]` of nodes with no parent *in the provided map*. These are the highest-level nodes available.
    -   `get_leaf_nodes()`: Returns `List[Node]` of nodes with no children *in the provided map*.
-   **Distance:**
    -   `taxonomic_distance(node1, node2)`: Computes the shortest path distance (number of edges) between two nodes in the tree via their Lowest Common Ancestor (LCA). Returns `float('inf')` if nodes are disconnected.
    -   `build_distance_matrix(task_key)`: Generates a `[N, N]` matrix of pairwise distances between all nodes *at the specified level*.
-   **Helper Matrices:**
    -   `build_hierarchy_matrices()`: Generates matrices mapping parent probabilities to child nodes, used by conditional heads. Structure: `Dict[f"{parent_task}_{child_task}", Tensor[num_parent, num_child]]`.
-   **Validation:**
    -   Implicitly validates structure during `__init__`. Raises `ValueError` on critical issues (cycles, multiple parents).

## 4. Handling Metaclades (Forests)

The term "metaclade" refers to a dataset derived from multiple distinct top-level taxonomic roots (e.g., combining Insecta and Arachnida, both classes under the phylum Arthropoda).

-   **Detection Limitation:** The `TaxonomyTree`, working only with the `hierarchy_map` derived from the dataset labels, **cannot definitively determine if the dataset represents a true biological metaclade**. It can only identify nodes at the highest *specified* `task_key` level that lack a parent *within the map*. These might be true roots or simply the highest level provided. Verifying true independence requires external taxonomic databases (like the `expanded_taxa` table used in the shell scripts).
-   **Handling Strategy:** Components needing to handle potential multiple roots should:
    1.  Check `len(taxonomy_tree.get_root_nodes())`.
    2.  Consult relevant configuration flags (e.g., `LOSS.TAXONOMY_SMOOTHING.UNIFORM_ROOTS`) to decide on the appropriate behavior (e.g., apply uniform smoothing if multiple roots are detected and the flag is set).

## 5. Conclusion

The `TaxonomyTree` class provides a robust, validated, and centralized interface to the taxonomic hierarchy within linnaeus, simplifying the logic within consuming components and ensuring consistent interpretation of the underlying taxonomic structure derived from the dataset.