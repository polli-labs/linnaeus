from typing import Any

import torch
import torch.distributions
import torch.nn as nn

# Attempt to import BaseModel, fallback to Any if not fully available during isolated runs
try:
    from linnaeus.models.base_model import BaseModel
except ImportError:
    BaseModel = Any # Fallback for isolated testing or if path issues occur

class LinnaeusPolicyWrapper(nn.Module):
    """
    Wraps a pre-trained Linnaeus model (instance of `linnaeus.models.base_model.BaseModel`)
    to be used as an RL policy, particularly for actor-critic algorithms like PPO.

    This wrapper facilitates the use of a Linnaeus model within an RL loop by:
    1.  Providing a standard way to get action distributions from observations.
    2.  Adding a value head that takes features from the Linnaeus model's backbone
        to estimate state values (V(s)).
    3.  Handling different operational modes ("sequential" or "multitask").
    4.  Deriving necessary information like rank order from the model's configuration
        if not explicitly provided.
    """

    def __init__(
        self,
        linnaeus_model: BaseModel,
        backbone_features_dim: int,
        num_classes_at_rank: dict[str, int],
        mode: str = "sequential",
        rank_order: list[str] | None = None,
    ):
        """
        Initializes the LinnaeusPolicyWrapper.

        Args:
            linnaeus_model: An instance of `linnaeus.models.base_model.BaseModel`.
                            The policy will use this model to extract features and get logits.
            backbone_features_dim: The output dimension of the Linnaeus model's feature
                                   extractor (backbone). This is used to size the value head.
            num_classes_at_rank: A dictionary mapping each rank name (str) to the number
                                 of output classes for that rank (this count should typically
                                 include a slot for abstention/null if the model's heads
                                 are designed that way).
            mode: Operational mode, either "sequential" (rank-by-rank decisions) or
                  "multitask" (all ranks decided at once). Defaults to "sequential".
            rank_order: Optional. An ordered list of taxonomic rank key strings. If None,
                        it attempts to derive this from `linnaeus_model.config.MODEL.TASK_KEYS`.

        Raises:
            TypeError: If `linnaeus_model` is not an instance of `BaseModel` (and `BaseModel`
                       could be imported).
            ValueError: If `rank_order` cannot be determined or is empty, or if
                        `num_classes_at_rank` is not provided or is empty.
        """
        super().__init__()

        if BaseModel is not Any and not isinstance(linnaeus_model, BaseModel): # Only check if BaseModel imported correctly
            raise TypeError(f"linnaeus_model must be an instance of BaseModel, got {type(linnaeus_model)}")

        self.linnaeus_model = linnaeus_model
        self.mode = mode.lower()

        if rank_order is None:
            if hasattr(self.linnaeus_model, 'config') and \
               hasattr(self.linnaeus_model.config, 'MODEL') and \
               hasattr(self.linnaeus_model.config.MODEL, 'TASK_KEYS'):
                self.rank_order = self.linnaeus_model.config.MODEL.TASK_KEYS
                if not self.rank_order:
                    raise ValueError("rank_order derived from model.config.MODEL.TASK_KEYS is empty.")
            else:
                raise ValueError("rank_order must be provided or available in model.config.MODEL.TASK_KEYS")
        else:
            self.rank_order = rank_order

        if not self.rank_order:
             raise ValueError("rank_order is empty or not properly set.")

        if not num_classes_at_rank:
            raise ValueError("num_classes_at_rank must be provided and non-empty.")
        self.num_classes_at_rank = num_classes_at_rank

        self.value_head = nn.Linear(backbone_features_dim, 1)
        self.backbone_features_dim = backbone_features_dim # Store for reference, used by value_head

    def forward(self, observation: dict[str, Any]) -> tuple[Union[torch.distributions.Distribution, list[torch.distributions.Distribution]], torch.Tensor]:
        """
        Performs a forward pass through the policy wrapper.

        This involves:
        1. Extracting backbone features from the input image using the Linnaeus model.
        2. Computing a state value estimate V(s) using the value head.
        3. Obtaining action logits from the Linnaeus model.
        4. Creating and returning an action distribution (or list of distributions for multitask)
           and the value estimate.

        Args:
            observation: A dictionary from the RL environment, expected to contain:
                         - "image" (torch.Tensor): The input image tensor (B, C, H, W or C, H, W).
                         - "current_rank_index" (int or torch.Tensor): For "sequential" mode,
                           the index of the current taxonomic rank being decided.

        Returns:
            A tuple `(action_distribution, value_estimate)`:
            - `action_distribution`:
                - For "sequential" mode: A `torch.distributions.Categorical` instance
                  for the current rank.
                - For "multitask" mode: A list of `torch.distributions.Categorical`
                  instances, one for each rank in `self.rank_order`.
            - `value_estimate`: A tensor of shape (B,) containing the state value estimates.

        Raises:
            RuntimeError: If backbone features cannot be extracted or have unexpected dimensions,
                          or if the Linnaeus model's forward pass does not return a dictionary of logits.
            ValueError: If required keys are missing from `observation` or model outputs,
                        or if logits shapes do not match expectations based on `num_classes_at_rank`.
        """
        image_tensor = observation["image"]
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.as_tensor(image_tensor, dtype=torch.float32)

        if image_tensor.ndim == 3 :
             image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim != 4:
             raise ValueError(f"Expected image_tensor to be 3D (C,H,W) or 4D (B,C,H,W), got {image_tensor.ndim}D")

        backbone_features: torch.Tensor
        if hasattr(self.linnaeus_model, 'extract_features') and callable(self.linnaeus_model.extract_features):
             backbone_features = self.linnaeus_model.extract_features(image_tensor)
        elif hasattr(self.linnaeus_model, 'backbone') and isinstance(self.linnaeus_model.backbone, nn.Module):
             backbone_output = self.linnaeus_model.backbone(image_tensor)
             if backbone_output.ndim == 3 and backbone_output.shape[0] == image_tensor.shape[0]:
                 backbone_features = backbone_output[:, 0, :]
             elif backbone_output.ndim == 2 and backbone_output.shape[0] == image_tensor.shape[0]:
                 backbone_features = backbone_output
             else:
                 raise RuntimeError(f"Unsupported backbone output shape: {backbone_output.shape}. Expected (B, D) or (B, N, D).")
        else:
            raise RuntimeError("Linnaeus model for policy wrapper must have an 'extract_features' method or a 'backbone' attribute that is an nn.Module.")

        if backbone_features.shape[-1] != self.backbone_features_dim:
            raise RuntimeError(f"Extracted backbone_features dim ({backbone_features.shape[-1]}) does not match "
                               f"expected backbone_features_dim ({self.backbone_features_dim}) for value head.")

        value_estimate = self.value_head(backbone_features)

        all_rank_logits = self.linnaeus_model(image_tensor) # Assumes model(img) returns dict of logits
        if not isinstance(all_rank_logits, dict):
            raise RuntimeError("Linnaeus model's forward pass should return a dictionary of logits for RL policy use.")

        action_distribution: Union[torch.distributions.Categorical, list[torch.distributions.Categorical]]

        if self.mode == "sequential":
            current_rank_index_any = observation["current_rank_index"]
            current_rank_index = int(current_rank_index_any.item()) if isinstance(current_rank_index_any, torch.Tensor) else int(current_rank_index_any)

            if current_rank_index >= len(self.rank_order):
                 raise ValueError(f"current_rank_index {current_rank_index} is out of bounds for rank_order length {len(self.rank_order)}")
            current_rank_name = self.rank_order[current_rank_index]

            if current_rank_name not in all_rank_logits:
                raise ValueError(f"Logits for current rank '{current_rank_name}' not found in model output. Available: {list(all_rank_logits.keys())}")

            action_logits = all_rank_logits[current_rank_name]

            expected_num_actions = self.num_classes_at_rank.get(current_rank_name)
            if expected_num_actions is None:
                raise ValueError(f"num_classes_at_rank not defined for {current_rank_name}")
            if action_logits.shape[-1] != expected_num_actions:
                raise ValueError(f"Logits for rank '{current_rank_name}' have shape {action_logits.shape[-1]}, "
                                 f"but num_classes_at_rank expects {expected_num_actions} (incl. abstain slot).")

            action_distribution = torch.distributions.Categorical(logits=action_logits)

        elif self.mode == "multitask":
            distributions = []
            for rank_name in self.rank_order:
                if rank_name not in all_rank_logits:
                    raise ValueError(f"Logits for rank '{rank_name}' not found in model output for multitask. Available: {list(all_rank_logits.keys())}")

                rank_logits = all_rank_logits[rank_name]
                expected_num_actions_multi = self.num_classes_at_rank.get(rank_name)
                if expected_num_actions_multi is None:
                     raise ValueError(f"num_classes_at_rank not defined for {rank_name} in multitask.")
                if rank_logits.shape[-1] != expected_num_actions_multi:
                    raise ValueError(f"Multitask logits for rank '{rank_name}' have shape {rank_logits.shape[-1]}, "
                                     f"but num_classes_at_rank expects {expected_num_actions_multi} (incl. abstain slot).")

                distributions.append(torch.distributions.Categorical(logits=rank_logits))
            action_distribution = distributions
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return action_distribution, value_estimate.squeeze(-1)

    def evaluate_actions(
        self,
        observation: dict[str, Any],
        actions_taken: Union[torch.Tensor, list[torch.Tensor]] # For multitask, List[Tensor] or stacked Tensor(B, NumRanks)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates actions taken in given observations, returning their
        log probabilities and the entropy of the action distribution.
        Also returns the value estimate for the observations.

        Args:
            observation: A dictionary matching the format from TaxonomicClassificationEnv.
                         Expected keys: "image" (torch.Tensor, shape (B, C, H, W)),
                         and for sequential mode, "current_rank_index" (torch.Tensor, shape (B,)).
            actions_taken: A tensor of actions that were taken.
                           - Sequential mode: Shape (B,).
                           - Multitask mode: List of Tensors (one per rank, each shape (B,)),
                                             or a stacked Tensor of shape (B, NumRanks).

        Returns:
            A tuple (value_estimates, new_log_probs, entropy):
            - value_estimates: Value V(s) for each observation in the batch, shape (B,).
            - new_log_probs: Log probability of `actions_taken` under the current policy, shape (B,).
            - entropy: Entropy of the action distribution for each observation, shape (B,).
        """
        image_tensor = observation["image"]
        if not isinstance(image_tensor, torch.Tensor):
            # Use a parameter's device as a default if tensor needs creation/moving
            param_device = next(self.value_head.parameters()).device
            image_tensor = torch.as_tensor(image_tensor, dtype=torch.float32, device=param_device)

        # 1. Get backbone features (consistent with forward method)
        backbone_features: torch.Tensor
        if hasattr(self.linnaeus_model, 'extract_features') and callable(self.linnaeus_model.extract_features):
            backbone_features = self.linnaeus_model.extract_features(image_tensor)
        elif hasattr(self.linnaeus_model, 'backbone') and isinstance(self.linnaeus_model.backbone, nn.Module):
            backbone_output = self.linnaeus_model.backbone(image_tensor)
            if backbone_output.ndim == 3 and backbone_output.shape[0] == image_tensor.shape[0]:
                 backbone_features = backbone_output[:, 0, :]
            elif backbone_output.ndim == 2 and backbone_output.shape[0] == image_tensor.shape[0]:
                 backbone_features = backbone_output
            else:
                raise RuntimeError(f"Unsupported backbone output shape for eval: {backbone_output.shape}. Expected (B,D) or (B,N,D)")

            if backbone_features.shape[-1] != self.backbone_features_dim:
                raise RuntimeError(f"Features dim mismatch in eval: {backbone_features.shape[-1]} vs {self.backbone_features_dim}")
        else:
            raise RuntimeError("Model must have 'extract_features' method or 'backbone' attribute (nn.Module) for eval.")

        value_estimates = self.value_head(backbone_features).squeeze(-1) # (B,)

        # 2. Get action logits (consistent with forward method)
        all_rank_logits = self.linnaeus_model(image_tensor)
        if not isinstance(all_rank_logits, dict):
            raise RuntimeError("Model forward pass should return dict of logits for eval.")

        new_log_probs: torch.Tensor
        entropy: torch.Tensor

        if self.mode == "sequential":
            current_rank_indices = observation["current_rank_index"]
            if not isinstance(current_rank_indices, torch.Tensor):
                current_rank_indices = torch.as_tensor(current_rank_indices, dtype=torch.long, device=image_tensor.device)

            # Assuming for a batch evaluation, all samples are at the same current_rank_index.
            # If not, this part needs to be more sophisticated (e.g. group by rank_index or use loops).
            # Taking the first element's rank_index as representative for the batch.
            rank_idx_for_batch = current_rank_indices[0].item()
            current_rank_name = self.rank_order[rank_idx_for_batch]

            action_logits = all_rank_logits[current_rank_name]
            current_action_dist = torch.distributions.Categorical(logits=action_logits)

            if not isinstance(actions_taken, torch.Tensor):
                 actions_taken = torch.as_tensor(actions_taken, dtype=torch.long, device=action_logits.device)

            new_log_probs = current_action_dist.log_prob(actions_taken)
            entropy = current_action_dist.entropy()

        elif self.mode == "multitask":
            log_probs_list = []
            entropy_list = []

            actions_taken_per_rank: list[torch.Tensor]
            if isinstance(actions_taken, list) and all(isinstance(at, torch.Tensor) for at in actions_taken):
                actions_taken_per_rank = actions_taken
            elif isinstance(actions_taken, torch.Tensor) and actions_taken.ndim == 2 and actions_taken.shape[1] == len(self.rank_order):
                actions_taken_per_rank = list(actions_taken.unbind(dim=1))
            else:
                raise ValueError("actions_taken for multitask mode should be a list of tensors (one per rank, shape (B,)) or a stacked tensor (B, NumRanks).")

            for i, rank_name in enumerate(self.rank_order):
                rank_logits = all_rank_logits[rank_name]
                rank_dist = torch.distributions.Categorical(logits=rank_logits)

                action_for_this_rank = actions_taken_per_rank[i]
                # Ensure it's a tensor on the correct device
                if not isinstance(action_for_this_rank, torch.Tensor):
                    action_for_this_rank = torch.as_tensor(action_for_this_rank, dtype=torch.long, device=rank_logits.device)
                elif action_for_this_rank.device != rank_logits.device:
                    action_for_this_rank = action_for_this_rank.to(rank_logits.device)


                log_probs_list.append(rank_dist.log_prob(action_for_this_rank))
                entropy_list.append(rank_dist.entropy())

            new_log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1) # Sum log_probs across ranks: (B,)
            entropy = torch.stack(entropy_list, dim=1).sum(dim=1)       # Sum entropies across ranks: (B,)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return value_estimates, new_log_probs, entropy

if __name__ == "__main__":
    from typing import Any  # Required if BaseModel falls back to Any
    from unittest.mock import MagicMock  # Import Any if used for BaseModel fallback

    batch_size = 2
    backbone_feat_dim = 64
    rank_names = ["family", "genus", "species"]
    num_classes_config = {"family": 3+1, "genus": 5+1, "species": 10+1}

    mock_model = MagicMock(spec=BaseModel if BaseModel is not Any else object)
    if BaseModel is Any: # If BaseModel could not be imported, spec against a generic object
        mock_model = MagicMock()


    mock_model.config = MagicMock()
    mock_model.config.MODEL = MagicMock()
    mock_model.config.MODEL.TASK_KEYS = rank_names

    # Mocking self.linnaeus_model.backbone behavior
    # Create a dummy nn.Module instance for the backbone, then mock its forward method.
    # This allows the policy to treat `self.linnaeus_model.backbone` as a callable nn.Module.
    dummy_backbone_nn_module = nn.Linear(1,1) # Actual layers don't matter, it's a mock target
    dummy_backbone_nn_module.forward = MagicMock(return_value=torch.randn(batch_size, backbone_feat_dim))
    mock_model.backbone = dummy_backbone_nn_module


    mock_model_output_logits = {
        "family": torch.randn(batch_size, num_classes_config["family"]),
        "genus": torch.randn(batch_size, num_classes_config["genus"]),
        "species": torch.randn(batch_size, num_classes_config["species"]),
    }
    # This defines what `mock_model(image_tensor)` will return.
    mock_model.return_value = mock_model_output_logits

    # --- Test Sequential Policy ---
    print("--- Testing Sequential Policy Wrapper (with mocked Linnaeus Model) ---")
    policy_seq = LinnaeusPolicyWrapper(
        linnaeus_model=mock_model,
        backbone_features_dim=backbone_feat_dim,
        num_classes_at_rank=num_classes_config, # Argument order updated
        mode="sequential",
        rank_order=rank_names
    )

    dummy_image_obs_seq = torch.randn(batch_size, 3, 224, 224)
    observation_seq_genus = { "image": dummy_image_obs_seq, "current_rank_index": 1 }

    action_dist_seq, value_seq = policy_seq(observation_seq_genus)

    print(f"Sequential Mode - Action Distribution type: {type(action_dist_seq)}")
    print(f"  Logits shape for genus: {action_dist_seq.logits.shape}")
    assert action_dist_seq.logits.shape == (batch_size, num_classes_config['genus'])
    print(f"Sequential Mode - Value Estimate shape: {value_seq.shape}")
    assert value_seq.shape == (batch_size,)
    sampled_action_seq = action_dist_seq.sample()
    print(f"  Sampled action shape for genus: {sampled_action_seq.shape}")
    assert sampled_action_seq.shape == (batch_size,)


    # --- Test Multitask Policy ---
    print("\n--- Testing Multitask Policy Wrapper (with mocked Linnaeus Model) ---")
    policy_multi = LinnaeusPolicyWrapper(
        linnaeus_model=mock_model,
        backbone_features_dim=backbone_feat_dim,
        num_classes_at_rank=num_classes_config, # Argument order updated
        mode="multitask",
        rank_order=rank_names
    )

    dummy_image_obs_multi = torch.randn(batch_size, 3, 224, 224)
    observation_multi = {"image": dummy_image_obs_multi}

    action_dist_list_multi, value_multi = policy_multi(observation_multi)

    print(f"Multitask Mode - Action Distribution type: {type(action_dist_list_multi)}")
    assert isinstance(action_dist_list_multi, list)
    assert len(action_dist_list_multi) == len(rank_names)
    print(f"  Number of distributions in list: {len(action_dist_list_multi)}")

    for i, rank_name in enumerate(rank_names):
        dist = action_dist_list_multi[i]
        print(f"  Distribution for rank '{rank_name}':")
        print(f"    Logits shape: {dist.logits.shape}")
        assert dist.logits.shape == (batch_size, num_classes_config[rank_name])
        sampled_action = dist.sample()
        print(f"    Sampled action shape: {sampled_action.shape}")
        assert sampled_action.shape == (batch_size,)

    print(f"Multitask Mode - Value Estimate shape: {value_multi.shape}")
    assert value_multi.shape == (batch_size,)

    print("\nRefactored Policy Tests completed.")
