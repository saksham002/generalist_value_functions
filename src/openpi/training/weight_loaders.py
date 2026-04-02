import dataclasses
import logging
import re
import copy
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        paligemma_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]

        # Value function models nest PaliGemma under network/ or q_network/
        # (and target variants for SARSA/CQL). Detect this and nest the loaded
        # params accordingly so keys align with the model state.
        if "q_network" in params and "PaliGemma" not in params:
            loaded_params = {"q_network": {"PaliGemma": paligemma_params}}
            if "target_q_network" in params:
                loaded_params["target_q_network"] = {"PaliGemma": copy.deepcopy(paligemma_params)}
        elif "network" in params and "PaliGemma" not in params:
            loaded_params = {"network": {"PaliGemma": paligemma_params}}
            if "target_network" in params:
                loaded_params["target_network"] = {"PaliGemma": copy.deepcopy(paligemma_params)}
        else:
            loaded_params = {"PaliGemma": paligemma_params}

        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen = True)
class Gemma3WeightLoader(WeightLoader):
    """Loads transformer + SigLIP weights from a Gemma 3 checkpoint.

    The remapping logic supports Gemma 3 checkpoints with varying transformer
    depth/width (e.g., 4B and 27B), as long as parameter names follow the
    canonical `transformer/layer_{i}/...` and `SigLiPFromPatches_0/...` layout.
    """

    checkpoint_path: str = "gs://gemma-data/checkpoints/gemma3-4b-pt"
    local_dir: str = "~/.cache/openpi/gemma3_checkpoints"

    def load(self, params: at.Params) -> at.Params:
        import os
        import subprocess

        local_dir = os.path.expanduser(self.local_dir)
        local_checkpoint = os.path.join(local_dir, os.path.basename(self.checkpoint_path))

        if not os.path.exists(local_checkpoint):
            logger.info(f"Downloading Gemma 3 checkpoint to {local_checkpoint}...")
            os.makedirs(local_dir, exist_ok = True)
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", self.checkpoint_path, local_dir],
                check = True,
            )

        logger.info(f"Loading Gemma 3 checkpoint from {local_checkpoint}...")
        import orbax.checkpoint as ocp

        checkpointer = ocp.StandardCheckpointer()
        raw_params = checkpointer.restore(local_checkpoint)
        # Newer Orbax/OCDBT restores can return a flat dict with slash-separated
        # paths (e.g., "transformer/layer_0/...") instead of a nested tree.
        # Normalize to the nested structure expected by remap helpers.
        if "transformer" not in raw_params and any("/" in key for key in raw_params):
            raw_params = flax.traverse_util.unflatten_dict(raw_params, sep = "/")

        remapped_llm = self._remap_checkpoint_params(raw_params)
        remapped_siglip = self._remap_siglip_params(raw_params)

        # Callers using raw Linen (e.g. test_inference_text_only) pass reference
        # params with a "module" prefix, while NNX callers (e.g. test_vlm_query,
        # PaliGemmaValueNetwork) do not. Adapt the loaded structure to match.
        if "network" in params and "PaliGemma" not in params:
            paligemma_ref = params["network"]["PaliGemma"]
        else:
            paligemma_ref = params.get("PaliGemma", {})
        use_module_prefix = "module" in paligemma_ref.get("llm", {})
        if use_module_prefix:
            llm_entry = {"module": remapped_llm}
            img_entry = {"module": remapped_siglip}
        else:
            llm_entry = remapped_llm
            img_entry = remapped_siglip

        paligemma_loaded = {"PaliGemma": {"llm": llm_entry, "img": img_entry}}

        # Value function configs (SARSA, MC, etc.) nest PaliGemma under network/ (and
        # target_network/ for SARSA). Detect this and nest the loaded params so keys
        # align with the model state, mirroring the same logic in PaliGemmaWeightLoader.
        if "network" in params and "PaliGemma" not in params:
            loaded_params = {"network": paligemma_loaded}
            if "target_network" in params:
                loaded_params["target_network"] = copy.deepcopy(paligemma_loaded)
        else:
            loaded_params = paligemma_loaded

        return _merge_params(loaded_params, params, missing_regex = ".*")

    def _remap_checkpoint_params(self, raw_params: dict) -> dict:
        """Remap Gemma 3 checkpoint params to match our Module's param structure.

        Handles:
        - Stripping `transformer/` prefix
        - Stacking per-layer params (layer_0..layer_N) along axis 0
        - Renaming `_query_norm` -> `query_norm`, `_key_norm` -> `key_norm`
        - Stripping `/w` suffixes from einsum params
        """
        transformer = raw_params["transformer"]

        result = {}

        # Embedder params
        result["embedder"] = {"input_embedding": transformer["embedder"]["input_embedding"]}

        # Final norm
        result["final_norm"] = {"scale": transformer["final_norm"]["scale"]}

        # Determine number of layers from checkpoint
        layer_indices = sorted(
            int(k.split("_")[1]) for k in transformer if k.startswith("layer_")
        )
        num_layers = len(layer_indices)
        logger.info(f"Found {num_layers} layers in checkpoint")

        # Collect and stack per-layer params
        # Map from checkpoint nested structure to our flat param names
        param_paths = [
            ("pre_attention_norm", "scale"),
            ("attn", "q_einsum", "w"),
            ("attn", "kv_einsum", "w"),
            ("attn", "attn_vec_einsum", "w"),
            ("attn", "_query_norm", "scale"),
            ("attn", "_key_norm", "scale"),
            ("post_attention_norm", "scale"),
            ("pre_ffw_norm", "scale"),
            ("mlp", "gating_einsum", "w"),
            ("mlp", "linear", "w"),
            ("post_ffw_norm", "scale"),
        ]

        # Our target param structure (what nn.scan creates):
        # layers/pre_attention_norm/scale  [depth, ...]
        # layers/attn/q_einsum             [depth, ...]
        # layers/attn/query_norm/scale     [depth, ...]
        # etc.

        stacked = {}
        for path in param_paths:
            arrays = []
            for idx in layer_indices:
                layer = transformer[f"layer_{idx}"]
                val = layer
                for key in path:
                    val = val[key]
                arrays.append(np.array(val))
            stacked_array = np.stack(arrays, axis = 0)

            # Build our target key, handling name remapping
            target_path = list(path)
            # Strip /w suffix for einsum params
            if target_path[-1] == "w":
                target_path = target_path[:-1]
            # Rename _query_norm -> query_norm, _key_norm -> key_norm
            target_path = [k.lstrip("_") if k.startswith("_") else k for k in target_path]

            # Build nested dict
            d = stacked
            for key in ["layers"] + target_path[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[target_path[-1]] = stacked_array

        result.update(stacked)
        return result

    def _remap_siglip_params(self, raw_params: dict) -> dict:
        """Remap SigLIP params from Gemma 3 checkpoint to our _Module's param structure.

        The checkpoint stores SigLIP under `SigLiPFromPatches_0/siglip_encoder/` with
        per-layer encoder blocks (encoderblock_0..encoderblock_26). Our SigLIP uses
        nn.scan, so we stack per-layer params along axis 0 under a single `encoderblock` key.
        """
        siglip_root = raw_params["SigLiPFromPatches_0"]
        siglip = siglip_root["siglip_encoder"]
        logger.info(f"SigLiPFromPatches_0 keys: {list(siglip_root.keys())}")
        logger.info(f"siglip_encoder keys: {list(siglip.keys())}")

        result = {}

        # Patch embedding conv
        result["embedding"] = {
            "kernel": np.array(siglip["embedding"]["kernel"]),
            "bias": np.array(siglip["embedding"]["bias"]),
        }

        # Position embedding
        result["pos_embedding"] = np.array(siglip["pos_embedding"])

        # Encoder norm (final layer norm)
        result["Transformer"] = {
            "encoder_norm": {
                "scale": np.array(siglip["Transformer"]["encoder_norm"]["scale"]),
                "bias": np.array(siglip["Transformer"]["encoder_norm"]["bias"]),
            },
        }

        # Stack per-layer encoder blocks along axis 0 for scan compatibility
        layer_indices = sorted(
            int(k.split("_")[1])
            for k in siglip["Transformer"]
            if k.startswith("encoderblock_")
        )
        num_layers = len(layer_indices)
        logger.info(f"Found {num_layers} SigLIP encoder layers in checkpoint")

        # Collect all leaf arrays from each layer and stack them
        first_layer = siglip["Transformer"][f"encoderblock_{layer_indices[0]}"]
        flat_first = flax.traverse_util.flatten_dict(first_layer, sep = "/")

        stacked_flat = {}
        for key in flat_first:
            arrays = []
            for idx in layer_indices:
                layer = siglip["Transformer"][f"encoderblock_{idx}"]
                flat_layer = flax.traverse_util.flatten_dict(layer, sep = "/")
                arrays.append(np.array(flat_layer[key]))
            stacked_flat[key] = np.stack(arrays, axis = 0)

        stacked_block = flax.traverse_util.unflatten_dict(stacked_flat, sep = "/")
        result["Transformer"]["encoderblock"] = stacked_block

        # Gemma 3 multimodal projector weights live in the LLM embedder but belong
        # logically with the SigLIP module since that is where the projection now happens.
        embedder = raw_params["transformer"]["embedder"]
        result["mm_soft_embedding_norm"] = np.array(embedder["mm_soft_embedding_norm"]["scale"])
        result["mm_input_projection"] = {"kernel": np.array(embedder["mm_input_projection"]["w"])}

        return result


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    loaded_not_in_ref = []
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
        else:
            loaded_not_in_ref.append(k)

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    filled_from_ref = []
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]
            filled_from_ref.append(k)

    logger.info(
        f"_merge_params: {len(result) - len(filled_from_ref)} loaded keys matched, "
        f"{len(loaded_not_in_ref)} loaded keys not in ref, "
        f"{len(filled_from_ref)} ref keys filled from reference"
    )
    if loaded_not_in_ref:
        loaded_prefixes = sorted({k.split("/")[0] for k in loaded_not_in_ref})
        ref_prefixes = sorted({k.split("/")[0] for k in flat_ref})
        logger.info(f"  Loaded top-level prefixes (unmatched): {loaded_prefixes}")
        logger.info(f"  Reference top-level prefixes: {ref_prefixes}")
        logger.info(f"  Sample unmatched loaded keys: {sorted(loaded_not_in_ref)[:5]}")
        logger.info(f"  Sample reference keys: {sorted(flat_ref.keys())[:5]}")
    if filled_from_ref:
        logger.info(f"  Ref keys not filled by loaded: {sorted(filled_from_ref)}")

    return flax.traverse_util.unflatten_dict(result, sep="/")
