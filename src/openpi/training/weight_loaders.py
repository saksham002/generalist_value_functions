import dataclasses
import logging
import re
import copy
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
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


@dataclasses.dataclass(frozen=True)
class PaliGemmaSiglipOnlyWeightLoader(WeightLoader):
    """Loads ONLY the SigLIP vision tower from the official PaliGemma checkpoint.

    Used for value-function configs that want a pretrained vision encoder but a
    from-scratch LLM (e.g. a smaller gemma variant). Only the `img` subtree of
    pt_224.npz is taken; the LLM and all value-head / projection params keep their
    random init. The SigLIP `head` (num_classes = LLM width) is dropped when its
    shape does not match the reference, since the checkpoint head projects to the
    gemma_2b width (2048) and a smaller backbone uses a different width.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        paligemma_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]
        img_params = {"img": paligemma_params["img"]}

        # Value function models nest PaliGemma under network/ or q_network/ (and
        # target variants for SARSA/CQL). Mirror PaliGemmaWeightLoader's nesting.
        if "q_network" in params and "PaliGemma" not in params:
            loaded_params = {"q_network": {"PaliGemma": img_params}}
            if "target_q_network" in params:
                loaded_params["target_q_network"] = {"PaliGemma": copy.deepcopy(img_params)}
        elif "network" in params and "PaliGemma" not in params:
            loaded_params = {"network": {"PaliGemma": img_params}}
            if "target_network" in params:
                loaded_params["target_network"] = {"PaliGemma": copy.deepcopy(img_params)}
        else:
            loaded_params = {"PaliGemma": img_params}

        # Drop any leaf whose shape does not match the reference (e.g. the SigLIP
        # head, which projects to the LLM width). _merge_params does not shape-check,
        # so filtering here is what keeps a width-mismatched head as random init.
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        kept = {}
        dropped = []
        for k, v in flat_loaded.items():
            ref = flat_ref.get(k)
            if ref is not None and tuple(ref.shape) == tuple(v.shape):
                kept[k] = v
            else:
                dropped.append(k)
        if not kept:
            raise ValueError(
                "PaliGemmaSiglipOnlyWeightLoader matched zero img params against the model. "
                f"Sample reference keys: {sorted(flat_ref)[:5]}"
            )
        logger.info(
            f"PaliGemmaSiglipOnlyWeightLoader: loading {len(kept)} SigLIP leaves, "
            f"dropping {len(dropped)} (shape mismatch / absent): {sorted(dropped)}"
        )
        loaded_params = flax.traverse_util.unflatten_dict(kept, sep="/")

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


@dataclasses.dataclass(frozen = True)
class Gemma4WeightLoader(WeightLoader):
    """Loads transformer + vision encoder weights from a Gemma 4 checkpoint.

    Unlike `Gemma3WeightLoader`, this loader does NOT stack per-layer params
    along axis 0 (our Gemma 4 Module uses an explicit per-layer loop rather
    than `nn.scan`, so each layer has its own named subtree `layer_{i}/...`).

    The checkpoint is expected to have the same Flax param names as the fork's
    Transformer module, which we reuse for Block/Embedder. The only remapping
    needed is splitting between the LLM subtree (goes under `PaliGemma/llm`)
    and the vision encoder subtree (goes under `PaliGemma/img/encoder`).
    """

    checkpoint_path: str = "gs://gemma-data/checkpoints/gemma4-e2b-pt"
    local_dir: str = "~/.cache/openpi/gemma4_checkpoints"

    def load(self, params: at.Params) -> at.Params:
        import os
        import shutil
        import subprocess

        local_dir = os.path.expanduser(self.local_dir)
        local_checkpoint = os.path.join(local_dir, os.path.basename(self.checkpoint_path))
        # Sentinel: written iff gsutil cp succeeded. Absent = partial cache, redownload.
        download_complete_marker = os.path.join(local_checkpoint, ".openpi_download_complete")

        cache_complete = os.path.exists(local_checkpoint) and os.path.exists(download_complete_marker)
        if os.path.exists(local_checkpoint) and not os.path.exists(download_complete_marker):
            logger.warning(
                f"Gemma 4 checkpoint cache at {local_checkpoint} is missing "
                f"completion marker — treating as partial and re-downloading."
            )
            shutil.rmtree(local_checkpoint)

        if not cache_complete:
            logger.info(f"Downloading Gemma 4 checkpoint to {local_checkpoint}...")
            os.makedirs(local_dir, exist_ok = True)
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", self.checkpoint_path, local_dir],
                check = True,
            )
            with open(download_complete_marker, "w") as f:
                f.write("ok\n")

        logger.info(f"Loading Gemma 4 checkpoint from {local_checkpoint}...")
        import orbax.checkpoint as ocp

        checkpointer = ocp.StandardCheckpointer()
        raw_params = checkpointer.restore(local_checkpoint)
        if "transformer" not in raw_params and any("/" in key for key in raw_params):
            raw_params = flax.traverse_util.unflatten_dict(raw_params, sep = "/")

        logger.info(f"Gemma 4 checkpoint top-level keys: {sorted(raw_params.keys())}")

        # Detect target param format. Two layouts are supported:
        # - per-layer (default): ``layer_{i}`` keys.
        # - stacked_layer_params=True: flat ``g{i}__*`` keys with leading axis
        #   = group size.
        if "network" in params and "PaliGemma" not in params:
            llm_ref = params["network"]["PaliGemma"].get("llm", {})
        else:
            llm_ref = params.get("PaliGemma", {}).get("llm", {})
        llm_ref_inner = llm_ref.get("module", llm_ref)

        stacked_layout = any(
            k.startswith(("g0__", "g1__", "g2__", "g3__"))
            for k in llm_ref_inner.keys()
        )
        if stacked_layout:
            logger.info("Detected stacked-layer-params layout")
            remapped_llm = self._remap_checkpoint_params(raw_params, scan_groups = None)
            remapped_llm = self._stacked_layer_params_stack(remapped_llm, llm_ref_inner)
        else:
            remapped_llm = self._remap_checkpoint_params(raw_params, scan_groups = None)
        remapped_vision = self._remap_vision_params(raw_params)

        if "network" in params and "PaliGemma" not in params:
            paligemma_ref = params["network"]["PaliGemma"]
        else:
            paligemma_ref = params.get("PaliGemma", {})
        use_module_prefix = "module" in paligemma_ref.get("llm", {})
        if use_module_prefix:
            llm_entry = {"module": remapped_llm}
            img_entry = {"module": remapped_vision}
        else:
            llm_entry = remapped_llm
            img_entry = remapped_vision

        paligemma_loaded = {"PaliGemma": {"llm": llm_entry, "img": img_entry}}

        if "network" in params and "PaliGemma" not in params:
            loaded_params = {"network": paligemma_loaded}
            if "target_network" in params:
                loaded_params["target_network"] = copy.deepcopy(paligemma_loaded)
        else:
            loaded_params = paligemma_loaded

        return _merge_params(loaded_params, params, missing_regex = ".*")

    def _remap_checkpoint_params(
        self,
        raw_params: dict,
        *,
        scan_groups: tuple[tuple[str, int, int], ...] | None = None,
    ) -> dict:
        """Remap Gemma 4 LLM params to match our Module's param structure.

        Two output formats are supported:

        - ``scan_groups=None`` (default): per-layer ``layer_{i}`` keys,
          matching the fork's original layout. Used when ``Module(use_scan=False)``.
        - ``scan_groups=((name, start_layer, num_super), ...)``: scan layout.
          For each group ``(name, start, num_super)`` the layers
          ``start..start + num_super * K - 1`` are stacked along axis 0 in
          groups of pattern-unit size ``K``, producing keys
          ``{name}/sub_{j}`` for ``j`` in ``0..K-1``. ``K`` is inferred from
          the total layer count divided by the sum of ``num_super`` across
          groups.
        """
        transformer = raw_params.get("transformer", raw_params)

        result = {}

        # Embedder: pass through all params (input_embedding, mm_*, per_layer_*).
        embedder = transformer["embedder"]
        result["embedder"] = {k: np.array(v) if not isinstance(v, dict) else {
            kk: np.array(vv) for kk, vv in v.items()
        } for k, v in embedder.items()}

        # Per-layer blocks: unwrap the `mlp/<name>/{w: tensor}` nesting and
        # collect into a per-layer dict so we can either pass through directly
        # or stack into super-block format.
        layer_indices = sorted(
            int(k.split("_")[1]) for k in transformer if k.startswith("layer_")
        )
        logger.info(f"Found {len(layer_indices)} LLM layers in checkpoint")
        per_layer_params: dict[int, dict] = {}
        for idx in layer_indices:
            layer_key = f"layer_{idx}"
            layer_params = _deep_convert_to_numpy(transformer[layer_key])
            mlp = layer_params.get("mlp")
            if isinstance(mlp, dict):
                for sub_name, sub_val in list(mlp.items()):
                    if isinstance(sub_val, dict) and "w" in sub_val:
                        mlp[sub_name] = sub_val["w"]
            per_layer_params[idx] = layer_params

        if scan_groups is None:
            for idx, lp in per_layer_params.items():
                result[f"layer_{idx}"] = lp
        else:
            num_layers = len(layer_indices)
            total_super = sum(num_super for _, _, num_super in scan_groups)
            assert total_super > 0 and num_layers % total_super == 0, (
                f"num_layers={num_layers} not divisible by total super-blocks={total_super}"
            )
            pattern_unit_size = num_layers // total_super
            for group_name, start_layer, num_super in scan_groups:
                group_dict: dict[str, dict] = {}
                for sub_idx in range(pattern_unit_size):
                    sub_layer_params = [
                        per_layer_params[start_layer + super_idx * pattern_unit_size + sub_idx]
                        for super_idx in range(num_super)
                    ]
                    group_dict[f"sub_{sub_idx}"] = jax.tree.map(
                        lambda *xs: np.stack(xs, axis = 0), *sub_layer_params
                    )
                result[group_name] = group_dict

        # Final norm
        result["final_norm"] = {"scale": np.array(transformer["final_norm"]["scale"])}
        return result

    def _stacked_layer_params_stack(self, per_layer_remapped: dict, llm_ref_inner: dict) -> dict:
        """Stack per-layer params (``layer_X``) into the flat-named per-group layout.

        The stacked layer params layout uses with names ``g{group_idx}__<path>__<...>``,
        where group_idx ∈ {0=unshared LOCAL, 1=unshared GLOBAL, 2=shared LOCAL,
        3=shared GLOBAL}. The leading axis = number of layers in the group.

        LOCAL vs GLOBAL is inferred from ``attn.q_einsum.w`` last dim (head_dim
        vs global_key_size). Unshared vs shared is determined by reading the
        leading axes of the reference ``g{i}__*`` params: num_unshared = N0+N1
        where N{i} is the layer count expected for group i. This is robust to
        variants like E4B where unshared and shared FFW dims are identical.
        """
        layer_keys = sorted(
            (k for k in per_layer_remapped if k.startswith("layer_")),
            key = lambda k: int(k.split("_")[1]),
        )
        per_layer = {int(k.split("_")[1]): per_layer_remapped[k] for k in layer_keys}

        # LOCAL vs GLOBAL discrimination by Q einsum's last dim.
        q_disc = {idx: lp["attn"]["q_einsum"]["w"].shape[-1] for idx, lp in per_layer.items()}
        q_dims = sorted(set(q_disc.values()))

        # Read num_unshared from the reference param tree's g0/g1 leading axes.
        # Each group's params share a leading-axis = group size; pick any leaf
        # under each group key to read the size.
        def _group_size(group_idx: int) -> int:
            prefix = f"g{group_idx}__"
            for k, v in llm_ref_inner.items():
                if k.startswith(prefix) and hasattr(v, "shape"):
                    return int(v.shape[0])
            return 0
        n_g0 = _group_size(0)  # unshared LOCAL
        n_g1 = _group_size(1)  # unshared GLOBAL
        num_unshared = n_g0 + n_g1
        logger.info(
            f"_stacked_layer_params_stack: num_unshared={num_unshared} "
            f"(g0={n_g0} unshared LOCAL, g1={n_g1} unshared GLOBAL); "
            f"shared layers = layer_{num_unshared}..end"
        )

        def get_group(idx):
            is_global = q_disc[idx] == max(q_dims) if len(q_dims) > 1 else False
            is_shared = idx >= num_unshared
            return (2 if is_shared else 0) + (1 if is_global else 0)

        group_layers: list[list[int]] = [[], [], [], []]
        for idx in sorted(per_layer.keys()):
            group_layers[get_group(idx)].append(idx)

        result: dict = {}
        for k, v in per_layer_remapped.items():
            if not k.startswith("layer_"):
                result[k] = v

        for group_idx in range(4):
            members = group_layers[group_idx]
            if not members:
                continue
            flats = [flax.traverse_util.flatten_dict(per_layer[i]) for i in members]
            for path in flats[0].keys():
                shapes = [f[path].shape for f in flats]
                if len(set(shapes)) > 1:
                    logger.error(
                        f"Stacked layer g{group_idx} path {path} has shape mismatch across "
                        f"layers {members}: {shapes}"
                    )
                stacked = np.stack([f[path] for f in flats], axis = 0)
                name = f"g{group_idx}__" + "__".join(path)
                result[name] = stacked
            logger.info(
                f"Stacked layer group g{group_idx}: {len(members)} layers stacked, "
                f"layer indices = {members}"
            )
        return result

    def _remap_vision_params(self, raw_params: dict) -> dict:
        """Remap Gemma 4 vision encoder params to our `Module.encoder/...` structure.

        The fork's VisionEncoder has submodules `entry`, `transformer`, `exit`
        (and optionally `standardize`). Our wrapper names the entire
        VisionEncoder as `encoder`, so we wrap the fork's subtree accordingly.
        """
        # Gemma 4's vision encoder is a submodule of the top-level Transformer,
        # so it typically lives at `transformer/vision_encoder/` in the ckpt.
        transformer = raw_params.get("transformer", raw_params)
        if "vision_encoder" in transformer:
            vision = transformer["vision_encoder"]
        elif "vision_encoder" in raw_params:
            vision = raw_params["vision_encoder"]
        else:
            raise KeyError(
                f"No vision_encoder found in checkpoint. Top-level keys: {sorted(raw_params.keys())}. "
                f"transformer keys: {sorted(transformer.keys())}"
            )
        logger.info(f"Vision encoder subtree keys: {sorted(vision.keys())}")
        return {"encoder": _deep_convert_to_numpy(vision)}


def _deep_convert_to_numpy(obj):
    """Recursively convert all leaf arrays in a nested dict to numpy arrays."""
    if isinstance(obj, dict):
        return {k: _deep_convert_to_numpy(v) for k, v in obj.items()}
    return np.array(obj)


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
