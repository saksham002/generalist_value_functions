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
    # Optional GCS path to a precomputed flat .npz of this checkpoint. If
    # present, every host downloads from here directly and skips the orbax
    # OCDBT → .npz conversion. The conversion subprocess is flaky on some
    # hosts (silent hangs that take >10 min and trip the JAX heartbeat
    # barrier), so prefer this fast path when available.
    # When left as None, defaults to a path derived from `checkpoint_path`'s
    # basename so e.g. `…/gemma4-e4b-it` → `…/gemma4-e4b-it.npz` rather than
    # silently fetching some other variant's weights.
    npz_gcs_path: str | None = None

    def load(self, params: at.Params) -> at.Params:
        import os
        import subprocess
        import sys
        import textwrap
        import jax

        local_dir = os.path.expanduser(self.local_dir)
        local_checkpoint = os.path.join(local_dir, os.path.basename(self.checkpoint_path))
        local_npz = local_checkpoint + ".npz"
        pidx = jax.process_index()

        # Resolve the precomputed .npz GCS source from the checkpoint basename
        # when not explicitly provided, so e2b-pt / e2b-it / e4b-pt / e4b-it
        # each pull their matching .npz instead of falling through to a stale
        # default.
        npz_gcs_path = self.npz_gcs_path or (
            f"gs://saksham-euw4/gemma4_checkpoints/{os.path.basename(self.checkpoint_path)}.npz"
        )

        # Step 0: If the precomputed .npz is hosted on GCS, gsutil cp it once
        # per host into the local cache. This avoids the in-process orbax
        # restore entirely.
        if not os.path.exists(local_npz) and npz_gcs_path:
            os.makedirs(local_dir, exist_ok = True)
            stat_proc = subprocess.run(
                ["gsutil", "-q", "stat", npz_gcs_path],
                capture_output = True, text = True,
            )
            if stat_proc.returncode == 0:
                logger.info(
                    f"[gemma4 pidx={pidx}] Downloading precomputed .npz from "
                    f"{npz_gcs_path} -> {local_npz}..."
                )
                # `-o GSUtil:check_hashes=never` skips CRC32c verification: the
                # GCS object was uploaded as a composite, so without crcmod's
                # C extension on the host gsutil aborts the download.
                subprocess.run(
                    [
                        "gsutil",
                        "-o", "GSUtil:check_hashes=never",
                        "-m", "cp", npz_gcs_path, local_npz,
                    ],
                    check = True,
                )
                logger.info(f"[gemma4 pidx={pidx}] Download complete.")

        # Step 1: Download the OCDBT checkpoint locally if neither the .npz
        # nor the OCDBT directory is cached. Only needed if we're falling back
        # to the in-process conversion below.
        if not os.path.exists(local_npz) and not os.path.exists(local_checkpoint):
            logger.info(f"Downloading Gemma 4 checkpoint to {local_checkpoint}...")
            os.makedirs(local_dir, exist_ok = True)
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", self.checkpoint_path, local_dir],
                check = True,
            )

        # Step 2: Convert OCDBT → flat .npz once per host. orbax's
        # StandardCheckpointer.restore() under multi-host JAX hangs CPU-bound
        # for many minutes on this checkpoint (likely cross-host barriers per
        # param while materializing jax.Array), and the 600 s barrier timeout
        # then breaks the run. Doing the conversion in a fresh single-process
        # Python (no jax.distributed.initialize, JAX_PLATFORMS=cpu) sidesteps
        # the multi-host coordination entirely; thereafter every host just
        # numpy-loads the .npz, mirroring how PaliGemmaWeightLoader works. No
        # cross-host sync is needed because each host has its own local OCDBT
        # / .npz under ~/.cache/openpi/. This path is a fallback when the
        # precomputed .npz is not available on GCS.
        if not os.path.exists(local_npz):
            logger.info(f"[gemma4 pidx={pidx}] Converting {local_checkpoint} -> {local_npz} (single-process subprocess)...")
            # `np.savez` auto-appends `.npz` if the path doesn't already end
            # in `.npz`, so use a tmp name that already has the suffix and
            # then atomically rename to the final path.
            conversion_script = textwrap.dedent(
                f"""
                import os
                os.environ.setdefault("JAX_PLATFORMS", "cpu")
                import numpy as np
                import flax.traverse_util
                import orbax.checkpoint as ocp
                ckpt = ocp.StandardCheckpointer()
                raw = ckpt.restore({local_checkpoint!r})
                if "transformer" not in raw and any("/" in k for k in raw):
                    raw = flax.traverse_util.unflatten_dict(raw, sep="/")
                flat = flax.traverse_util.flatten_dict(raw, sep="/")
                arrays = {{k: np.asarray(v) for k, v in flat.items()}}
                tmp_path = {local_npz!r} + ".partial.npz"
                np.savez(tmp_path, **arrays)
                os.replace(tmp_path, {local_npz!r})
                print(f"wrote {{len(arrays)}} arrays to {local_npz!r}")
                """
            )
            env = {**os.environ, "JAX_PLATFORMS": "cpu"}
            # Strip every JAX/TPU coordination env var so the subprocess
            # starts as a clean single-process JAX run on CPU only — no
            # multi-host barriers, no TPU device claim.
            for var in (
                "JAX_COORDINATOR_ADDRESS", "JAX_COORDINATOR_PORT",
                "JAX_NUM_PROCESSES", "JAX_PROCESS_ID",
                "TPU_HOST_BOUNDS", "TPU_PROCESS_BOUNDS",
                "TPU_PROCESS_PORT", "TPU_PROCESS_ADDRESSES",
                "TPU_VISIBLE_DEVICES", "TPU_VISIBLE_CHIPS",
                "TPU_WORKER_HOSTNAMES", "TPU_WORKER_ID",
                "LIBTPU_INIT_ARGS",
            ):
                env.pop(var, None)
            proc = subprocess.run(
                [sys.executable, "-c", conversion_script],
                env = env, capture_output = True, text = True,
            )
            if proc.stdout:
                logger.info(f"[gemma4 conversion pidx={pidx} stdout] {proc.stdout.strip()}")
            if proc.stderr:
                logger.info(f"[gemma4 conversion pidx={pidx} stderr] {proc.stderr.strip()}")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Gemma4 OCDBT->.npz conversion subprocess exited {proc.returncode} on pidx={pidx}; "
                    f"stderr above for details."
                )
            logger.info(f"[gemma4 pidx={pidx}] Conversion complete: {local_npz}")

        # Step 3: Each host independently numpy-loads the flat .npz — same
        # pattern as PaliGemmaWeightLoader. No cross-host coordination needed.
        def _rss_gib():
            try:
                with open("/proc/self/status") as _f:
                    for _l in _f:
                        if _l.startswith("VmRSS:"):
                            return int(_l.split()[1]) / 2**20
            except Exception:
                return -1.0
        logger.info(f"[gemma4 pidx={pidx}] BEFORE np.load RSS={_rss_gib():.2f}GiB")
        logger.info(f"Loading Gemma 4 checkpoint (.npz form) from {local_npz}...")
        with np.load(local_npz, allow_pickle = False) as npz_file:
            flat_params = {k: npz_file[k] for k in npz_file.files}
        logger.info(f"[gemma4 pidx={pidx}] AFTER flat_params dict ({len(flat_params)} keys) RSS={_rss_gib():.2f}GiB")
        raw_params = flax.traverse_util.unflatten_dict(flat_params, sep = "/")
        logger.info(f"[gemma4 pidx={pidx}] AFTER unflatten_dict RSS={_rss_gib():.2f}GiB")

        logger.info(f"Gemma 4 checkpoint top-level keys: {sorted(raw_params.keys())}")

        logger.info(f"[gemma4 pidx={pidx}] starting _remap_checkpoint_params")
        remapped_llm = self._remap_checkpoint_params(raw_params)
        logger.info(f"[gemma4 pidx={pidx}] _remap_checkpoint_params done, starting _remap_vision_params")
        remapped_vision = self._remap_vision_params(raw_params)
        logger.info(f"[gemma4 pidx={pidx}] _remap_vision_params done")

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
                # Build a structural twin of `paligemma_loaded` for target_network
                # that shares the same numpy arrays at the leaves but has fresh
                # dict containers at each level. `copy.deepcopy` allocates ~10 GB
                # of fresh array buffers and stalls indefinitely on hosts under
                # memory/IO pressure (observed: heartbeat starvation on slow
                # disk hosts). Safe because `init_critic` later casts target
                # paths to a separate dtype via `.astype`, which materialises
                # fresh JAX arrays on TPU and breaks any host-side aliasing.
                logger.info(f"[gemma4 pidx={pidx}] starting shallow target_network tree (no array copy)")
                def _share_arrays(tree):
                    if isinstance(tree, dict):
                        return {k: _share_arrays(v) for k, v in tree.items()}
                    return tree
                loaded_params["target_network"] = _share_arrays(paligemma_loaded)
                logger.info(f"[gemma4 pidx={pidx}] shallow target_network tree done")
        else:
            loaded_params = paligemma_loaded

        logger.info(f"[gemma4 pidx={pidx}] entering _merge_params")
        merged = _merge_params(loaded_params, params, missing_regex = ".*")
        logger.info(f"[gemma4 pidx={pidx}] _merge_params returned, returning from load()")
        return merged

    def _remap_checkpoint_params(self, raw_params: dict) -> dict:
        """Remap Gemma 4 LLM params to match our Module's param structure.

        Our `Module` uses per-layer named blocks (`layer_{i}`), which matches
        the fork's Transformer naming convention, so no stacking is needed.
        The only caveat: if the fork saves a top-level `transformer/` wrapper,
        we strip it.
        """
        transformer = raw_params.get("transformer", raw_params)

        result = {}

        # Embedder: pass through all params (input_embedding, mm_*, per_layer_*).
        # Our module reuses the fork's Embedder, so param names already match.
        # Use `np.asarray` (not `np.array`) so we share storage with the loaded
        # `.npz` arrays instead of copying — copies double peak RAM and have
        # OOM'd hosts with heavier shards (e.g. v5litepod-64 worker 8).
        embedder = transformer["embedder"]
        result["embedder"] = {k: np.asarray(v) if not isinstance(v, dict) else {
            kk: np.asarray(vv) for kk, vv in v.items()
        } for k, v in embedder.items()}

        # Per-layer blocks: unwrap the `mlp/<name>/{w: tensor}` nesting that the
        # checkpoint stores into bare tensors at `mlp/<name>`. The fork's
        # FeedForward wires its sub-Einsums via `nn.share_scope` with custom
        # `weight_name` ("gating_einsum", "linear"), so the live model expects
        # bare tensors at those paths. Mirrors upstream
        # `gemma/gm/ckpts/_compat.py::param_remapper`.
        layer_indices = sorted(
            int(k.split("_")[1]) for k in transformer if k.startswith("layer_")
        )
        logger.info(f"Found {len(layer_indices)} LLM layers in checkpoint")
        for idx in layer_indices:
            layer_key = f"layer_{idx}"
            layer_params = _deep_convert_to_numpy(transformer[layer_key])
            mlp = layer_params.get("mlp")
            if isinstance(mlp, dict):
                for sub_name, sub_val in list(mlp.items()):
                    if isinstance(sub_val, dict) and "w" in sub_val:
                        mlp[sub_name] = sub_val["w"]
            result[layer_key] = layer_params

        # Final norm
        result["final_norm"] = {"scale": np.asarray(transformer["final_norm"]["scale"])}
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
    """Recursively convert all leaf arrays in a nested dict to numpy arrays.

    Uses `np.asarray` (zero-copy when input is already an ndarray) rather
    than `np.array` (always copies) — only callers are the Gemma 4 loader
    paths, which read leaves directly from a 20 GB .npz; copying everything
    once more doubles peak RAM and has OOM'd individual hosts.
    """
    if isinstance(obj, dict):
        return {k: _deep_convert_to_numpy(v) for k, v in obj.items()}
    return np.asarray(obj)


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
