"""LoRA key mapping for Qwen-Image-2.1 (mflux >= 0.20.0).

mflux's qwen21 transformer keeps diffusers module names verbatim
(``transformer_blocks.N.attn.to_q``, ``img_mlp.proj`` — see
``Qwen21WeightMapping.get_transformer_mapping``), so adapters trained
against the diffusers pipeline (e.g. Viggle's turbo LoRAs) key their
tensors ``transformer.<module path>.lora_A/lora_B.weight`` 1:1 with the
MLX paths below. The generic loader already tries the ``transformer.`` /
``diffusion_model.`` / ``base_model.model.`` prefixes and the
lora_A/B, lora_up/down and ComfyUI-flat spellings, so this mapping only
enumerates the transformer's linear module paths.

Every linear layer of the transformer is listed — not just the modules
Viggle's adapter targets — so other qwen21 LoRAs load without mapping
changes. Extra targets are harmless: the loader applies only the
patterns a given file actually matches.
"""

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget

_UP_SUFFIXES = (
    "lora_B.weight",
    "lora_B.default.weight",
    "lora_up.weight",
    "lora_up.default.weight",
    "lora.up.weight",
    "lora.up.default.weight",
)
_DOWN_SUFFIXES = (
    "lora_A.weight",
    "lora_A.default.weight",
    "lora_down.weight",
    "lora_down.default.weight",
    "lora.down.weight",
    "lora.down.default.weight",
)
_PREFIXES = ("", "transformer.", "diffusion_model.", "base_model.model.")


class Qwen21LoRAMapping(LoRAMapping):
    NUM_TRANSFORMER_BLOCKS = 32

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        targets.extend(Qwen21LoRAMapping._get_global_targets())
        targets.extend(Qwen21LoRAMapping._get_transformer_block_targets())
        return targets

    @staticmethod
    def _get_global_targets() -> list[LoRATarget]:
        return [
            Qwen21LoRAMapping._target("img_in"),
            Qwen21LoRAMapping._target("time_text_embed.timestep_embedder.linear_1"),
            Qwen21LoRAMapping._target("time_text_embed.timestep_embedder.linear_2"),
            Qwen21LoRAMapping._target("txt_in.in_layer"),
            Qwen21LoRAMapping._target("txt_in.out_layer"),
            Qwen21LoRAMapping._target("modulation.1"),
            Qwen21LoRAMapping._target("norm_out.linear"),
            Qwen21LoRAMapping._target("proj_out"),
        ]

    @staticmethod
    def _get_transformer_block_targets() -> list[LoRATarget]:
        block = "transformer_blocks.{block}"
        return [
            Qwen21LoRAMapping._target(f"{block}.attn.to_q"),
            Qwen21LoRAMapping._target(f"{block}.attn.to_k"),
            Qwen21LoRAMapping._target(f"{block}.attn.to_v"),
            Qwen21LoRAMapping._target(f"{block}.attn.to_out.0"),
            Qwen21LoRAMapping._target(f"{block}.img_mlp.proj"),
            Qwen21LoRAMapping._target(f"{block}.img_mlp.out"),
            Qwen21LoRAMapping._target(f"{block}.img_mlp.gate_layer"),
        ]

    @staticmethod
    def _target(model_path: str) -> LoRATarget:
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=Qwen21LoRAMapping._matrix_patterns(model_path, _UP_SUFFIXES),
            possible_down_patterns=Qwen21LoRAMapping._matrix_patterns(model_path, _DOWN_SUFFIXES),
            possible_alpha_patterns=[f"{prefix}{model_path}.alpha" for prefix in _PREFIXES],
        )

    @staticmethod
    def _matrix_patterns(model_path: str, suffixes: tuple[str, ...]) -> list[str]:
        return [
            f"{prefix}{model_path}.{suffix}" for prefix in _PREFIXES for suffix in suffixes
        ]
