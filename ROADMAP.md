# Roadmap

Future work for fluxgen-cli, ordered by readiness. Items here are tracked but not yet
in active development.

---

## Planned

- **`qwen21` edit wiring** — **blocked on upstream.** The `QwenImage21Edit`
  variant (multi-reference editing, up to 10 images, RGBA, KV-cached) exists
  only on mflux's GitHub main; the released 0.20.0 ships txt2img only. When
  upstream ships it, wiring still needs real editor plumbing: its generate
  signature (multi-`image_paths`, `output_resolution`, no `image_strength`)
  does not fit the current `ImageEditor` contract — likely also a decision on
  whether it replaces `flux2-klein-edit` as the default edit model. Note the
  edit variant reuses the same `Qwen/Qwen-Image-2.1` checkpoint as
  generation, so no extra download.
- **`fluxgen tui` — interactive session TUI** — Textual app that keeps a
  generation model warm in memory and supports visual iterate-on-prompt loops,
  img2img/edit chaining, persisted sessions, and an output gallery. Full plan
  in `TUI_PLAN.md` (phases 1-4; phase 1 = session core loop). Needs Python
  >= 3.12 and pinned `textual` / `textual-image` under a `tui` extra.

## Recently completed

- **FLUX 3 Image Klein support (`--model flux3-klein` + `--model flux3-klein-edit`)** — shipped.
  `flux3-klein` is registered as a generation-only model (16-step guidance-free flow-map
  distilled; no negative prompt, no img2img) and `flux3-klein-edit` as an editing model
  (up to ten reference images); `mflux` floor is `>=0.24.0` (the `flux3` module landed
  upstream in 0.23.0) and the Python floor moved to 3.11 with it.
- **Qwen-Image-2.1 support (`--model qwen21`)** — shipped. `qwen21` is registered as
  a generation-only model (40-step guidance-free, CFG 1.0); `mflux` floor is `>=0.20.0`
  (the `qwen21` module landed upstream there). Download/memory caveats live in `README.md`
  under the generation docs. The upstream `QwenImage21Edit` variant is tracked under Planned.
- **Krea 2 support (`--model krea2`)** — shipped. `krea2` (Krea 2 Turbo) is registered as
  a generation-only model (8-step distilled, CFG 1.0); `mflux` floor is `>=0.19.1`
  (Krea 2 landed upstream in 0.18.1). Recommended usage, download/memory caveats, and
  the strength-based img2img note live in `README.md` under the generation docs.
  Krea 2 Raw (non-distilled) remains unsupported — it needs a new upstream initializer.
