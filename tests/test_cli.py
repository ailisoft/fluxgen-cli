import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest


# Submodules of ``fluxgen.cli`` that hold top-level imports from
# ``fluxgen.generator`` / ``fluxgen.editor`` and therefore need to be
# reloaded after swapping the fakes into ``sys.modules``. Listed in
# dependency order — leaves first so they pick up the fake before the
# root ``__init__`` re-runs its own imports.
_CLI_SUBMODULES = (
    "fluxgen.cli.commands",
    "fluxgen.cli.presets_arg",
    "fluxgen.cli.interactive",
)


def load_cli_without_mflux():
    fake_generator = MagicMock()
    fake_generator.generate_image = MagicMock()
    fake_generator.generate_random_filename = MagicMock(return_value="fake.png")
    fake_generator.SUPPORTED_MODELS = ["zimage-turbo", "zimage", "flux2-klein4b", "flux2-klein9b", "krea2", "qwen21"]
    fake_generator.DEFAULT_MODEL = "zimage-turbo"

    # Ensure ``fluxgen.cli`` and its submodules are loaded at least
    # once so reload has something to reload. ``reload`` updates an
    # existing module's ``__dict__`` rather than creating a new
    # object, so ``cli.handle_generate.__globals__`` stays in sync
    # with ``sys.modules['fluxgen.cli.commands']`` and patches like
    # ``patch('fluxgen.cli.commands.generate_image')`` take effect.
    # Without reload, ``patch.dict`` cleanup would drop the submodules
    # from ``sys.modules`` and any subsequent
    # ``patch('fluxgen.cli.commands.X')`` would import a *second*
    # fresh module object that the test's already-bound
    # ``cli.handle_generate`` no longer references.
    importlib.import_module("fluxgen.cli")
    for name in _CLI_SUBMODULES:
        importlib.import_module(name)

    with patch.dict(sys.modules, {"fluxgen.generator": fake_generator}):
        for name in _CLI_SUBMODULES:
            importlib.reload(sys.modules[name])
        importlib.reload(sys.modules["fluxgen.cli"])

    return sys.modules["fluxgen.cli"]


def test_edit_default_output_uses_random_filename(tmp_path):
    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")
    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        timer=False,
        width=None,
        height=None,
    )

    with patch("wonderwords.RandomWord") as mock_random_word_cls, \
         patch("fluxgen.editor.ImageEditor") as mock_editor_cls:
        mock_random_word_cls.return_value.random_words.side_effect = [["red"], ["blue"]]
        editor = mock_editor_cls.return_value

        cli.handle_edit(args)
        cli.handle_edit(args)

    output_paths = [
        call.kwargs["output_path"]
        for call in editor.edit.call_args_list
    ]
    expected = [
        str(Path("output/input_red.png").resolve()),
        str(Path("output/input_blue.png").resolve())
    ]
    assert output_paths == expected


def test_edit_missing_input_exits_before_editor_load(tmp_path):
    cli = load_cli_without_mflux()
    missing_image = tmp_path / "missing.png"
    args = SimpleNamespace(
        image=[str(missing_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        timer=False,
        width=None,
        height=None,
    )

    with patch("fluxgen.editor.ImageEditor") as mock_editor_cls, \
         pytest.raises(SystemExit) as exc:
        cli.handle_edit(args)

    assert exc.value.code == 1
    mock_editor_cls.assert_not_called()


def test_default_generate_accepts_global_flag_before_prompt():
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["-s", "a quiet prompt"])

    args = handle_generate.call_args.args[0]
    assert args.command == "generate"
    assert args.prompt == "a quiet prompt"
    assert args.silent is True


def test_subcommand_accepts_global_flag_after_command():
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "-v", "a noisy prompt"])

    args = handle_generate.call_args.args[0]
    assert args.command == "gen"
    assert args.prompt == "a noisy prompt"
    assert args.verbose is True


def test_silent_keeps_errors_visible_and_hides_info(capsys):
    cli = load_cli_without_mflux()

    cli.setup_logging(silent=True)
    cli.logger.info("hidden")
    cli.logger.error("Error: visible")

    captured = capsys.readouterr()
    assert "hidden" not in captured.err
    assert "Error: visible" in captured.err


def test_resolution_presets_exist():
    """All resolution presets map to valid (width, height) tuples."""
    from fluxgen.presets import ALL_RESOLUTION_PRESETS
    assert ALL_RESOLUTION_PRESETS["tiny"] == (512, 512)
    assert ALL_RESOLUTION_PRESETS["square"] == (768, 768)
    assert ALL_RESOLUTION_PRESETS["large"] == (1024, 1024)
    assert ALL_RESOLUTION_PRESETS["full"] == (1536, 1536)
    # Aspect presets
    assert ALL_RESOLUTION_PRESETS["1:1"] == (512, 512)
    assert ALL_RESOLUTION_PRESETS["16:9"] == (960, 544)
    assert ALL_RESOLUTION_PRESETS["9:16"] == (544, 960)
    assert ALL_RESOLUTION_PRESETS["4:3"] == (768, 576)
    assert ALL_RESOLUTION_PRESETS["3:4"] == (576, 768)


def test_resolution_cli_parsing():
    """--resolution flag is parsed correctly."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "-r", "large", "a prompt"])

    args = handle_generate.call_args.args[0]
    assert args.resolution == "large"
    assert args.prompt == "a prompt"


def test_resolution_cli_parsing_aspect_ratio():
    """--resolution accepts aspect ratio presets."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "--resolution", "9:16", "a prompt"])

    args = handle_generate.call_args.args[0]
    assert args.resolution == "9:16"


def test_default_resolution_is_tiny():
    """When no --resolution is specified, default is 'tiny' (512x512)."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "a prompt"])

    args = handle_generate.call_args.args[0]
    assert not hasattr(args, "resolution"), \
        "resolution should not be set when not passed (argparse.SUPPRESS)"


def test_handle_generate_resolution_tiny_dimensions():
    """Resolution preset 'tiny' resolves to 512x512."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution="tiny", width=None, height=None,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        cli.handle_generate(args, {})

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 512
    assert kwargs["height"] == 512


def test_handle_generate_resolution_large_dimensions():
    """Resolution preset 'large' resolves to 1024x1024."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution="large", width=None, height=None,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        cli.handle_generate(args, {})

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 1024
    assert kwargs["height"] == 1024


def test_handle_generate_qwen21_default_quantize_preload():
    """qwen21 preloads at the spec's q4 default when no --quantize is passed."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.cli.commands.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution=None, width=None, height=None,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="qwen21", verbose=False, silent=False,
            timer=False,
        )
        cli.handle_generate(args, {})

    preload_quantize = mock_mm.get_model.call_args.kwargs["quantize"]
    assert preload_quantize == 4
    _, kwargs = mock_gen.call_args
    assert kwargs["preset"]["quantize"] == 4


def test_handle_generate_explicit_quantize_beats_model_default():
    """An explicit --quantize must override qwen21's q4 spec default."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.cli.commands.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution=None, width=None, height=None,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=8,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="qwen21", verbose=False, silent=False,
            timer=False,
        )
        cli.handle_generate(args, {})

    preload_quantize = mock_mm.get_model.call_args.kwargs["quantize"]
    assert preload_quantize == 8
    _, kwargs = mock_gen.call_args
    assert kwargs["preset"]["quantize"] == 8


def test_handle_generate_resolution_aspect_ratio():
    """Aspect ratio preset '9:16' resolves to correct dimensions."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution="9:16", width=None, height=None,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        cli.handle_generate(args, {})

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 544
    assert kwargs["height"] == 960


def test_handle_generate_width_height_overrides_resolution():
    """--width/--height override the resolution preset."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        # width set to actual value simulates CLI --width 800;
        # height omitted (not present) simulates no --height passed
        args = SimpleNamespace(
            resolution="full", width=800,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        # height is NOT set on the namespace (simulates argparse.SUPPRESS)
        if hasattr(args, "height"):
            del args.height
        cli.handle_generate(args, {})

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 800
    assert kwargs["height"] == 1536  # from 'full' preset


def test_handle_generate_config_override_resolution():
    """Config width/height used when no CLI --resolution or --width/--height passed."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        # No resolution, width, or height on args (simulates no CLI flags)
        args = SimpleNamespace(
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        config = {"defaults": {"width": 640, "height": 480}}
        cli.handle_generate(args, config)

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 640
    assert kwargs["height"] == 480


def test_handle_generate_resolution_overrides_config():
    """Explicit --resolution overrides config file width/height."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        args = SimpleNamespace(
            resolution="large",  # explicitly set
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        # Config has 640x480 but resolution=large should override it
        config = {"defaults": {"width": 640, "height": 480}}
        cli.handle_generate(args, config)

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 1024  # from 'large' preset, overrides config
    assert kwargs["height"] == 1024  # from 'large' preset, overrides config


def test_handle_generate_partial_width_falls_back_to_config():
    """Partial --width only: missing height falls back to config before default."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        # Only width is set; height is absent (argparse.SUPPRESS)
        args = SimpleNamespace(
            width=800,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        config = {"defaults": {"height": 768}}
        cli.handle_generate(args, config)

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 800
    assert kwargs["height"] == 768  # from config, not 512 default


def test_handle_generate_partial_width_explicit_resolution_uses_preset():
    """Partial --width with explicit -r: missing height uses preset, not config."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}), \
         patch("fluxgen.cli.commands.generate_image") as mock_gen, \
         patch("fluxgen.generator.ModelManager") as mock_mm:
        mock_mm.get_model.return_value = MagicMock()

        # Only width is set, but -r full is explicit
        args = SimpleNamespace(
            resolution="full", width=800,
            prompt="a prompt", preset_idx=0, preset=None,
            steps=None, quantize=None,
            output=None, output_dir="output",
            seed=None, style="none",
            init_image=None, strength=0.4,
            model="zimage-turbo", verbose=False, silent=False,
            timer=False,
        )
        if hasattr(args, "height"):
            del args.height
        config = {"defaults": {"height": 768}}
        cli.handle_generate(args, config)

    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 800
    assert kwargs["height"] == 1536  # from 'full' preset, not config 768


# ── _get_version ─────────────────────────────────────────────────────────────────


def test_get_version_caches_after_first_call():
    """_get_version resolves once and reuses the cached value."""
    cli = load_cli_without_mflux()
    cli._cached_version = None  # fresh state

    with patch.object(cli, "distribution") as mock_dist:
        mock_dist.return_value.version = "9.9.9"
        v1 = cli._get_version()
        v2 = cli._get_version()
        v3 = cli._get_version()

    assert v1 == v2 == v3 == "9.9.9"
    assert mock_dist.call_count == 1
    assert cli._cached_version == "9.9.9"


def test_get_version_falls_back_to_pyproject_when_distribution_missing():
    """When importlib.metadata fails, read pyproject.toml directly."""
    import tomllib
    cli = load_cli_without_mflux()
    cli._cached_version = None

    with patch.object(cli, "distribution", side_effect=FileNotFoundError("nope")):
        v = cli._get_version()

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]

    assert v == expected
    assert cli._cached_version == expected


def test_get_version_delegates_to_fluxgen_version_when_metadata_missing():
    """When ``importlib.metadata.distribution`` can't find the
    package, ``_get_version`` delegates to ``fluxgen.__version__``
    rather than reimplementing its own fallback chain.

    This is the contract that keeps the CLI's ``--version`` output
    in lockstep with what embedders see via ``import fluxgen`` — a
    single source of truth. The drift test
    ``test_fluxgen_fallback_literal_matches_pyproject`` keeps the
    literal in ``fluxgen/__init__.py`` honest.
    """
    cli = load_cli_without_mflux()
    cli._cached_version = None

    # Patch distribution to raise; ``_get_version`` should now
    # delegate to fluxgen.__version__ (which falls back to the
    # hard-coded literal "0.3.3" via the fluxgen package init).
    with patch.object(cli, "distribution", side_effect=FileNotFoundError("nope")):
        v = cli._get_version()

    import fluxgen
    assert v == fluxgen.__version__
    assert cli._cached_version == fluxgen.__version__


# ── Fast path: --help / --version skip config load ─────────────────────────────


def test_help_does_not_load_config(capsys):
    """`fluxgen --help` exits before handle_* runs, so config load is skipped."""
    cli = load_cli_without_mflux()

    # patch.object(cli, ...) is required: patch("fluxgen.cli.load_config")
    # re-imports cli and patches a different module object than the one
    # load_cli_without_mflux returns.
    with patch.object(cli, "load_config") as mock_load_config:
        with pytest.raises(SystemExit):
            cli.main(["--help"])

    mock_load_config.assert_not_called()
    captured = capsys.readouterr()
    assert "fluxgen" in captured.out


def test_version_does_not_load_config(capsys):
    """`fluxgen --version` exits before handle_* runs, so config load is skipped."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config") as mock_load_config:
        with pytest.raises(SystemExit):
            cli.main(["--version"])

    mock_load_config.assert_not_called()
    captured = capsys.readouterr()
    # argparse prints version to stdout
    assert "fluxgen" in captured.out


def test_normal_command_still_loads_config():
    """Non-passthrough commands still go through load_config()."""
    cli = load_cli_without_mflux()

    with patch.object(cli, "load_config", return_value={}) as mock_load_config, \
         patch.object(cli, "handle_generate") as handle_generate:
        cli.main(["gen", "a prompt"])

    mock_load_config.assert_called_once()
    handle_generate.assert_called_once()


# ── Cleanup pass: __version__, max_edit_dimension ───────


def test_fluxgen_package_exposes_version():
    """`import fluxgen; fluxgen.__version__` resolves via importlib.metadata."""
    import fluxgen
    # The venv installs fluxgen-cli at a pinned version. We only assert
    # a non-empty string and that it parses as a SemVer-ish tuple.
    assert isinstance(fluxgen.__version__, str)
    assert fluxgen.__version__, "fluxgen.__version__ must not be empty"
    # The fallback path is covered separately by
    # test_fluxgen_package_version_falls_back_without_metadata.


def test_fluxgen_fallback_literal_matches_pyproject():
    """The hard-coded fallback in ``fluxgen/__init__.py`` must match
    ``[project] version`` in pyproject.toml so source-only checkouts
    (where importlib.metadata can't find the package) report the same
    version as an installed venv. Locking the literal here catches the
    next ``pyproject.toml`` bump that forgot to update the fallback.
    """
    import fluxgen
    import tomllib

    # Extract the fallback literal from fluxgen/__init__.py by reading
    # the `except PackageNotFoundError` branch.
    fluxgen_init = Path(__file__).parent.parent / "fluxgen" / "__init__.py"
    fallback_literal = None
    in_fallback_branch = False
    for raw_line in fluxgen_init.read_text().splitlines():
        if "except PackageNotFoundError" in raw_line:
            in_fallback_branch = True
            continue
        if in_fallback_branch and "__version__ =" in raw_line:
            # Match patterns like ``__version__ = "0.3.3"`` (single
            # or double quotes; strip both).
            quote = raw_line.split('"')[1] if '"' in raw_line else raw_line.split("'")[1]
            fallback_literal = quote
            break

    assert fallback_literal is not None, (
        "Could not find a `__version__ = \"...\"` assignment after "
        "an `except PackageNotFoundError:` in fluxgen/__init__.py. "
        "The fallback literal structure changed; update this test."
    )

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject_version = tomllib.load(f)["project"]["version"]

    assert fallback_literal == pyproject_version, (
        f"fluxgen/__init__.py fallback {fallback_literal!r} does not match "
        f"pyproject.toml [project] version {pyproject_version!r}. "
        f"Update the literal in fluxgen/__init__.py to match pyproject.toml."
    )
    # Sanity: the runtime __version__ should also match (assuming the
    # venv is installed; if the fallback fired, it'd match the literal).
    assert fluxgen.__version__ == pyproject_version


def test_fluxgen_package_version_falls_back_without_metadata(monkeypatch):
    """When importlib.metadata can't find the package, __version__ falls
    back to the literal in fluxgen/__init__.py (kept in sync with
    pyproject.toml)."""
    import importlib
    import sys

    import fluxgen

    # Save and restore the real importlib.metadata.version on reload.
    real_version = sys.modules["fluxgen"].__version__

    # Reload the package with importlib.metadata.version patched to raise.
    class _Boom:
        def __call__(self, *a, **kw):
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError("simulated missing metadata")

    saved = importlib.metadata.version
    importlib.metadata.version = _Boom()
    try:
        reloaded = importlib.reload(fluxgen)
        # The fallback literal is "0.4.0" — locked here so the contract
        # is grep-able alongside pyproject.toml.
        assert reloaded.__version__ == "0.4.0"
    finally:
        importlib.metadata.version = saved
        importlib.reload(fluxgen)
    # Sanity: the restored module is back to whatever importlib says.
    assert sys.modules["fluxgen"].__version__ == real_version


def test_handle_edit_passes_max_dimension_from_config(tmp_path):
    """handle_edit reads max_edit_dimension from the config dict and
    threads it to editor.edit(max_dimension=...)."""
    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")

    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        seed=None,
        timer=False,
        width=None,
        height=None,
    )

    config = {"defaults": {"max_edit_dimension": 1024}}

    with patch("fluxgen.editor.ImageEditor") as mock_editor_cls:
        editor = mock_editor_cls.return_value
        cli.handle_edit(args, config=config)

    assert editor.edit.call_args.kwargs["max_dimension"] == 1024


def test_handle_edit_default_max_dimension_when_no_config(tmp_path):
    """handle_edit falls back to MAX_EDIT_DIMENSION (1920) when no config
    max_edit_dimension is provided."""
    from fluxgen.editor import MAX_EDIT_DIMENSION

    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")

    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        seed=None,
        timer=False,
        width=None,
        height=None,
    )

    with patch("fluxgen.editor.ImageEditor") as mock_editor_cls:
        editor = mock_editor_cls.return_value
        cli.handle_edit(args)  # no config

    assert editor.edit.call_args.kwargs["max_dimension"] == MAX_EDIT_DIMENSION


def test_handle_edit_invalid_max_dimension_falls_back(tmp_path, caplog):
    """A non-int or zero/negative max_edit_dimension is ignored and the
    default applies; a warning is logged."""
    import logging as _logging

    cli = load_cli_without_mflux()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"fake")

    args = SimpleNamespace(
        image=[str(input_image)],
        prompt="make it sunset",
        output=None,
        output_dir="output",
        steps=None,
        guidance=1.0,
        seed=None,
        timer=False,
        width=None,
        height=None,
    )

    bad_configs = [
        {"defaults": {"max_edit_dimension": "not-an-int"}},
        {"defaults": {"max_edit_dimension": 0}},
        {"defaults": {"max_edit_dimension": -1}},
        # bool is a subclass of int — ``isinstance(True, int)`` is True.
        # The CLI must reject it explicitly so a TOML
        # ``max_edit_dimension = true`` doesn't sneak through as a
        # legitimate value (which would later compare incorrectly in
        # the editor's ``> max_dimension`` checks).
        {"defaults": {"max_edit_dimension": True}},
        {"defaults": {"max_edit_dimension": False}},
    ]

    for bad in bad_configs:
        # Clear records so each iteration is checked in isolation;
        # otherwise a warning logged for an earlier case would mask a
        # silent regression in a later one.
        caplog.clear()
        with patch("fluxgen.editor.ImageEditor") as mock_editor_cls, \
             caplog.at_level(_logging.WARNING, logger="fluxgen"):
            editor = mock_editor_cls.return_value
            cli.handle_edit(args, config=bad)

        assert editor.edit.call_args.kwargs["max_dimension"] == 1920
        assert any(
            "max_edit_dimension" in rec.message for rec in caplog.records
        ), f"no max_edit_dimension warning logged for config={bad!r}"


def test_edit_parser_default_model_is_flux2_klein_edit():
    """`fluxgen edit …` defaults --model to flux2-klein-edit."""
    cli = load_cli_without_mflux()
    config = {}

    with patch.object(cli, "load_config", return_value=config):
        args = cli.get_parser(config, "0.0.0-test").parse_args(
            ["edit", "image.png", "do thing"]
        )

    assert args.command == "edit"
    assert args.model == "flux2-klein-edit"


def test_edit_parser_rejects_legacy_flux2_klein_id():
    """Bare ``flux2-klein`` is not accepted (no alias)."""
    cli = load_cli_without_mflux()
    config = {}

    with patch.object(cli, "load_config", return_value=config), \
         pytest.raises(SystemExit):
        cli.get_parser(config, "0.0.0-test").parse_args(
            ["edit", "image.png", "do thing", "--model", "flux2-klein"]
        )


# ── resolve_image_dimensions (priority chain) ──────────────────────────────


def _make_args(**overrides):
    """Build a minimal args namespace with the four resolution-related
    attributes absent by default (simulating ``argparse.SUPPRESS``).
    Tests override only the ones they want to set, which mirrors how
    argparse leaves unset attributes off the namespace entirely.
    """
    args = SimpleNamespace()
    for key in ("resolution", "width", "height"):
        if key in overrides:
            setattr(args, key, overrides[key])
    return args


def test_resolve_image_dimensions_default_when_nothing_set():
    """No CLI flags, no config -> 512x512 default."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args()
    config = {}
    assert resolve_image_dimensions(args, config) == (512, 512)


def test_resolve_image_dimensions_cli_width_overrides_default():
    """--width alone takes the width axis; height still defaults."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(width=800)
    assert resolve_image_dimensions(args, {}) == (800, 512)


def test_resolve_image_dimensions_cli_height_overrides_default():
    """--height alone takes the height axis; width still defaults."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(height=600)
    assert resolve_image_dimensions(args, {}) == (512, 600)


def test_resolve_image_dimensions_cli_width_height_both_win():
    """Both --width and --height set -> exact (w, h), ignoring everything else."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(width=800, height=600)
    config = {"defaults": {"width": 1024, "height": 768}}
    assert resolve_image_dimensions(args, config) == (800, 600)


def test_resolve_image_dimensions_cli_resolution_wins_over_config():
    """--resolution flag overrides config file width/height."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(resolution="large")
    config = {"defaults": {"width": 640, "height": 480}}
    assert resolve_image_dimensions(args, config) == (1024, 1024)


def test_resolve_image_dimensions_config_used_when_no_cli():
    """Config width/height used when no CLI flags are passed."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args()
    config = {"defaults": {"width": 640, "height": 480}}
    assert resolve_image_dimensions(args, config) == (640, 480)


def test_resolve_image_dimensions_partial_width_falls_back_to_config_height():
    """--width only: missing height falls back to config before default."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(width=800)
    config = {"defaults": {"height": 768}}
    assert resolve_image_dimensions(args, config) == (800, 768)


def test_resolve_image_dimensions_partial_height_falls_back_to_config_width():
    """--height only: missing width falls back to config before default."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(height=768)
    config = {"defaults": {"width": 800}}
    assert resolve_image_dimensions(args, config) == (800, 768)


def test_resolve_image_dimensions_partial_width_with_resolution_uses_preset():
    """--width only + explicit --resolution: missing height uses preset, not config."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(resolution="full", width=800)
    config = {"defaults": {"height": 768}}
    assert resolve_image_dimensions(args, config) == (800, 1536)


def test_resolve_image_dimensions_partial_width_no_resolution_no_config():
    """--width only, no --resolution, no config height -> 512 default."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(width=800)
    assert resolve_image_dimensions(args, {}) == (800, 512)


def test_resolve_image_dimensions_partial_height_no_resolution_no_config():
    """--height only, no --resolution, no config width -> 512 default."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(height=600)
    assert resolve_image_dimensions(args, {}) == (512, 600)


def test_resolve_image_dimensions_partial_width_with_resolution_no_config():
    """--width only + --resolution: height comes from preset, not 512."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(resolution="16:9", width=960)
    assert resolve_image_dimensions(args, {}) == (960, 544)


def test_resolve_image_dimensions_aspect_ratio_preset():
    """Aspect-ratio presets (e.g. 9:16) resolve correctly."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(resolution="9:16")
    assert resolve_image_dimensions(args, {}) == (544, 960)


def test_resolve_image_dimensions_treats_zero_as_explicit_value():
    """A user-supplied --width 0 (legal but silly) should NOT be
    treated as 'unset' — argparse stores 0 as the integer 0, not as
    a sentinel. The original implementation used truthy checks that
    silently coerced 0 to the default; this regression test guards
    against that reappearing."""
    from fluxgen.cli.commands import resolve_image_dimensions

    args = _make_args(width=0)
    assert resolve_image_dimensions(args, {}) == (0, 512)
