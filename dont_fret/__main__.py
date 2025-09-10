from pathlib import Path
from typing import Literal, Optional

import click
from solara.__main__ import run

from dont_fret.config import CONFIG_DEFAULT_DIR, CONFIG_HOME_DIR, update_config_from_yaml
from dont_fret.process import batch_search_and_save, search_and_save

ROOT = Path(__file__).parent
APP_PATH = ROOT / "web" / "main.py"


@click.group()
def cli():
    """Don't FRET! CLI for analyzing confocal solution smFRET data."""
    pass


def find_config_file(config_path: Path) -> Optional[Path]:
    if config_path.exists():
        return config_path
    elif (pth := CONFIG_HOME_DIR / config_path).exists():
        return pth
    elif (pth := CONFIG_DEFAULT_DIR / config_path).exists():
        return pth


def load_config(config_path: Path) -> None:
    resolved_cfg_path = find_config_file(Path(config_path))
    if not resolved_cfg_path:
        raise click.BadParameter(f"Configuration file '{config_path}' not found")

    update_config_from_yaml(resolved_cfg_path)
    click.echo("Loading config file at: " + str(resolved_cfg_path))


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config", default=None, help="Configuration file to use")
@click.argument("solara_args", nargs=-1, type=click.UNPROCESSED)
def serve(config: Optional[str] = None, solara_args=None):
    """Run the don't fret web application."""
    if config is not None:
        load_config(Path(config))
    else:
        update_config_from_yaml(CONFIG_DEFAULT_DIR / "default_web.yaml")

    solara_args = solara_args or tuple()
    if "--production" not in solara_args:
        solara_args = (*solara_args, "--production")
    args = [str(APP_PATH), *solara_args]

    run(args)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--config", default=None, help="Configuration file to use")
@click.option(
    "--write-photons/--no-write-photons", default=False, help="Whether to write photon data"
)
@click.option(
    "--output-type", type=click.Choice([".pq", ".csv"]), default=".pq", help="Output file type"
)
@click.option("--max-workers", type=int, default=None, help="Maximum number of worker threads")
def process(
    input_path: str,
    config: Optional[str] = None,
    write_photons: bool = False,
    output_type: Literal[".pq", ".csv"] = ".pq",
    max_workers: Optional[int] = None,
):
    """Process photon file(s) and perform burst search."""

    pth = Path(input_path)
    if pth.is_file():
        files = [pth]
    elif pth.is_dir():
        files = [f for f in pth.iterdir() if f.suffix in SUPPORTED_SUFFIXES]
    else:
        raise click.BadParameter("Input path must be a file or directory")

    if not files:
        click.echo("No supported files found.")
        return

    click.echo(f"Found {len(files)} file(s) to process.")

    if config is not None:
        load_config(Path(config))
    else:
        update_config_from_yaml(CONFIG_DEFAULT_DIR / "default.yaml")

    # Convert burst_colors to the expected format

    if len(files) == 1:
        click.echo(f"Processing file: {files[0]}")
        search_and_save(
            files[0],
            write_photons=write_photons,
            output_type=output_type,
        )
    else:
        click.echo("Processing files in batch mode.")
        batch_search_and_save(
            files,
            write_photons=write_photons,
            output_type=output_type,
            max_workers=max_workers,
        )

    click.echo("Processing completed.")


if __name__ == "__main__":
    cli()


@cli.command()
@click.option("--user", "user", is_flag=True, help="Create config file in user's home directory")
def config(user: bool):
    """Create a local or global default configuration file."""
    src = ROOT / "config" / "default.yaml"
    if user:
        (CONFIG_HOME_DIR).mkdir(exist_ok=True, parents=True)
        output = CONFIG_HOME_DIR / "dont_fret.yaml"
    else:
        output = Path.cwd() / "dont_fret.yaml"

    if output.exists():
        click.echo(f"Configuration file already exists at '{str(output)}'")
        return

    else:
        output.write_text(src.read_text())

    click.echo(f"Configuration file created at '{str(output)}'")


SUPPORTED_SUFFIXES = {
    ".ptu",
}
