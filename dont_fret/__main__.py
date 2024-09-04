import os
from pathlib import Path
from typing import List, Literal, Optional

import click
import yaml
from solara.__main__ import run

from dont_fret.config import CONFIG_HOME, cfg
from dont_fret.process import batch_search_and_save, search_and_save

ROOT = Path(__file__).parent
APP_PATH = ROOT / "web" / "main.py"


@click.group()
def cli():
    """Don't FRET! CLI for analyzing confocal solution smFRET data."""
    pass


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config", default=None, help="Configuration file to use")
@click.argument("solara_args", nargs=-1, type=click.UNPROCESSED)
def serve(config: Optional[str] = None, solara_args=None):
    """Run the don't fret web application."""
    if config is not None:
        data = yaml.safe_load(Path(config).read_text())
        cfg.update(data)

    solara_args = solara_args or tuple()
    args = [str(APP_PATH), *solara_args]

    run(args)


@cli.command()
@click.option(
    "--global", "is_global", is_flag=True, help="Create config file in user's home directory"
)
def config(is_global: bool):
    """Create a local or global default configuration file."""
    src = ROOT / "config" / "default.yaml"
    if is_global:
        (CONFIG_HOME / "dont_fret").mkdir(exist_ok=True, parents=True)
        output = CONFIG_HOME / "dont_fret" / "dont_fret.yaml"
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


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--burst-colors", default=None, multiple=True, help="Burst colors to process")
@click.option(
    "--write-photons/--no-write-photons", default=False, help="Whether to write photon data"
)
@click.option(
    "--output-type", type=click.Choice([".pq", ".csv"]), default=".pq", help="Output file type"
)
@click.option("--max-workers", type=int, default=None, help="Maximum number of worker threads")
def process(
    input_path: str,
    burst_colors: Optional[list[str]],
    write_photons: bool,
    output_type: Literal[".pq", ".csv"],
    max_workers: Optional[int],
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

    # Convert burst_colors to the expected format
    burst_colors_param = list(burst_colors) if burst_colors else None

    if len(files) == 1:
        click.echo(f"Processing file: {files[0]}")
        search_and_save(
            files[0],
            burst_colors=burst_colors_param,
            write_photons=write_photons,
            output_type=output_type,
        )
    else:
        click.echo("Processing files in batch mode.")
        batch_search_and_save(
            files,
            burst_colors=burst_colors_param,
            write_photons=write_photons,
            output_type=output_type,
            max_workers=max_workers,
        )

    click.echo("Processing completed.")


if __name__ == "__main__":
    cli()
