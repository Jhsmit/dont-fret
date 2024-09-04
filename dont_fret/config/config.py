from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import polars as pl
import yaml
from dacite import Config, from_dict
from dacite.data import Data

from dont_fret.utils import clean_types

CONFIG_HOME = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))


@dataclass
class Channel:
    # name: str
    target: str
    value: Union[int, float, list[int], list[float]]
    modulo: Union[float, int, None] = None


@dataclass
class BurstFilterItem:
    name: str
    min: Optional[float] = field(default=None)
    max: Optional[float] = field(default=None)

    def as_expr(self) -> list[pl.Expr]:
        expr = []
        if self.min is not None:
            expr.append(pl.col(self.name) >= self.min)
        if self.max is not None:
            expr.append(pl.col(self.name) <= self.max)
        return expr


@dataclass
class Web:
    """settings related to web application"""

    default_dir: Path
    protect_filebrowser: bool = True
    burst_filters: list[BurstFilterItem] = field(default_factory=list)
    password: Optional[str] = None


@dataclass
class BurstColor:
    streams: list[str] = field(default_factory=lambda: ["DD", "DA", "AA"])
    L: int = 50
    M: int = 100
    T: float = 500e-6


@dataclass
class DontFRETConfig:
    channels: dict[str, Channel]
    streams: dict[str, list[str]]
    burst_search: dict[str, list[BurstColor]]
    web: Web

    @classmethod
    def from_dict(cls, data: Data):
        config = Config(type_hooks={Path: lambda v: Path(v).expanduser()})
        return from_dict(cls, data, config)

    @classmethod
    def from_yaml(cls, fpath: Path):
        data = yaml.safe_load(fpath.read_text())
        return cls.from_dict(data)

    def to_yaml(self, fpath: Path) -> None:
        s = yaml.dump(clean_types(asdict(self)), sort_keys=False)
        fpath.write_text(s)

    def update(self, data: Data):
        new_data = {**self.__dict__, **data}

        # we use `from_dict` to cast to the correct types
        new_cfg = DontFRETConfig.from_dict(new_data)
        vars(self).update(vars(new_cfg))


cfg_file_paths = [
    CONFIG_HOME / "dont_fret" / "dont_fret.yaml",
    Path(__file__).parent / "default.yaml",
]

# take the first one which exists
cfg_fpath = next((p for p in cfg_file_paths if p.exists()), None)
assert cfg_fpath
cfg = DontFRETConfig.from_yaml(cfg_fpath)
