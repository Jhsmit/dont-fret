import os
import struct
import time
from pathlib import Path

import numpy as np
import polars as pl
from phconvert.pqreader import (
    _convert_multi_tags,
    _ptu_rec_type_r,
    _ptu_tag_type_r,
    _ptu_TDateTime_to_time_t,
    load_ptu,
)

from dont_fret.config.config import DontFRETConfig, cfg
from dont_fret.expr import from_channel, reduce_and
from dont_fret.models import Bursts


def export_bursts(bursts: Bursts):
    """
    export to .csv format as per
    https://github.com/Fluorescence-Tools/exchange-formats/tree/master/burst/pam

    :param bursts:
    :return:
    """
    # FRET Efficiency,Stoichiometry,Proximity Ratio,Stoichiometry (raw),Lifetime D [ns],Lifetime A [ns],Anisotropy D,Anisotropy A,|TDX-TAA| Filter,ALEX 2CDE Filter,|TDD-TDA| Filter,FRET 2CDE Filter,Duration [ms],Mean Macrotime [s],Number of Photons,Count rate [kHz],Count rate (DD) [kHz],Count rate (DA) [kHz],Count rate (AA) [kHz],Count rate (DD par) [kHz],Count rate (DD perp) [kHz],Count rate (DA par) [kHz],Count rate (DA perp) [kHz],Count rate (AA par) [kHz],Count rate (AA perp) [kHz],Number of Photons (DD),Number of Photons (DA),Number of Photons (AA),Number of Photons (DD par),Number of Photons (DD perp),Number of Photons (DA par),Number of Photons (DA perp),Number of Photons (AA par),Number of Photons (AA perp),log(FD/FA),M1-M2,Number of Photons (DX),Count rate (DX) [kHz],Distance (from intensity) [A],FRET efficiency (from lifetime),Distance (from lifetime) [A],FRET efficiency (sens. Acc. Em.)

    return None


class PhotonFile(object):
    """Base class for file objects of photon data"""

    def __init__(self, file_path: os.PathLike):
        self.file_path = Path(file_path)

        if isinstance(self.file_path, Path) and not self.file_path.exists():
            raise ValueError("Supplied file path does not exist")

    @property
    def filename(self) -> str:
        try:
            return self.file_path.name
        except AttributeError:
            return ""

    def load_file(self, cfg: DontFRETConfig = cfg) -> tuple[pl.DataFrame, dict]:
        """
        returns data, metadata
        """
        timestamps, detectors, nanotimes, metadata = load_ptu(self.file_path)
        lazy_data = pl.LazyFrame(
            {"timestamps": timestamps, "detectors": detectors, "nanotimes": nanotimes}
        )

        channel_expr = {k: from_channel(v) for k, v in cfg.channels.items()}
        stream_expr = {
            k: reduce_and([channel_expr[ch] for ch in v]) for k, v in cfg.streams.items()
        }

        dtype = pl.Enum(stream_expr.keys())
        queries = [
            lazy_data.filter(v).with_columns(pl.lit(k).alias("stream").cast(dtype))
            for k, v in stream_expr.items()
        ]
        concat_query = pl.concat(queries).sort("timestamps")
        data = concat_query.collect()

        return data, metadata


def _read_tag(f):
    tag_struct = struct.unpack("32s i I q", f.read(48))

    # and save it into a dict
    tagname = tag_struct[0].rstrip(b"\0").decode()
    keys = ("idx", "type", "value")
    tag = {k: v for k, v in zip(keys, tag_struct[1:])}
    # tag["offset"] = f.tell() # Uncomment if you want offset information
    # Recover the name of the type (a string)
    tag["type"] = _ptu_tag_type_r[tag["type"]]

    # Some tag types need conversion
    if tag["type"] == "tyFloat8":
        tag["value"] = np.int64(tag["value"]).view("float64")
    elif tag["type"] == "tyBool8":
        tag["value"] = bool(tag["value"])
    elif tag["type"] == "tyTDateTime":
        TDateTime = np.uint64(tag["value"]).view("float64")
        t = time.gmtime(_ptu_TDateTime_to_time_t(TDateTime))
        tag["value"] = time.strftime("%Y-%m-%d %H:%M:%S", t)

    # Some tag types have additional data
    if tag["type"] == "tyAnsiString":
        byte_string = f.read(tag["value"]).rstrip(b"\0")
        try:
            tag["data"] = byte_string.decode()  # try decoding from UTF-8
        except UnicodeDecodeError:
            # Not UTF-8, trying 'latin1'
            # See https://github.com/Photon-HDF5/phconvert/issues/35
            tag["data"] = byte_string.decode("latin1")
    elif tag["type"] == "tyFloat8Array":
        buffer = f.read(tag["value"])
        tag["data"] = np.frombuffer(buffer, dtype="float", count=tag["value"] / 8)
    elif tag["type"] == "tyWideString":
        # WideString use type WCHAR in the original C++ demo code.
        # WCHAR size is not fixed by C++ standard, but on windows
        # is 2 bytes and the default encoding is UTF-16.
        # I'm assuming this is what the PTU requires.
        tag["data"] = f.read(tag["value"] * 2).decode("utf16")
    elif tag["type"] == "tyBinaryBlob":
        tag["data"] = f.read(tag["value"])
    return tagname, tag


def _read_ptu_tags(f) -> dict:
    tags = {}

    while True:
        tagname, tag = _read_tag(f)
        # In case a `tagname` appears multiple times, we make a list
        # to hold all the tags with the same name
        if tagname in tags:
            if not isinstance(tags[tagname], list):
                tags[tagname] = [tags[tagname]]
            tags[tagname].append(tag)
        else:
            tags[tagname] = tag

        if tagname == "Header_End":
            break

    return tags


def _check_ptu_magic(f) -> tuple[bytes, bytes]:
    magic = f.read(8).rstrip(b"\0")
    version = f.read(8).rstrip(b"\0")
    if magic != b"PQTTTR":
        raise IOError("This file is not a valid PTU file. " "Magic: '%s'." % magic)

    return magic, version


def convert_ptu_tags(tags: dict) -> dict:
    """Convert PTU tags to metadata dict"""

    acquisition_duration = tags["MeasDesc_AcquisitionTime"]["value"] * 1e-3
    ctime_t = time.strptime(tags["File_CreatingTime"]["value"], "%Y-%m-%d %H:%M:%S")
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S", ctime_t)
    record_type = _ptu_rec_type_r[tags["TTResultFormat_TTTRRecType"]["value"]]
    hw_type = tags["HW_Type"]
    if isinstance(hw_type, list):
        hw_type = hw_type[0]
    metadata = {
        "timestamps_unit": tags["MeasDesc_GlobalResolution"]["value"],  # both T3 and T2
        "acquisition_duration": acquisition_duration,
        "software": tags["CreatorSW_Name"]["data"],
        "software_version": tags["CreatorSW_Version"]["data"],
        "creation_time": creation_time,
        "hardware_name": hw_type["data"],
        "record_type": record_type,
        "tags": _convert_multi_tags(tags),
    }
    if record_type.endswith("T3"):
        metadata["nanotimes_unit"] = tags["MeasDesc_Resolution"]["value"]
        metadata["laser_repetition_rate"] = tags["TTResult_SyncRate"]["value"]

    return metadata


def read_ptu_metadata(filename: os.PathLike) -> dict:
    with open(filename, "rb") as f_obj:
        _check_ptu_magic(f_obj)
        tags = _read_ptu_tags(f_obj)

    metadata = convert_ptu_tags(tags)

    return metadata
