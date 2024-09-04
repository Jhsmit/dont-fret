import dont_fret.datagen as datagen
import numpy as np
from pathlib import Path

# __file__ = Path().resolve() / "script.py"
cwd = Path(__file__).parent

output_data_dir = cwd / "test_data" / "input" / "ds2"

ds_names = [
    "twostate",
    "threestate",
    "twostate_static",
]

for s in ds_names:
    func = getattr(datagen, s)
    data = func()
    np.savetxt(output_data_dir / f"{s}.txt", data)
