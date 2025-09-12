# %%
from pathlib import Path

from dont_fret.process import batch_search_and_save, search_and_save

# %%

cwd = Path(__file__).parent
test_data_dir = cwd.parent / "tests" / "test_data" / "input" / "ds2"
ptu_files = list(test_data_dir.glob("*.ptu"))

# %%
search_and_save(ptu_files[0], output_type=".csv")

# %%
batch_search_and_save(ptu_files)
# %%
