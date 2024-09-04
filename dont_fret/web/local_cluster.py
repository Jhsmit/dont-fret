import time
from pathlib import Path
from urllib.parse import urlparse

import yaml
from distributed import LocalCluster

# import executor module such that sizeof functions are registered
from dont_fret.config import cfg

root = Path(__file__).parent.parent.parent
data = yaml.safe_load((root / "default_testing.yaml").read_text())
cfg.update(data)

address = cfg.web.executor.address  # type: ignore
parsed_adress = urlparse(address)


def blocking_cluster():
    """Start a dask LocalCluster and block until iterrupted"""

    try:
        local_cluster = LocalCluster(n_workers=6, threads_per_worker=2)
        print(f"Started local cluster at {local_cluster.scheduler_address}")
    except OSError:
        # print(f"Could not start local cluster with at port: {port}")
        raise
    try:
        loop = True
        while loop:
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                print("Interrupted")
                loop = False
    finally:
        local_cluster.close()


if __name__ == "__main__":
    blocking_cluster()
