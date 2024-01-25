import pandas as pd
import pathlib
import logging
import json
import yaml
from aegis.help.config import get_default_parameters


class Container:
    """Wrapper class

    Contains paths to output files which it can read and return.
    """

    def __init__(self, basepath):
        self.basepath = pathlib.Path(basepath).absolute()
        self.name = self.basepath.stem
        self.paths = {
            path.stem: path for path in self.basepath.glob("**/*") if path.is_file() and path.suffix == ".csv"
        }
        self.paths["log"] = self.basepath / "progress.log"
        self.paths["output_summary"] = self.basepath / "output_summary.json"
        self.paths["input_summary"] = self.basepath / "input_summary.json"
        self.paths["snapshots"] = self.basepath / "snapshots"
        self.data = {}

        if not self.paths["log"].is_file():
            logging.error(f"No AEGIS log found at path {self.paths['log']}")

    def get_log(self, reload=False):
        if ("log" not in self.data) or reload:
            df = pd.read_csv(self.paths["log"], sep="|")
            df.columns = [x.strip() for x in df.columns]

            def dhm_inverse(dhm):
                nums = dhm.replace("`", ":").split(":")
                return int(nums[0]) * 24 * 60 + int(nums[1]) * 60 + int(nums[2])

            # TODO resolve deprecated function
            try:
                df[["ETA", "t1M", "runtime"]].map(dhm_inverse)
            except:
                df[["ETA", "t1M", "runtime"]].applymap(dhm_inverse)
            self.data["log"] = df
        return self.data["log"]

    def get_df(self, stem, reload=False):
        file_read = stem in self.data
        file_exists = stem in self.paths
        # TODO Read also files that are not .csv

        if not file_exists:
            logging.error(f"File {self.paths[stem]} does not exist")
        elif (not file_read) or reload:
            self.data[stem] = pd.read_csv(self.paths[stem])

        return self.data.get(stem, pd.DataFrame())

    def get_config(self):
        if "config" not in self.data:
            path = self.basepath.parent / f"{self.basepath.stem}.yml"
            with open(path, "r") as file_:
                custom_config = yaml.safe_load(file_)
            default_config = get_default_parameters()
            self.data["config"] = {**default_config, **custom_config}

        return self.data["config"]

    def get_json(self, stem):
        df = self.get_df(stem)
        json = df.T.to_json(index=False, orient="split")
        return json

    def get_output_summary(self):
        if self.paths["output_summary"].exists():
            with open(self.paths["output_summary"], "r") as file_:
                return json.load(file_)

    def get_input_summary(self):
        if self.paths["input_summary"].exists():
            with open(self.paths["input_summary"], "r") as file_:
                return json.load(file_)

    def get_snapshot(self, kind, index):
        paths = sorted(
            (self.paths["snapshots"] / kind).glob("*"),
            key=lambda path: int(path.stem),
        )

        if index < len(paths):
            return pd.read_feather(paths[index])

    def __str__(self):
        return self.name

    def get_json(self):
        return json.dumps(self)
