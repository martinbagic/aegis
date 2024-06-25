"""Data recorder

Records data generated by the simulation.

When thinking about recording additional data, consider that there are three recording methods:
    I. Snapshots (record data from the population at a specific stage)
    II. Flushes (collect data over time then flush)
    III. One-time records
    IV. Other: TE records
"""

import pathlib
import shutil

from .terecorder import TERecorder
from .picklerecorder import PickleRecorder
from .popgenstatsrecorder import PopgenStatsRecorder
from .visorrecorder import VisorRecorder
from .flushrecorder import FlushRecorder
from .featherrecorder import FeatherRecorder
from .phenomaprecorder import PhenomapRecorder
from .summaryrecorder import SummaryRecorder
from .progressrecorder import ProgressRecorder
from .ticker import Ticker

# TODO write tests


class RecordingManager:
    """
    Container class for various recorders.
    Each recorder records a certain type of data.
    Most recorders record data as tables, except SummaryRecorder and PickleRecorder which record JSON files and pickles (a binary python format).
    Headers and indexes of all tabular files are explicitly recorded.

    -----
    VISOR
    AEGIS records a lot of different data.
    In brief, AEGIS records
    genomic data (population-level allele frequencies and individual-level binary sequences) and
    phenotypic data (observed population-level phenotypes and intrinsic individual-level phenotypes),
    as well as
    derived demographic data (life, death and birth tables),
    population genetic data (e.g. effective population size, theta), and
    survival analysis data (TE / time-event tables).
    Furthermore, it records metadata (e.g. simulation log, processed configuration files) and python pickle files.

    Recorded data is distributed in multiple files.
    Almost all data are tabular, so each file is a table to which rows are appended as the simulation is running.
    The recording rates are frequencies at which rows are added; they are expressed in simulation stages.
    """

    def __init__(self, custom_config_path, overwrite):
        odir = self.make_odir(custom_config_path=custom_config_path, overwrite=overwrite)
        # self.paths = self.get_paths(odir)
        # TODO make subfolders
        self.make_subfolders(paths=[])

    # @staticmethod
    # def get_paths(odir: pathlib.Path) -> dict:
    #     return {
    #         "BASE_DIR": odir,
    #         "visor_spectra": odir / "visor" / "spectra",
    #         "pickles": odir / "pickles",
    #         "popgen": odir / "popgen",
    #         "te": odir / "te",
    #     }

    def initialize_recorders(self, TICKER_RATE):
        self.terecorder = TERecorder(odir=self.odir)
        self.picklerecorder = PickleRecorder(odir=self.odir)
        self.popgenstatsrecorder = PopgenStatsRecorder(odir=self.odir)
        self.visorrecorder = VisorRecorder(odir=self.odir)
        self.flushrecorder = FlushRecorder(odir=self.odir)
        self.featherrecorder = FeatherRecorder(odir=self.odir)
        self.phenomaprecorder = PhenomapRecorder(odir=self.odir)
        self.summaryrecorder = SummaryRecorder(odir=self.odir)
        self.progressrecorder = ProgressRecorder(odir=self.odir)
        self.ticker = Ticker(odir=self.odir, TICKER_RATE=TICKER_RATE)

    #############
    # UTILITIES #
    #############

    @staticmethod
    def make_odir(custom_config_path, overwrite) -> pathlib.Path:
        output_path = custom_config_path.parent / custom_config_path.stem  # remove .yml
        is_occupied = output_path.exists() and output_path.is_dir()
        if is_occupied:
            if overwrite:
                shutil.rmtree(output_path)
            else:
                raise Exception(f"{output_path} already exists. To overwrite, add flag --overwrite or -o.")
        return output_path

    @staticmethod
    def make_subfolders(paths):
        # TODO relink paths that are now in classes
        for path in paths:
            path.mkdir(exist_ok=True, parents=True)
