########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Plot                                                         #
# Classes: APlot [better name tbc]                                     #
# Description: Wrapper object that takes and stores a passed Record    #
#   object and implements plotting methods on it.                      #
########################################################################
# TODO:
# - Formatting code (e.g. theme_bw)
# - Overlaying/averaging multiple runs
# - Restore code for limits
# - Write tests

import numpy as np, pandas as pd, ggplot, os, shutil
try:
       import cPickle as pickle
except:
       import pickle

class Plotter:
    """Wrapper class for storing a Record and its associated plots."""

    def __init__(self, record):
        """Import a Record object and initialise plotting methods."""
        rfile = open(record, "rb")
        try:
            self.record = pickle.load(rfile)
            self.plot_methods = ["plot_population_resources"]
            self.plot_names = ["0_pop-res"]
            self.plots = []
        finally:
            rfile.close()

    def generate_plots(self):
        for m in self.plot_methods:
            self.plots.append(getattr(self, m)())

    def save_plots(self):
        # Make/replace output directory
        pm,pn,p = self.plot_methods, self.plot_names, self.plots
        if not len(pm) == len(pn) == len(p):
            errstr = "Plot names, methods and images are of different lengths."
            raise ValueError(errstr)
        outdir = self.record["output_prefix"] + "_plots"
        if os.path.exists(outdir): # Overwrite existing output
                shutil.rmtree(outdir)
        os.makedirs(outdir)
        for n in xrange(len(self.plots)):
            outpath = os.path.join(outdir, self.plot_names[n] + ".png")
            self.plots[n].save(outpath)


    def get_flattened_coords(self, key, dim, source=""):
        """Given an array and a dimension, returns the original
        co-ordinate in that dimension of each element in a flattened
        version of the input array."""
        # Import data and verify dimensions
        source = source if source else self.record
        arr = source[key]
        values = np.arange(arr.shape[dim]) # Set of possible co-ord values
        shape = [int(np.prod(arr.shape[dim+1:])),
                int(np.prod(arr.shape[:dim+1]))/arr.shape[dim]]
        return np.tile(values, shape).T.flatten().astype(int)

    def make_dataframe(self, keys, dimlabels, subkeys="", source=""):
        """Convert one or more Record entries into a data-frame, with
        columns for the corresponding key, the value, and the
        co-ordinates in the entry array."""
        # check subkeys valid
        if subkeys:
            if not isinstance(subkeys, list):
                raise TypeError("subkeys is of invalid type: {}".format(type(subkeys)))
            elif not all(isinstance(k, str) for k in subkeys):
                raise TypeError("subkeys elements are of invalid type: {}".format(type(subkeys[0])))

        source = source if source else self.record
        df = pd.DataFrame()
        for key in keys:
            val = source[key]
            if isinstance(val, dict):
                df = df.append(self.make_dataframe(val.keys(),
                    dimlabels, "", val))
            else:
                ddict = {"key":key,"value":val.flatten()}
                for n in xrange(len(dimlabels)):
                    ddict[dimlabels[n]] = self.get_flattened_coords(
                            key, n, source)
                df = df.append(pd.DataFrame(ddict))

        if subkeys:
            return df[df.key.isin(subkeys)]
        else:
            return df

    def solo_plot(self, keys, dimlabels, geoms, subkeys="", overlay=False):
        """Create a single plot, either of multiple data series (overlay=False)
        or of overlayed slices of a single series (overlay=True)."""
        data = self.make_dataframe(keys, dimlabels, subkeys)
        if overlay: data[dimlabels[0]] = data[dimlabels[0]].astype(str)
        plot = ggplot.ggplot(data, ggplot.aes(x=dimlabels[int(overlay)],
            y="value", color = dimlabels[0] if overlay else "key"))
        for g in geoms:
            plot += getattr(ggplot, "geom_"+g)()
        return plot

    def grid_plot(self, keys, dimlabels, facet, geoms, subkeys=""):
        """Create a grid of plots, showing corresponding slices of one
        or more data series (e.g. at different time points."""
        data = self.make_dataframe(keys, dimlabels, subkeys="")
        xlabel = dimlabels[1] if dimlabels[0] == facet else dimlabels[0]
        plot = ggplot.ggplot(data, ggplot.aes(x=xlabel, y="value",
            color = "key")) + ggplot.facet_wrap(facet)
        for g in geoms:
            plot += getattr(ggplot, "geom_"+g)()
        return plot

    # Plot types

    def stage_trace(self, keys):
        """Simple per-stage line trace for specified Record keys."""
        return self.solo_plot(keys, ["stage"], ["step"])

    def snapshot_trace(self, keys):
        """Simple per-snapshot line + point trace for specified Record
        keys"""
        return self.solo_plot(keys, ["snapshot"], ["line","point"])

    def snapshot_overlay(self, key, dimlabel):
        """Per-snapshot overlay line plot of a single Record key."""
        return self.solo_plot([key], ["snapshot", dimlabel], ["line"], True)

    def plot_population_resources(self):
        # TODO: enable swapping order of series for constant v variable resources
        # TODO: Set limits at object level?
        return self.stage_trace(["population_size", "resources"])

    def plot_starvation(self):
        # TODO: plot y-axis in log-space of base matching starvation increment
        return self.stage_trace(["surv_penf","repr_penf"])

    def plot_age_distribution(self):
        return self.snapshot_overlay(["snapshot_age_distribution"], "age")

    def plot_fitness(self):
        return self.snapshot_trace(["fitness", "junk_fitness"])

    def plot_per_age_fitness(self):
        return self.snapshot_overlay(["fitness_term"], "age")

    def plot_entropy_gt(self):
        return self.snapshot_trace(["entropy_gt"])

    def plot_entropy_bits(self):
        return self.snapshot_trace(["entropy_bits"])

    def plot_density(self):
        return self.grid_plot(["density"], ["genotype", "snapshot"],
                "snapshot", ["line"])
        # TODO: Enable plotting density overlays of single locus type
        # ("subkeys" argument to make_dataframe?)

    def plot_mean_gt(self):
        return self.grid_plot(["mean_gt"], ["snapshot","age"], "snapshot",
                ["point"])
        # TODO: To get this to work, will need subkey ability (like for
        # plot_density) and the ability to apply some sort of transformation
        # to the data
