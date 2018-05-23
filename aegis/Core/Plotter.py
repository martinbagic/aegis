########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Plot                                                         #
# Classes: Plotter                                                     #
# Description: Wrapper object that takes and stores a passed Record    #
#   object and implements plotting methods on it.                      #
########################################################################

from .functions import make_windows, timenow, get_runtime
import numpy as np, pandas as pd, os, shutil
import matplotlib
import matplotlib.pyplot as plt

try:
       import cPickle as pickle
except:
       import pickle

class Plotter:
    """Wrapper class for storing a Record and its associated plots."""

    def __init__(self, record):
        """Import a Record object and initialise plotting methods."""
        self.starttime = timenow(False)
        print "\nBeginning plotting {}.".format(timenow())
        print "Working directory: "+os.getcwd()
        print "Reading record from ./{}.".format(record)
        rfile = open(record, "rb")
        try:
            self.record = pickle.load(rfile)
            s = "Import succeeded. Time needed"
            print get_runtime(self.starttime, timenow(False), s)
            self.plot_methods = ["plot_population_resources",\
                                 "plot_phenotype_distribution",\
                                 "plot_phenotype_distribution_grouped",\
                                 #"plot_genetic_variance",\
                                 "plot_density",\
                                 "plot_phenotype_mean",\
                                 "plot_phenotype_var",\
                                 "plot_bits_mean",\
                                 "plot_bits_sliding_mean",\
                                 "plot_death_rate",\
                                 "plot_age_distribution"\
                                 ]
            self.plot_names = ["01_pop-res",\
                               "02_phtyp-dist",\
                               "03_phtyp-dist-group",\
                               #"04_gen-var",\
                               "04_density",\
                               "05_phtyp-mean",\
                               "06_phtyp-var",\
                               "07_bits",\
                               "08_bits_window",\
                               "09_death-rate",\
                               "10_age-dist"\
                               ]
            self.figures = []
        finally:
            rfile.close()
        # define universal aestethics
        self.fsize = (12,8) # figure size

    def generate_figures(self):
        for m in self.plot_methods:
            p = getattr(self,m)()
            if p: self.figures.append(p)

    def save_figures(self):
        # Remove not generated plot names
        if self.record["age_dist_N"] == "all":
            self.plot_methods.remove("plot_age_distribution")
            self.plot_names.remove("10_age-dist")
        # Make/replace output directory
        pm,pn,p = self.plot_methods, self.plot_names, self.figures
        if not len(pm) == len(pn) == len(p):
            errstr = "Plot names, methods and images are of different lengths."
            raise ValueError(errstr)
        outdir = self.record["output_prefix"] + "_plots"
        if os.path.exists(outdir): # Overwrite existing output
                shutil.rmtree(outdir)
        os.makedirs(outdir)
        print "\nSaving plot:"
        for n in xrange(len(self.figures)):
            outpath = os.path.join(outdir, self.plot_names[n] + ".png")
            print self.plot_names[n]
            self.figures[n].savefig(outpath)
        s = "\nSuccessfully saved all figures. Total runtime"
        print get_runtime(self.starttime, timenow(False), s)

    ###########
    ## plots ##
    ###########

    # vertical lines for maturity, reproduction, neutral
    def add_vlines(self, axes, color="black", expand=False):
        exp = self.record["n_base"] if expand else 1
        l1 = self.record["maturity"]*exp
        l2 = self.record["max_ls"]*exp
        l3 = (2*l2-l1)

        axes.axvline(l1,c=color,ls="--",lw=0.5)
        axes.axvline(l2,c=color,ls="--",lw=0.5)
        axes.axvline(l3,c=color,ls="--",lw=0.5)

    # population and resources
    def plot_population_resources(self):
        pop = self.record["population_size"]
        res = self.record["resources"]
        # subset to filled entries
        ix = pop > 0
        pop = pop[ix]
        res = res[ix]
        df = pd.DataFrame({"pop":pop, "res":res})
        plot = df.plot(figsize=self.fsize)
        fig = plot.get_figure()
        fig.suptitle("Population size")
        return fig

    # phenotype distribution
    def plot_phenotype_distribution(self):
        data = self.record["density_per_locus"]["a"]

        shape = data.shape
        nsnap = shape[0]
        nloci = shape[1]
        npts = shape[2]

        df = pd.DataFrame(data.reshape(nsnap*nloci,npts))
        df["snap"] = np.repeat(np.arange(nsnap),nloci)
        df["locus"] = np.tile(np.arange(nloci),nsnap)

        #d1 = np.sqrt(nsnap).round()
        #d2 = (nsnap/d1).round()
        #dims = [int(d1),int(d2)]
        #ix = [[i,j] for i in range(dims[0]) for j in range(dims[1])]
        cmap = matplotlib.cm.get_cmap("Spectral")
        colors = [cmap(i) for i in np.linspace(0,1,npts)]

        c = 0
        fig,axes = plt.subplots(nsnap,1,sharex=True,sharey=True,\
                figsize=self.fsize)
        for name,group in df.groupby("snap"):
            group[range(npts)].plot(kind="area", color=colors, legend=False,\
                    x=group["locus"],ax=axes[c])
            axes[c].set_title(name,loc="right")
            self.add_vlines(axes[c], color="white", expand=False)
            c+=1

        labels = ["least fit"]+["..."]*(npts-2)+["most fit"]
        #labels = np.arange(npts).astype(str)
        lines = axes[0].get_lines()
        fig.legend(lines,labels,loc="center right")
        fig.suptitle("Phenotype distribution")
        return fig

    # phenotype distribution grouped
    def plot_phenotype_distribution_grouped(self):
        """(least fit, middle, 3 top)"""
        data = self.record["density_per_locus"]["a"]

        shape = data.shape
        nsnap = shape[0]
        nloci = shape[1]
        npts = shape[2]

        df = pd.DataFrame(data.reshape(nsnap*nloci,npts))
        df["snap"] = np.repeat(np.arange(nsnap),nloci)
        df["locus"] = np.tile(np.arange(nloci),nsnap)

        #d1 = np.sqrt(nsnap).round()
        #d2 = (nsnap/d1).round()
        #dims = [int(d1),int(d2)]
        #ix = [[i,j] for i in range(dims[0]) for j in range(dims[1])]
        cmap = matplotlib.cm.get_cmap("Spectral")
        rg = np.linspace(0,1,3)
        rg = np.hstack((rg[0],np.repeat(rg[1],npts-4),np.repeat(rg[2],3)))
        colors = [cmap(i) for i in rg]

        c = 0
        fig,axes = plt.subplots(nsnap,1,sharex=True,sharey=True,\
                figsize=self.fsize)
        for name,group in df.groupby("snap"):
            group[range(npts)].plot(kind="area", color=colors, legend=False,\
                    x=group["locus"],ax=axes[c])
            axes[c].set_title(name,loc="right")
            self.add_vlines(axes[c], color="white", expand=False)
            c+=1

        labels = ["least fit"]+["..."]*(npts-2)+["most fit"]
        #labels = np.arange(npts).astype(str)
        lines = axes[0].get_lines()
        fig.legend(lines,labels,loc="center right")
        fig.suptitle("Phenotype distribution grouped")
        return fig

    # genetic variance
#    def plot_genetic_variance(self):
#        data = self.record["genetic_variance"]
#        nsnap = data.shape[0]
#        df = pd.DataFrame(data.T)
#
#        fig,axes = plt.subplots(figsize=self.fsize)
#        # leave out first since it messes up scale
#        plot = df[range(1,nsnap)].plot(figsize=self.fsize)
#        plot.set_title("Genetic variance")
#        fig = plot.get_figure()
#        return fig
#
#    def plot_genetic_variance_subplots(self):
#        data = self.record["genetic_variance"]
#        nsnap = data.shape[0]
#        df = pd.DataFrame(data.T)
#
#        fig,axes = plt.subplots(figsize=self.fsize)
#        # leave out first since it messes up scale
#        plot = df[range(1,nsnap)].plot(subplots=True,figsize=self.fsize) # now an np.array
#        plot.set_title("Genetic variance")
#        fig = plot.get_figure()
#        return fig

    # density
    def plot_density(self):
        data = self.record["density"]["a"]
        df = pd.DataFrame(data.T)
        plot = df.plot(figsize=self.fsize)
        fig = plot.get_figure()
        fig.suptitle("Density")
        return fig

    # phenotype mean
    def plot_phenotype_mean(self):
        """Plot the mean for phenotype value for last snapshot taken."""
        mean = self.record["mean_gt"]["a"]
        var = self.record["mean_gt"]["a"]
        nsnap = mean.shape[0]
        nloci = mean.shape[1]

        df = pd.DataFrame()
        df["mean"] = mean.reshape(nsnap*nloci)
        df["var"] = var.reshape(nsnap*nloci)
        df["snap"] = np.repeat(np.arange(nsnap),nloci)
        df["locus"] = np.tile(np.arange(nloci),nsnap)

        fig, axes = plt.subplots(figsize=self.fsize)
        df[df["snap"]==nsnap-1].plot.scatter(x="locus", y="mean", ax=axes)
        #df[df["snap"]==nsnap-1].plot.scatter(x="locus", y="mean", s=df["var"]*100, ax=axes)
        self.add_vlines(axes, color="black", expand=False)
        fig.suptitle("Phenotype mean value")
        return fig

    # phenotype variance
    def plot_phenotype_var(self):
        """Plot the variance for phenotype value for last snapshot taken."""
        var = self.record["var_gt"]["a"]
        nsnap = var.shape[0]
        nloci = var.shape[1]

        df = pd.DataFrame()
        df["var"] = var.reshape(nsnap*nloci)
        df["snap"] = np.repeat(np.arange(nsnap),nloci)
        df["locus"] = np.tile(np.arange(nloci),nsnap)

        fig, axes = plt.subplots(figsize=self.fsize)
        df[df["snap"]==nsnap-1].plot.scatter(x="locus", y="var", ax=axes)
        self.add_vlines(axes, color="black", expand=False)
        fig.suptitle("Phenotype value variance")
        return fig

    # bits
    def plot_bits(self, key, title):
        """Plot bit value distribution for last snapshot taken."""
        bits = self.record[key]
        nsnap = bits.shape[0]
        nbits = bits.shape[1]

        df = pd.DataFrame()
        df["value"] = bits.reshape(nsnap*nbits)
        df["snap"] = np.repeat(np.arange(nsnap),nbits)
        df["bit"] = np.tile(np.arange(nbits),nsnap)

        def find_dim(n):
            return int(np.ceil(np.sqrt(n)))

        fig, axes = plt.subplots(find_dim(nsnap),find_dim(nsnap),figsize=self.fsize,sharex=True,sharey=True)
        axes = axes.flatten()

        c = 0
        for name,group in df.groupby("snap"):
            group.plot(kind="scatter", x="bit", y="value", s=5, legend=False,\
                    ax=axes[c])
            axes[c].set_title(name,loc="right")
            #axes[c].set_xlim(0,1)
            axes[c].set_ylim(0,1)
            self.add_vlines(axes[c], color="black", expand=True)
            c += 1

        fig.suptitle(title)
        return fig

    def plot_bits_mean(self):
        return self.plot_bits("n1", "Bit value distribution")

    def plot_bits_sliding_mean(self):
        return self.plot_bits("n1_window_mean", "Bit value distribution - sliding window")

    # death rate
    # TODO what when age_dist_N="all"
    def plot_death_rate(self):
        self.record.compute_actual_death()
        data = self.record["actual_death_rate"].mean(1)
        df = pd.DataFrame(data.T)
        plot = df.plot(figsize=self.fsize)
        fig = plot.get_figure()
        fig.suptitle("Death rate")
        return fig

    def plot_death_rate_subplots(self):
        self.record.compute_actual_death()
        data = self.record["actual_death_rate"].mean(1)
        df = pd.DataFrame(data.T)
        plot = df.plot(subplots=True,figsize=self.fsize) # now an np.array
        fig = plot.get_figure()
        fig.suptitle("Death rate")
        return fig

    # age distribution
    # TODO what when age_dist_N="all"
    def plot_age_distribution(self):
        if self.record["age_dist_N"] == "all": return
        self.record.compute_snapshot_age_dist_avrg()
        data = self.record["snapshot_age_distribution_avrg"]
        df = pd.DataFrame(data.T)
        plot = df.plot(figsize=self.fsize)
        fig = plot.get_figure()
        fig.suptitle("Age distribution")
        return fig
