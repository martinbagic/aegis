# TODO universalize naming across files
# TODO maybe interpolate
# TODO update tests for fitness

# Libraries and arguments

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker
import argparse, os
try:
       import cPickle as pickle
except:
       import pickle

parser = argparse.ArgumentParser(description="Load record and plot\
        the simulation data.")
parser.add_argument("file", help="path to file containing record data")
parser.add_argument("-o", metavar="<str>", default="", help="path to directory\
        in which to deposit figures (default: directory of target file)")
parser.add_argument("-r", metavar="<int>", default=-1, help="if target file \
        contains a simulation object with multiple runs, specify from which \
        run to take record data (default: average of all runs)")
#parser.add_argument("-All", metavar="<bool>", default=False, help="toggle the All option (default: False)")

args = parser.parse_args()

# Process I/O
def get_record(target, run=-1):
    """Retrieve a Record object from a pickled REC or SIM file."""
    if not os.path.isfile(target):
        print "No such file: " + target
        q = raw_input("Enter correct file path, or skip to abort: ")
        if q == "": exit("Aborting.")
        return self.get_startpop(q, run)
    f = open(target, "rb")
    obj = pickle.load(f)
    if isinstance(obj, Record): return obj
    elif isinstance(obj, Run): return obj.record
    elif isinstance(obj, Simulation): 
        if run == -1: return obj.avg_record # Summary record
        else: return obj.runs[run].record # Record of specific run
    else: 
        x = "Inappropriate object type; cannot extract record data."
        raise TypeError(x)

def get_outdir(outpath, target):
    """Tests validity of output directory path and creates figure directory
    if necessary."""
    if outpath=="": outpath = os.path.dirname(target)
    if not os.path.isdir(outpath):
        print "No such directory: " + outpath
        q = raw_input("Enter path to existing directory, or skip to abort: ")
        if q == "": exit("Aborting.")
        return self.get_outdir(q, target)
    figdir = os.path.join(outpath, "figures")
    if not os.path.isdir(figdir):
        os.mkdir(figdir)
    return figdir

L = get_record(args.file, args.r)
O = get_outdir(args.o, args.file)


# plotting variables (not meant for UI)
tick_size = 7

### FULL LIST OF RECORD ITEMS ###
#    "n_bases" : number of bases making up one genetic unit
#    "maturity" : age at which sexual maturity is reached
#    "gen_map" : genome map for the run
#    "chr_len" : length of each chromosome in bits
#    "d_range" : range of possible death probabilities, from max to min
#    "r_range" : range of possible reproduction probabilities (min->max)
#    "snapshot_stages" : stages of run at which detailed info recorded
#    "population_size" : Value of N
#    "resources" : Resource level
#    "starvation_factor" : Value of x
#    "age_distribution" : Proportion of population at each age
#    "death_mean" : Mean genetic death probability at each age
#    "death_sd" : SD generic death probability at each age
#    "actual_death_rate" : per-age stage-to-stage fraction of survivors
#    "repr_mean" : Mean reproductive probability at each age
#    "repr_sd" : Mean reproductive probability at each age
#    "density_surv" : Distribution of number of 1's at survival loci
#    "density_repr" : Distribution of number of 1's at reproductive loci
#    "n1" : Average number of 1's at each position along the length of the chromosome
#    "n1_std" : n1 standard deviation
#    "age_wise_n1" : n1 averaged in intervals of n_bases
#    "age_wise_n1_std" : age_wise_n1 standard deviation
#    "s1" : Sliding-window SD of number of 1's along chromosome
#    "entropy" : Shannon-Weaver entropy across entire population array
#    "junk_death" : Average death probability as predicted from neutral locus
#    "junk_repr"  : Average reproductive probability as predicted from neutral locus
#    "fitness" : Average population fitness as defined in our article
#    "product_fitness" : fitness calculated as s_i*r_i
#    "junk_product_fitness" : fitness calculated as js_i*jr_i
#    "age_wise_fitness_contribution" : summands of fitness
#    "junk_age_wise_fitness_contribution" : summands of junk_fitness
#! Update this

######################
### PLOT FUNCTIONS ###
######################

# survival and standard deviation (2x1)
# colors represent values for snapshot stages with red being the most
# recent one, the blue line represents junk values for the most recent
# stage, vertical line is maturation age and the green line represents
# standard deviation

def save_close(name): 
    plt.savefig(os.path.join(O, name + ".png"))
    plt.close()

def survival(out=args.out):
    colormap = plt.cm.YlOrRd # select colormap
    colors = [colormap(i) for i in np.linspace(0.1, 0.8, L["n_snapshots"])] # subset colormap
    m = L["maturity"]
    y_bound = ((1-L["d_range"])[0], (1-L["d_range"])[-1])
    x_bound = (0, L["max_ls"])

    fig, ax = plt.subplots(2, sharex=True)

    # subplot 0
    for i in range(L["n_snapshots"]):
        ax[0].plot(1-L["death_mean"][i], color=colors[i]) # survival
    ax[0].plot((m,m),(y_bound[0],1), "k-") # maturity
    ax[0].plot(x_bound,(1-L["junk_death"][L["n_snapshots"]-1],1-L["junk_death"][L["n_snapshots"]-1]), "b-") # junk
    ax[0].set_ylim(y_bound)
    ax[0].set_xlim(x_bound)

    # subplot 1
    ax[1].plot(L["death_sd"][L["n_snapshots"]-1], color="green") # standard deviation
    ax[1].plot((m,m),(0,1), "k-") # maturity
    ax[1].set_ylim((0,max(L["death_sd"][L["n_snapshots"]-1])))
    # no need to set xlim because it's shared with subplot 0

    ax[0].set_ylabel("s", rotation="horizontal")
    ax[1].set_ylabel("s_sd", rotation="horizontal")
    ax[1].set_xlabel("age")
    fig.suptitle("Survival")
    save_close("survival")

# reproduction and standard deviation (2x1)
# colors represent values for snapshot stages with red being the most
# recent one, the blue line represents junk values for the most recent
# stage, vertical line is maturation age and the green line represents
# standard deviation
def reproduction(out=args.out):
    colormap = plt.cm.YlOrRd # select colormap
    colors = [colormap(i) for i in np.linspace(0.1, 0.8, L["n_snapshots"])] # subset colormap
    m = L["maturity"]
    y_bound = ((L["r_range"])[0], (L["r_range"])[-1])
    x_bound = (L["maturity"], L["max_ls"])

    fig, ax = plt.subplots(2, sharex=True)

    # subplot 0
    for i in range(L["n_snapshots"]):
        ax[0].plot(L["repr_mean"][i], color=colors[i]) # reproduction
    ax[0].plot(x_bound,(L["junk_repr"][L["n_snapshots"]-1],L["junk_repr"][L["n_snapshots"]-1]), "b-") # junk
    ax[0].set_ylim(y_bound)
    ax[0].set_xlim(x_bound)
    #ax[0].yaxis.set_ticks(np.linspace(L["r_range"][0],L["r_range"][-1],5))

    # subplot 1
    ax[1].plot(L["repr_sd"][L["n_snapshots"]-1], color="green") # standard deviation
    ax[1].set_ylim((0,max(L["repr_sd"][L["n_snapshots"]-1])))
    # no need to set xlim because it's shared with subplot 0

    ax[0].set_ylabel("r", rotation="horizontal")
    ax[1].set_ylabel("r_sd", rotation="horizontal")
    ax[1].set_xlabel("age")
    fig.suptitle("Reproduction")
    save_close("reproduction")

# population (blue) and resources (red)
# plot from stage s1 to stage s2, default: whole run
def pop_res(s1=0, s2=L["n_stages"], out=args.out):
    # based on L["var"], plot either res or pop_size on top
    if L["var"]: l1,l2 = plt.plot(L["resources"][s1:s2+1],"r-",L["population_size"][s1:s2+1],"b-")
    else: l2,l1 = plt.plot(L["population_size"][s1:s2+1],"b-",L["resources"][s1:s2+1],"r-")

    plt.figure(1).legend((l1,l2),("resources","population"),"upper right",prop={"size":7})
    plt.title("Resources and population")
    plt.xlabel("stage")
    plt.ylabel("N",rotation="horizontal")
    plt.axis([s1,s2,0,max(max(L["resources"][s1:s2+1]),max(L["population_size"][s1:s2+1]))])
    plt.xticks(np.linspace(0,s2-s1,6),map(str,(np.linspace(s1,s2,6)).astype(int)))

    save_close("pop_res")

# age distribution
# colors represent values for snapshot stages with red being the most
# recent one
def age_distribution(out=args.out):
    colormap = plt.cm.YlOrRd # select colormap
    colors = [colormap(i) for i in np.linspace(0.1, 0.8, L["n_snapshots"])] # subset colormap

    for i,j in zip(L["snapshot_stages"]-1,range(L["n_snapshots"])):
        plt.plot(L["age_distribution"][i], color=colors[j])

    plt.title("Age distribution")
    plt.xlabel("age")
    plt.ylabel("fraction",rotation="vertical")
    save_close("age_distribution")

# frequency of 1's
# (n1 is already sorted in age-ascending order, surv preceds repr)
# red lines mark maturation and where reproduction begins
# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted with the recording standard deviation
def frequency(out=args.out, All=args.All):
    mv = (L["maturity"]*L["n_bases"], L["maturity"]*L["n_bases"]) # maturity vertical
    rv = (L["max_ls"]*L["n_bases"], L["max_ls"]*L["n_bases"]) # reproduction vertical

    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["n_snapshots"])) # index for 4x4 subplots

        for i,j,k in ix:
            ax[i,j].scatter(range(L["chr_len"]),L["n1"][k],s=5,c="k",marker=".")
            ax[i,j].plot(mv, (0,1), "r--")
            ax[i,j].plot(rv, (0,1), "r-")
            ax[i,j].xaxis.set_ticks([0,mv[0],rv[0]])
            ax[i,j].set_xticklabels([0,mv[0][0],rv[0][0]],fontsize=tick_size)
            ax[i,j].set_yticklabels(np.linspace(0,1,6),fontsize=tick_size)
            ax[i,j].set_ylim((0,1))
            ax[i,j].set_xlim((0,L["chr_len"]))
        fig.text(0.03,0.55,"frequency",rotation="vertical",fontsize=12)
        fig.text(0.45,0.03,"position",rotation="horizontal",fontsize=12)
        fig.suptitle("Frequency of 1's")

    else:
        fig, ax = plt.subplots(2,sharex=True)

        # subplot 0
        ax[0].scatter(range(L["chr_len"]),L["n1"][-1],c="k")
        ax[0].plot(mv, (0,1), "r--")
        ax[0].plot(rv, (0,1), "r-")
        ax[0].xaxis.set_ticks([0,mv[0],rv[0]])
        ax[0].yaxis.set_ticks([0,1])
        ax[0].set_xlim((0,L["chr_len"]))
        ax[0].set_ylim((0,1))

        # subplot 1
        ax[1].plot(L["n1_std"][-1], "k-")
        ax[1].plot(mv, (0,1), "r--")
        ax[1].plot(rv, (0,1), "r-")
        ax[1].set_ylim((0,max(L["n1_std"][-1])))

        ax[0].set_ylabel("frequency", rotation="vertical")
        ax[1].set_ylabel("sd", rotation="vertical")
        ax[1].set_xlabel("position")
        fig.suptitle("Frequency of 1's")

    save_close("frequency_1s")

# age-wise frequency of 1's
# (age_wise_n1 is already sorted in age-ascending order, surv preceds
# repr)
# red lines mark maturation and where reproduction begins
# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted with the recording standard deviation
def age_wise_frequency(out=args.out, All=args.All):
    mv = (L["maturity"], L["maturity"]) # maturity vertical
    rv = (L["max_ls"], L["max_ls"]) # reproduction vertical

    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["n_snapshots"])) # index for 4x4 subplots

        for i,j,k in ix:
            ax[i,j].scatter(range(L["chr_len"]/L["n_bases"]),L["age_wise_n1"][k],s=7,c="k",marker=".")
            ax[i,j].plot(mv, (0,1), "r--")
            ax[i,j].plot(rv, (0,1), "r-")
            ax[i,j].xaxis.set_ticks([0,mv[0],rv[0]])
            ax[i,j].set_xticklabels([0,mv[0][0],rv[0]],fontsize=tick_size)
            ax[i,j].set_yticklabels(np.linspace(0,1,6),fontsize=tick_size)
            ax[i,j].set_ylim((0,1))
            ax[i,j].set_xlim((0,L["chr_len"]/L["n_bases"]))
        fig.text(0.03,0.55,"frequency",rotation="vertical",fontsize=12)
        fig.text(0.45,0.03,"position",rotation="horizontal",fontsize=12)
        fig.suptitle("Age-wise frequency of 1's")

    else:
        fig, ax = plt.subplots(2,sharex=True)

        # subplot 0
        ax[0].scatter(range(L["chr_len"]/L["n_bases"]),L["age_wise_n1"][-1],c="k")
        ax[0].plot(mv, (0,1), "r--")
        ax[0].plot(rv, (0,1), "r-")
        ax[0].xaxis.set_ticks([0,mv[0],rv[0]])
        ax[0].yaxis.set_ticks([0,1])
        ax[0].set_xlim((0,L["chr_len"]/L["n_bases"]))
        ax[0].set_ylim((0,1))

        # subplot 1
        ax[1].plot(L["age_wise_n1_std"][-1], "k-")
        ax[1].plot(mv, (0,1), "r--")
        ax[1].plot(rv, (0,1), "r-")
        ax[1].set_ylim((0,max(L["age_wise_n1_std"][-1])))

        ax[0].set_ylabel("frequency", rotation="vertical")
        ax[1].set_ylabel("sd", rotation="vertical")
        ax[1].set_xlabel("age")
        fig.suptitle("Age-wise frequency of 1's")

    save_plot("age_wise_frequency_1s")
    plt.close()

# density of genotypes (age unspecific)
# survival (green) and reproduction (red)
# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted
def density(out=args.out, All=args.All):
    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["n_snapshots"])) # index for 4x4 subplots
        y_bound = (0,np.around(max(np.max(L["density_surv"]),np.max(L["density_repr"])),2))

        for i,j,k in ix:
            ax[i,j].plot(L["density_surv"][k],"g-")
            ax[i,j].plot(L["density_repr"][k],"r-")
            ax[i,j].set_ylim(y_bound)
            ax[i,j].yaxis.set_ticks(np.linspace(y_bound[0],y_bound[1],5))
            ax[i,j].tick_params(axis="both",labelsize=7)
        fig.text(0.03,0.55,"genotype",rotation="vertical",fontsize=12)
        fig.text(0.45,0.03,"density",rotation="horizontal",fontsize=12)
        fig.suptitle("Genotype density")

    else:
        plt.plot(L["density_surv"][-1],"g-")
        plt.plot(L["density_repr"][-1],"r-")

        plt.ylabel("fraction", rotation="vertical")
        plt.xlabel("genotype")
        plt.title("Genotype density")

    save_plot("density")
    plt.close()

# observed death rate (calculated from population_size*age_distribution)
# averaged over s1:s2, default: last 100 stages
def observed_death_rate(out=args.out, s1=L["n_stages"]-100, s2=L["n_stages"]):
    plt.plot(np.mean(L["actual_death_rate"][s1:s2],0))
    plt.ylabel(r"$\mu$", rotation="horizontal")
    plt.xlabel("age")
    plt.title("Observed death rate")

    save_plot("observed_death_rate.png")
    plt.close()


def set_axis_labels(ax, ylab, xlab, xticks):
    ax.set_ylabel(ylab, rotation="horizontal")
    ax.set_xlabel(xlab)
    if xticks != "": ax.set_xticklabels(xticks)
    return ax
# entropy (shannon diversity index)
def shannon_diversity():
    """Plot Shannon entropy of population gene pool per snapshot."""
    fig,ax = plt.subplots()
    ax.plot(L["entropy"], "o-")
    ax = set_axis_labels("H", "stage", L["snapshot_stages"][::2])
    plt.title("Shannon diversity index")

    title_save_close("Shannon diversity intex","shannon_diversity")
    plt.close()

def fitness():
    """Plot population genomic fitness per snapshot."""
    fig,ax = plt.subplots()
    ax.plot(L["fitness"], "o-")
    ax = set_axis_labels("F", "stage", L["snapshot_stages"][::2])
    plt.title("Fitness")
    save_plot("fitness")
    plt.close()

# product fitness
# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted with the recording junk values
def product_fitness(out=args.out, All=args.All):
    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["n_snapshots"])) # index for 4x4 subplots

        for i,j,k in ix:
            ax[i,j].plot(L["product_fitness"][k])
            ax[i,j].set_xlim((L["maturity"],L["max_ls"]-1))
            ax[i,j].yaxis.set_major_locator(ticker.MaxNLocator(5)) # set tick number to 5
            ax[i,j].tick_params(axis="both",labelsize=7)
        fig.text(0.01,0.52,"$prod \\ F_{age}$",rotation="horizontal",fontsize=12)
        fig.text(0.48,0.03,"age",rotation="horizontal",fontsize=12)
        fig.suptitle("Product fitness")

    else:
        fig, ax = plt.subplots()

        l1 = ax.plot(L["product_fitness"][-1])
        l2 = ax.plot(L["junk_product_fitness"][-1], "g-")
        ax.set_xlim((L["maturity"],L["max_ls"]-1))

        blue_proxy = mpatches.Patch(color="blue", label="product $F_{age})$")
        green_proxy = mpatches.Patch(color="green", label="junk product $F_{age}$")
        ax.legend(handles=[blue_proxy,green_proxy],loc="upper right",prop={"size":7})
        ax.set_ylabel("product $F_{age}$", rotation="horizontal")
        ax.set_xlabel("age")
        plt.title("Product fitness")

    plt.savefig(out+"/figures/product_fitness.png")
    plt.close()

# age wise fitness contribution
# All determines wether all snapshot stages are plotted on a 4x4 figure
# or just the last is plotted with the recording junk values
def age_wise_fitness_contribution(out=args.out, All=args.All):
    if All:
        fig, ax = plt.subplots(4,4,sharex="col",sharey="row")
        ix = zip((0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),range(L["n_snapshots"])) # index for 4x4 subplots

        for i,j,k in ix:
            ax[i,j].plot(np.log(L["age_wise_fitness_contribution"][k]))
            ax[i,j].set_xlim((L["maturity"],L["max_ls"]-1))
            ax[i,j].yaxis.set_major_locator(ticker.MaxNLocator(5)) # set tick number to 5
            ax[i,j].tick_params(axis="both",labelsize=7)
        fig.text(0.01,0.52,"$log(F_{age})$",rotation="horizontal",fontsize=12)
        fig.text(0.48,0.03,"age",rotation="horizontal",fontsize=12)
        fig.suptitle("Age-wise fitness contribution")

    else:
        fig, ax = plt.subplots()

        l1 = ax.plot(np.log(L["age_wise_fitness_contribution"][-1]))
        l2 = ax.plot(np.log(L["junk_age_wise_fitness_contribution"][-1]), "g-")
        ax.set_xlim((L["maturity"],L["max_ls"]-1))

        blue_proxy = mpatches.Patch(color="blue", label="$log(F_{age})$")
        green_proxy = mpatches.Patch(color="green", label="junk $log(F_{age})$")
        ax.legend(handles=[blue_proxy,green_proxy],loc="upper right",prop={"size":7})
        ax.set_ylabel("$log(F_{age})$", rotation="horizontal")
        ax.set_xlabel("age")
        plt.title("Age-wise fitness contribution")

    plt.savefig(out+"/figures/age_wise_fitness_contribution.png")
    plt.close()

# collect all defined functions
# pr_s: pop_res stage
# d_s: observed_death_rate stage
def plot_all(out=args.out, All=args.All, pr_s1=0, pr_s2=L["n_stages"], d_s1=L["n_stages"]-100, d_s2=L["n_stages"]):
    survival(out=out)
    reproduction(out=out)
    pop_res(out=out, s1=pr_s1, s2=pr_s2)
    age_distribution(out=out)
    frequency(out=out, All=All)
    age_wise_frequency(out=out, All=All)
    density(out=out, All=All)
    observed_death_rate(out=out, s1=d_s1, s2=d_s2)
    shannon_diversity(out=out)
    fitness(out=out)
    product_fitness(out=out, All=All)
    age_wise_fitness_contribution(out=out, All=All)

###############
### EXECUTE ###
###############

plot_all()
