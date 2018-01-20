################################################
## GENOME SIMULATION v.2.1 CONFIGURATION FILE ##
################################################

## CORE PARAMETERS ##
random_seed = "" # If numeric, sets random seed to that value before execution
output_prefix = "test" # Prefix for output files within simulation directory
number_of_runs = 1 # Total number of independent runs
number_of_stages = "auto" # Total number of stages per run [int/"auto"]
number_of_snapshots = 5 # Points in run at which to record detailed data
path_to_seed_file = "scen2_run0.pop" # Path to simulation seed file, if no seed then ""
    # see README for which parameters are inherited from seed, which are
    # defined anew in this config file
output_mode = 1 # 0 = return records only, 1 = return records + final pop,
                # 2 = return records + all snapshot populations
max_fail = 10 # Maximum number of failed attempts tolerated for each run

## STARTING PARAMETERS ##
repr_mode = 'asexual' # sexual, asexual, assort_only or recombine_only
res_start = 1000 # Starting resource value
res_var = True # Resources vary with population and time; else constant
start_pop = 500 # Starting population size

## RESOURCE PARAMETERS ##
V = 1.6 # Geometric resource regrowth factor, if variable
R = res_start # Arithmetic resource increment, if variable
res_limit = res_start*5 # Maximum resource value, if variable; -1 = infinite

## AUTOCOMPUTING STAGE NUMBER ##
delta = 10**-10 # Maximum difference between final and equilibrium neutral genotypes
scale = 1.1 # Scaling factor applied to target generation estimated for delta
max_stages = 500000 # Maximum number of stages to run before terminating

## SIMULATION FUNDAMENTALS: CHANGE WITH CARE ##
death_bound = [0.001, 0.02] # min and max death rates
repr_bound = [0, 0.2] # min and max reproduction rates
r_rate = 0.01 # recombination rate, if sexual
m_rate = 0.001 # mutation rate
m_ratio = 0.1 # Ratio of positive (0->1) to negative (1->0) mutations
g_dist_s = 0.5 # Proportion of 1's in survival loci of initial genomes
g_dist_r = g_dist_s #                       reproductive loci
g_dist_n = g_dist_s #                       neutral loci
n_neutral = 10 # Number of neutral loci in genome
n_base = 10 # Number of bits per locus
repr_offset = 100 # Offset for repr loci in genome map (must be <= max_ls)
neut_offset = 200 # Offset for neut loci (<= repr_offset + max_ls - maturity)
max_ls = 98 # Maximum lifespan (must be > repr_offset) (-1 = infinite)
maturity = 21 # Age from which an individual can reproduce (must be <= max_ls)
surv_pen = True # Survival penalty under starvation
repr_pen = False # Reproduction penalty under starvation
death_inc = 3 # Per-stage death rate increase under starvation
repr_dec = death_inc # Per-stage reproduction rate decrease under starvation
# Size of sliding windows for recording averaged statistics:
windows = {"population_size": 1000, "resources":1000, "n1":10}
