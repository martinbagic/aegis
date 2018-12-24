import os, shutil, numpy as np, pandas as pd
from aegis.Core.functions import chance, quantile, fivenum, init_gentimes,\
        init_ages, init_genomes, init_generations, deep_key, deep_eq, make_windows,\
        correct_r_rate, make_mapping
from aegis.Core.Population import Population
from aegis.Core.Config import Config
from aegis.Core.Record import Record
from aegis.Core.Run import Run
from aegis.Core.Simulation import Simulation
from aegis.Core.Plotter import Plotter

try:
    import cPickle as pickle
except:
    import pickle

###########
### run ###
###########

def run(config_file, report_n, verbose):
    """Execute a complete simulation from a specified config file."""
    s = Simulation(config_file, report_n, verbose)
    s.execute()
    s.finalise()

###########
### get ###
###########

def getconfig(outpath):
    """Create a default config file at the specified destination."""
    dirpath = os.path.dirname(os.path.realpath(__file__))
    inpath = os.path.join(dirpath, "config_default.py")
    shutil.copyfile(inpath, outpath)

############
### read ###
############

def getrseed(inpath, outpath):
    """Get prng from a record object."""
    fin = open(inpath, "r")
    record = pickle.load(fin)
    fin.close()
    fout = open(outpath, "w")
    pickle.dump(record["random_seed"], fout)
    fout.close()

def getrecinfo(inpath, outpath):
    """Get information on record object."""
    rec_name = inpath.split('/')[-1] # get record name
    # load record
    infile = open(inpath)
    rec = pickle.load(infile)
    infile.close()
    # formatting
    n_tabs = 3
    ml_type = 4
    ml_name = 4
    ml_shape = 5
    ml_subkeys = 7
    for key in rec.keys():
        ml_type = max(ml_type, len(str(type(rec[key]))))
        ml_name = max(ml_name, len(key))
        if isinstance(rec[key], np.ndarray): ml_shape = max(ml_shape,\
                len(str(rec[key].shape)))
        elif isinstance(rec[key], list): ml_shape = max(ml_shape,\
                len(str(len(rec[key]))))
        elif isinstance(rec[key], tuple): ml_shape = max(ml_shape,\
                len(str(len(rec[key]))))
        elif isinstance(rec[key], dict): ml_subkeys = max(ml_subkeys,\
                len(str(rec[key].keys())))
    ll = []
    ll_title = ['type'+(ml_type-4)*' '+'|'+n_tabs*'\t'+\
            'name'+(ml_name-4)*' '+'|'+n_tabs*'\t'+\
            'shape']
    ll_sep = ['-'*ml_type+'|'+'-'*(ml_name+4*n_tabs)+'|'+'-'*(ml_shape-1+4*n_tabs)]
    ll_dicts = []
    ll_dicts_sep = ['-'*ml_type+'|'+'-'*(ml_name+4*n_tabs)+'|'+\
            '-'*(ml_name+4*n_tabs)]
    ll_dicts_title = ['type'+(ml_type-4)*' '+'|'+n_tabs*'\t'+\
            'name'+(ml_name-4)*' '+'|'+n_tabs*'\t'+\
            'subkeys']
    for key in rec.keys():
        s = str(type(rec[key]))
        s += ' '*(ml_type-len(s))
        s += '|'+(n_tabs * '\t')
        s += key
        s += ' '*(ml_name-len(key))
        s += '|'+(n_tabs * '\t')
        if isinstance(rec[key], dict):
            s += str(rec[key].keys())
            ll_dicts.append(s)
        else:
            if isinstance(rec[key],np.ndarray): ss = str(rec[key].shape)
            elif isinstance(rec[key],list): ss = str(len(rec[key]))
            elif isinstance(rec[key],tuple): ss = str(len(rec[key]))
            else: ss= '/'
            s += ss+' '*(ml_shape-len(ss))
            ll.append(s)
    # finalise
    ll.sort()
    ll = ll_title + ll_sep + ll
    ll_dicts.sort()
    ll_dicts = ll_dicts_title + ll_dicts_sep + ll_dicts
    out = '\n'.join([rec_name+' record entries\n']+ll+['\ndictionaries\n']+ll_dicts)
    # write to file
    outfile = open(outpath, 'w')
    outfile.write(out)
    outfile.close()

def get_csv(inpath, outpath, last_K=500):
    """Only specific data is output from the Record file to a csv files.
    The pandas dataframes are organized with respect to dimensions of the belonging
    numpy arrays in Record."""

    # get rec
    infile = open(inpath)
    rec = pickle.load(infile)
    infile.close()

    # keys (some lists are used, some just here to give the overview)

    single_keys = [ "auto",\
                    "dieoff",\
                    "pen_cuml",\
                    "deltabar",\
                    "m_rate",\
                    "m_ratio",\
                    "r_rate",\
                    "r_rate_input",\
                    "scale",\
                    "chr_len",\
                    "kill_at",\
                    "maturity",\
                    "max_fail",\
                    "max_ls",\
                    "max_stages",\
                    "min_gen",\
                    "n_base",\
                    "n_neutral",\
                    "n_snapshots",\
                    "n_stages",\
                    "n_states",\
                    "output_mode",\
                    "res_start",\
                    "start_pop",\
                    "repr_pen",\
                    "surv_pen",\
                    "age_dist_N",\
                    "output_prefix",\
                    "path_to_seed_file",\
                    "repr_mode",\
                    # need special treatment since dicts
                    "g_dist_s",\
                    "g_dist_r",\
                    "g_dist_n",\
                    "n1_window_size"]

    nstagex1_keys = [   "poulation_size",\
                        "resources",\
                        # need special treatment since 5 columns
                        "generation_dist",\
                        "gentime_dist"]

    nstagexmaxls_keys = ["age_distribution"]

    maxlsx1_keys = ["surv_curve"] # special

    nsnapxmaxls_keys = ["cmv_surv",\
                        "fitness_term",\
                        "junk_cmv_surv",\
                        "junk_fitness_term",\
                        "junk_repr",\
                        "junk_repr_value",\
                        "mean_repr",\
                        "repr_value",\
                        "snapshot_age_distribution",\
                        "snapshot_gentime_distribution",\
                        # need special treatment since dict with surv and repr
                        "prob_mean",\
                        "prob_var"]

    nsnapxnbit_keys = [ "n1",\
                        "n1_var"]

    sliding_window_keys = [ "n1_window_mean",\
                            "n1_window_var"]

    nsnapxnloci_keys = ["mean_gt",\
                        "var_gt"]

    nsnapx1_keys = ["fitness"]

    # construct pandas objects

    # single values
    def single_series(rec, keys):
        ds = pd.Series()
        for key in keys[:-4]:
            ds[key] = rec[key]
        for ss in ["s","r","n"]:
            ds["g_dist_"+ss] = rec["g_dist"][ss]
        ds["n1_window_size"] = rec["windows"]["n1"]
        return ds

    # shape = (nstage,)
    def nstagex1_df(rec):
        df = pd.DataFrame()
        df["population_size"] = rec["population_size"].astype(int)
        df["resources"] = rec["resources"].astype(int)
        # generation distribution
        df["generation_min"] = rec["generation_dist"][:,0]
        df["generation_25_percentile"] = rec["generation_dist"][:,1]
        df["generation_median"] = rec["generation_dist"][:,2]
        df["generation_75_percentile"] = rec["generation_dist"][:,3]
        df["generation_max"] = rec["generation_dist"][:,4]
        # gentime distribution
        df["gentime_min"] = rec["gentime_dist"][:,0]
        df["gentime_25_percentile"] = rec["gentime_dist"][:,1]
        df["gentime_median"] = rec["gentime_dist"][:,2]
        df["gentime_75_percentile"] = rec["gentime_dist"][:,3]
        df["gentime_max"] = rec["gentime_dist"][:,4]
        return df

    # shape = (nstage, maxls)
    def nstagexmaxls_df(rec, keys):
        df = pd.DataFrame()
        sh = rec[keys[0]].shape
        df["stage"] = np.repeat(np.arange(sh[0]),sh[1])
        df["age"] =  np.tile(np.arange(sh[1]),sh[0])
        for key in keys:
            df[key] = rec[key].flatten()
        return df

    def compute_surv_curve(rec,last_K=last_K):
        # get distribution of ages per stage in total numbers
        agedist = rec["age_distribution"][np.where(rec["population_size"]>0)][-last_K:]
        popn = rec["population_size"][np.where(rec["population_size"]>0)][-last_K:]
        agehist = np.repeat(popn,agedist.shape[1]).reshape(agedist.shape)*agedist
        sh = agehist.shape
        # construct indices so as to track survival groups
        ix1 = np.tile(np.arange(sh[1]),sh[0]-sh[1]).reshape((sh[0]-sh[1],sh[1]))+\
                np.repeat(np.arange(sh[0]-sh[1]),sh[1]).reshape((sh[0]-sh[1],sh[1]))
        ix2 = np.tile(np.arange(sh[1]),sh[0]-sh[1]).reshape(ix1.shape)
        surv_curve = agehist[ix1,ix2]
        surv_curve /= np.tile(surv_curve[:,0].reshape((surv_curve.shape[0],1)),\
                surv_curve.shape[1]).astype("float")
        return (surv_curve.mean(0), surv_curve.std(0))

    # shape = (maxls,)
    def maxlsx1_df(rec):
        df = pd.DataFrame()
        sc_mean, sc_std = compute_surv_curve(rec)
        df["age"] = np.arange(sc_mean.size)
        df["surv_mean"] = sc_mean.flatten()
        df["surv_std"] = sc_std.flatten()
        return df

    # shape = (nsnap, maxls)
    def nsnapxmaxls_df(rec, keys):
        df = pd.DataFrame()
        sh = rec[keys[0]].shape
        df["snap"] = np.repeat(np.arange(sh[0]),sh[1])
        df["age"] =  np.tile(np.arange(sh[1]),sh[0])
        for key in keys[:-2]:
            df[key] = rec[key].flatten()
        for ss in ["mean", "var"]:
            df["surv_prob_"+ss] = rec["prob_"+ss]["surv"].flatten()
            df["repr_prob_"+ss] = np.concatenate((np.zeros((sh[0],rec["maturity"])),\
                                    rec["prob_"+ss]["repr"]),axis=1).flatten()
        return df

    # shape = (snap, nbit)
    def nsnapxnbit_df(rec, keys):
        df = pd.DataFrame()
        sh = rec[keys[0]].shape
        df["snap"] = np.repeat(np.arange(sh[0]),sh[1])
        df["bit"] =  np.tile(np.arange(sh[1]),sh[0])
        df["type"] = np.tile(np.concatenate((\
                        np.repeat("surv",rec["max_ls"]*rec["n_base"]),\
                        np.repeat("repr",(rec["max_ls"]-rec["maturity"])*rec["n_base"]),\
                        np.repeat("neut",rec["n_neutral"]*rec["n_base"]))),\
                        sh[0])
        for key in keys:
            df[key] = rec[key].flatten()
        return df

    # special: sliding window
    def sliding_window_df(rec, keys):
        df = pd.DataFrame()
        sh = rec[keys[0]].shape
        df["snap"] = np.repeat(np.arange(sh[0]),sh[1])
        df["bit"] =  np.tile(np.arange(sh[1]),sh[0])
        for key in keys:
            df[key] = rec[key].flatten()
        return df

    # shape = (nsnap, nloci)
    def nsnapxnloci_df(rec):
        df = pd.DataFrame()
        sh = rec["mean_gt"]["a"].shape
        df["snap"] = np.repeat(np.arange(sh[0]),sh[1])
        df["bit"] =  np.tile(np.arange(sh[1]),sh[0])
        df["type"] = np.tile(np.concatenate((\
                        np.repeat("surv",rec["max_ls"]),\
                        np.repeat("repr",(rec["max_ls"]-rec["maturity"])),\
                        np.repeat("neut",rec["n_neutral"]))),\
                        sh[0])
        df["mean_gt"] = rec["mean_gt"]["a"].flatten()
        df["var_gt"] = rec["var_gt"]["a"].flatten()
        return df

    # shape = (nsnap,)
    def nsnapx1_df(rec, keys):
        df = pd.DataFrame()
        df["snap"] = np.arange(rec[keys[0]].size)
        for key in keys:
            df[key] = rec[key]
        return df

    # create dir in outpath and update outpath
    outpath = os.path.join(outpath,"csv_files")
    if not os.path.exists(outpath): os.makedirs(outpath)

    # output to csv
    single_series(rec, single_keys).to_csv(os.path.join(outpath,"single.csv"),index=False)
    nstagex1_df(rec).to_csv(os.path.join(outpath,"nstage-x-1.csv"),index=False)
    nstagexmaxls_df(rec,nstagexmaxls_keys).to_csv(os.path.join(outpath,\
            "nstage-x-maxls.csv"),index=False)
    if last_K <= rec["population_size"].size:
        maxlsx1_df(rec).to_csv(os.path.join(outpath,"maxls-x-1.csv"),index=False)
    nsnapxmaxls_df(rec,nsnapxmaxls_keys).to_csv(os.path.join(outpath,\
            "nsnap-x-maxls.csv"),index=False)
    nsnapxnbit_df(rec,nsnapxnbit_keys).to_csv(os.path.join(outpath,\
            "nsnap-x-nbit.csv"),index=False)
    sliding_window_df(rec,sliding_window_keys).to_csv(os.path.join(outpath,\
            "sliding_window.csv"),index=False)
    nsnapxnloci_df(rec).to_csv(os.path.join(outpath,"nsnap-x-nloci.csv"),index=False)
    nsnapx1_df(rec,nsnapx1_keys).to_csv(os.path.join(outpath,"nsnap-x-1.csv"),index=False)

############
### plot ###
############

def plot(record_file):
    a = Plotter(record_file)
    a.generate_figures()
    a.save_figures()

def plot_n1_sliding_window(record_file, wsize):
    a = Plotter(record_file)
    a.compute_n1_windows(wsize)
    a.gen_save_single("n1_mean_sliding_window")
    a.gen_save_single("n1_var_sliding_window")
    a.gen_save_single("n1_mean_sliding_window_grid")
    a.gen_save_single("n1_var_sliding_window_grid")
