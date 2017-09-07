#!/usr/bin/env python

# Import libraries and define arguments
import argparse,os,sys,warnings,pyximport; pyximport.install()
from gs_core import Simulation

parser = argparse.ArgumentParser(description='Run the genome ageing \
        simulation.')
parser.add_argument('dir', help="path to simulation directory")
parser.add_argument('-o', metavar="<str>", default="output",
        help="prefix of simulation output file (default: output)")
parser.add_argument('-t', metavar="<int>", default=-1,
        help="number of cores to use for multiprocessing (default:\
                all available cores)")
parser.add_argument('-l', metavar="<str>", default="log",
        help="prefix of simulation log file (default: log)")
parser.add_argument('-s', default="",
        help="path to simulation seed file (default: no seed)")
parser.add_argument('-S', default=-1, 
        help="Run number in seed file from which to take seed population\
                (default: seed each run with the corresponding seed run)")
parser.add_argument('-m', type=int, metavar="<int>", default=10,
        help="maximum number of failed runs to repeat before accepting result"\
                +" (default: 10)")
parser.add_argument('-c', metavar='<str>', default="config",
        help="name of configuration file within simulation directory \
                (default: config.py)")
parser.add_argument('-r', type=int, metavar="<int>", default=100,
        help="report information every <int> stages (default: 100)")
parser.add_argument('-p', '--profile', action="store_true",
        help="profile genome simulation with cProfile")
parser.add_argument('-v', '--verbose', action="store_true",
        help="display full information at each report stage \
                (default: only starting population)")
args = parser.parse_args()

if args.profile:
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable() # start profiling

if args.s != "":
    args.s = os.path.abspath(args.s) # Get abspath before changing dir

#Change to simulation directory
try:
    sys.path.remove(os.getcwd())
    os.chdir(args.dir)
    sys.path = [os.getcwd()] + sys.path
except OSError:
    exit("Error: Specified simulation directory does not exist.")

with warnings.catch_warnings(DeprecationWarning):
    sim = Simulation(args.c, args.s, args.S, args.r, args.o, args.m, 
            args.verbose)
    sim.execute(int(args.t))
    sim.finalise(args.l)

if args.profile:
    pr.create_stats()
    pr.dump_stats('timestats.txt')
