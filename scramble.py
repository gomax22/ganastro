import os
import argparse
from astropy.io import fits
import numpy as np
from pathlib import Path
from classes.night import Night
import multiprocessing
from tqdm import tqdm


def check_args(args):
    # check if dataset directory exists
    if not Path(args["dataset"]).exists():
        raise ValueError(f"Dataset directory {args['dataset']} does not exist.")
    
    # check if samples per night is positive
    if args["samples_per_night"] <= 0:
        raise ValueError("Samples per night must be a positive integer.")
    
    # check if k is positive
    if args["k"] <= 0:
        raise ValueError("K must be a positive integer.")
    
    # check if sampling ratio is positive
    if args["sampling_ratio"] <= 0:
        raise ValueError("Sampling ratio must be a positive float.")
    
    # check if max nights is positive
    if args["max_nights"] is not None and args["max_nights"] <= 0:
        raise ValueError("Max nights must be a positive integer.")
    
    # check if cutoff begin is positive
    if args["cutoff_begin"] is not None and args["cutoff_begin"] <= 0:
        raise ValueError("cutoff begin must be a positive float.")
    
    # check if cutoff end is positive
    if args["cutoff_end"] is not None and args["cutoff_end"] <= 0:
        raise ValueError("cutoff end must be a positive float.")
    
    # check if cutoff end is greater than cutoff begin
    if args["cutoff_begin"] is not None and args["cutoff_end"] is not None and args["cutoff_end"] < args["cutoff_begin"]:
        raise ValueError("cutoff end must be greater than cutoff begin.")
    
    # check if concurrency is a boolean
    if not isinstance(args["concurrency"], bool):
        raise ValueError("Concurrency must be a boolean.")
    
    return True

def main(args):
    # get arguments
    dataset = args["dataset"]
    output = args["output"]
    samples_per_night = args["samples_per_night"]
    k = args["k"]
    sampling_ratio = args["sampling_ratio"]
    max_nights = args["max_nights"]
    cutoff_begin = args["cutoff_begin"]
    cutoff_end = args["cutoff_end"]
    concurrency = args["concurrency"]
    
    # print input parameters
    print(f"Dataset: {dataset}")
    print(f"Output: {output}")
    print(f"Samples per night: {samples_per_night}")
    print(f"K: {k}")
    print(f"Sampling ratio: {sampling_ratio}")
    print(f"cutoff begin: {cutoff_begin}")
    print(f"cutoff end: {cutoff_end}")
    print(f"Concurrency: {concurrency} (no. of cores: {multiprocessing.cpu_count()})")

    # create output directory
    if not Path(output).exists():
        Path(output).mkdir(parents=True, exist_ok=True)

    # list nights stored in the data directory
    # night_paths = [os.path.join(dataset, entry) for entry in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, entry)) and entry != "tars"]
    # print(f"Found {len(night_paths)} nights in the dataset directory")

    # for night_path in night_paths:
    print(f"\nProcessing night {dataset}")

    # get observations from night
    night = Night.from_directory(dataset, cutoff_begin, cutoff_end)
    
    # interpolate the night observations
    night.interpolate()

    # generate samples of night from this night
    date = dataset.split(os.sep)[-1]
    night.generate(
        k=k, 
        samples_per_night=samples_per_night, 
        sampling_ratio=sampling_ratio,
        max_nights=max_nights, 
        out_path=output, 
        date=date,
        concurrency=concurrency)
    
    # print("\n")
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate scrambled nights from night observations")
    ap.add_argument("-d", "--dataset", required=True, help="path to data directory")
    ap.add_argument("-o", "--output", required=True, help="path to output directory")
    ap.add_argument("-s", "--samples-per-night", required=True, type=int, default=58, help="number of samples to generate per night")
    ap.add_argument("-k", "--k", required=True, default=5, type=int, help="number of observations to combine")
    ap.add_argument("-r", "--sampling-ratio", required=False, default=1.0, type=float, help="sampling ratio")
    ap.add_argument("-m", "--max-nights", required=False, default=None, type=int, help="number of nights to generate")
    ap.add_argument("-b", "--cutoff-begin", required=False, default=None, type=float, help="cutoff beginning of wavelength range (in Angstroms)")
    ap.add_argument("-e", "--cutoff-end", required=False, default=None, type=float, help="cutoff end of wavelength range (in Angstroms)")
    ap.add_argument("-c", "--concurrency", required=False, default=True, type=bool, help="use concurrency")
    
    args = vars(ap.parse_args())
    check_args(args)
    main(args)