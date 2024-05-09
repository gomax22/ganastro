import os
import argparse
from astropy.io import fits
import numpy as np
from pathlib import Path
from classes.night import Night
import multiprocessing
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to data directory")
    ap.add_argument("-o", "--output", required=True, help="path to output directory")
    ap.add_argument("-s", "--samples-per-night", required=True, type=int, default=58, help="number of samples to generate per night")
    ap.add_argument("-k", "--k", required=True, default=5, type=int, help="number of observations to combine")
    ap.add_argument("-r", "--sampling-ratio", required=False, default=1.0, type=float, help="sampling ratio")
    ap.add_argument("-m", "--max-nights", required=False, default=None, type=int, help="number of nights to generate")
    ap.add_argument("-b", "--cut-begin", required=False, default=None, type=float, help="cut beginning of wavelength range")
    ap.add_argument("-e", "--cut-end", required=False, default=None, type=float, help="cut end of wavelength range")
    ap.add_argument("-c", "--concurrency", required=False, default=True, type=bool, help="use concurrency")
    
    args = vars(ap.parse_args())

    dataset = args["dataset"]
    output = args["output"]
    cut_begin = args["cut_begin"]
    cut_end = args["cut_end"]
    samples_per_night = args["samples_per_night"]
    samples_per_night = samples_per_night if samples_per_night is not None and samples_per_night > 0 else 58
    k = args["k"]
    sampling_ratio = args["sampling_ratio"]
    sampling_ratio = sampling_ratio if sampling_ratio is not None and sampling_ratio > 0 else 1.0
    max_nights = args["max_nights"]
    max_nights = max_nights if max_nights is not None and max_nights > 0 else None
    concurrency = args["concurrency"]
    

    # print input parameters
    print(f"Dataset: {dataset}")
    print(f"Output: {output}")
    print(f"Samples per night: {samples_per_night}")
    print(f"K: {k}")
    print(f"Sampling ratio: {sampling_ratio}")
    print(f"Cut begin: {cut_begin}")
    print(f"Cut end: {cut_end}")
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
    night = Night.from_directory(dataset, cut_begin, cut_end)
    
    # interpolate the night observations
    night.interpolate()
    
    # cutoff wavelength ranges eventually
    night.cutoff()

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
        
    """
    # save generated nights using np.savedz_compressed
    
    pbar = tqdm(total=len(generated_nights), desc="Saving generated nights...")
    
    for idx, generated_night in enumerate(generated_nights):
        generated_night.save(output, date, idx)
        
        pbar.update(1)
    pbar.close()
    """    
    

if __name__ == "__main__":
    main()