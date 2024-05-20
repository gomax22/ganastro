import os
import argparse
from pathlib import Path
from classes.night import Night


def check_args(args):
    # check if dataset directory exists
    if not Path(args["dataset"]).exists():
        raise ValueError(f"Dataset directory {args['dataset']} does not exist.")
    
    # check if samples per night is positive
    if args["samples_per_night"] <= 0:
        raise ValueError("Samples per night must be a positive integer.")
    
    # check if cutoff begin is positive
    if args["cutoff_begin"] is not None and args["cutoff_begin"] <= 0:
        raise ValueError("cutoff begin must be a positive float.")
    
    # check if cutoff end is positive
    if args["cutoff_end"] is not None and args["cutoff_end"] <= 0:
        raise ValueError("cutoff end must be a positive float.")
    
    # check if cutoff end is greater than cutoff begin
    if args["cutoff_begin"] is not None and args["cutoff_end"] is not None and args["cutoff_end"] < args["cutoff_begin"]:
        raise ValueError("cutoff end must be greater than cutoff begin.")
    
    return True

def main(args):
    # get arguments
    dataset = args["dataset"]
    output = args["output"]
    samples_per_night = args["samples_per_night"]
    cutoff_begin = args["cutoff_begin"]
    cutoff_end = args["cutoff_end"]
    
    # print input parameters
    print(f"Dataset: {dataset}")
    print(f"Output: {output}")
    print(f"Samples per night: {samples_per_night}")
    print(f"cutoff begin: {cutoff_begin}")
    print(f"cutoff end: {cutoff_end}")

    # create output directory
    if not Path(output).exists():
        Path(output).mkdir(parents=True, exist_ok=True)

    # list nights stored in the data directory
    night_paths = [os.path.join(dataset, entry) for entry in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, entry)) and entry != "tars"]
    print(f"Found {len(night_paths)} nights in the dataset directory")

    for night_path in night_paths:
        print(f"\nProcessing night {night_path}")

        # get observations from night
        night = Night.from_directory(night_path, cutoff_begin, cutoff_end)
        
        # interpolate the night observations
        night.interpolate()
        
        # cutoff wavelength range
        night.cutoff()
        
        # select a limited amount of observations
        night.select(samples_per_night, region='center')

        # save the night
        date = night_path.split(os.sep)[-1]
        out_fname = os.path.join(output, f"night_{date}_cb{int(night.cutoff_begin)}_ce{int(night.cutoff_end)}.npz")
        night.save(out_fname)

        print("\n")
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess real nights")
    ap.add_argument("-d", "--dataset", required=True, help="path to data directory")
    ap.add_argument("-o", "--output", required=True, help="path to output directory")
    ap.add_argument("-s", "--samples-per-night", required=True, type=int, default=58, help="number of samples to generate per night")
    ap.add_argument("-b", "--cutoff-begin", required=False, default=None, type=float, help="cutoff beginning of wavelength range (in Angstroms)")
    ap.add_argument("-e", "--cutoff-end", required=False, default=None, type=float, help="cutoff end of wavelength range (in Angstroms)")
    
    args = vars(ap.parse_args())
    check_args(args)
    main(args)