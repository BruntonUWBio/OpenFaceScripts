
import argparse
import pandas as pd
import os
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

def pandasToDask(pdir):
    hdf_lock = os.path.join(pdir, 'hdfs', 'au_0.hdf')

    try:
        if os.path.exists(hdf_lock):
            df = pd.read_hdf(hdf_lock, '/data')
            df = df.rename(index=str, columns={'day': 'session', 'session': 'vid'})

            new_colnames = []

            for name in df.columns:
                new_colnames.append(name.lstrip(''))
            df.columns = new_colnames
            df.to_hdf(os.path.join(pdir, 'hdfs', 'au.hdf'), key='/data', format='table')
            os.remove(hdf_lock)
    except:
        print(pdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdir")
    args = parser.parse_args()
    pdir = args.pdir

    subdirs = [os.path.join(pdir, x) for x in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, x))]

    with tqdm(total=len(subdirs)) as pbar:
        for i in Pool().imap(pandasToDask, subdirs):
            pbar.update(1)
