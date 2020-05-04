DESCRIPTION:

This Python module provides functions for handling micromagnetic simulation data 
that is too large to fit into main memory. It is built around dask.array for
parallel processing of data and hdf5 for storing data in chunked files for better
random read and write performance than contiguous data files on disk.

AUTHOR(S):

* Angus Laurenson, PhD student 2015-2019
* asl203@exeter.ac.uk # depreciation warning: no longer a student at Exeter
* 06angus06@gmail.com # personal email address if you get really stuck

USAGE:

Below is a typical example of using magma
```
import magma
import h5py
from glob import glob
import dask.array as da

# use glob to get a list of ovf files
ovfs = glob('path/to/*.ovf')

# filter the ovf files to get ones you want
ovfs = [x for x in ovfs if x.startswith('m')]

# read ovf files into a .hdf5 files, removing .ovf files as we go
magma.ovf_to_hdf('data.hdf5',ovfs, delete_ovfs=True)

# Perform fft along the time dimension
magma.fft_dask('data.hdf5','mag','xf_data.hdf5','xf_data',-1)

# To read from .hdf5 file
with h5py.File('data.hdf5','r') as f:
    arr = da.from_array(f['mag'])
    
    # dask.array objects are lazy, so the statement is not evaluated immediately
    # you can select a subset of data or setup an agregation
    
    # subsampling, load to main memory
    m_lowres = arr[::10,::10,::10].load()
    
    # get sum over time of the data
    m_sum_time = arr.sum(axis=-1)
    
    # You can write dask.arrays to disk like
    with hd.File(hdf_name,'r+',libver="latest") as f:
        f[dataset_name]= m_lowres
   
    # Applying numpy function directly converts the lazy dask array 
    # into an eager numpy array in memory
    arr = np.array(arr).compute()
    
    
```