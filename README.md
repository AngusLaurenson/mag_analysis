# mag_analysis
This project is to build a custom module for analysing time series data for magnetisation dynamics. The aim is to read in raw magnetisation data and calculate properties of interest such as dispersion of spin waves, profile of modes, magnetisation texture, permeability etc.

<<<<<<< HEAD
#### Issues

The read_ovf() function takes a long time to read data from the ovf files when 
compared to the I/O for the hdf5 files. While it could be that the single hdf5 
file is better than 1,000s of ovf files anyway. Not sure what is going on here 
maybe the sequential reading of ovfs is super inefficient and a single hdf5 
can be read from disk more cleverly using the hardware? I really don't know 
but it seems weird. We find that initiating hdf5 files with the flag 
libver='latest' yields a 4 fold write rate increase. However it is still way 
too slow ~5MB/s instead of ~100MB/s (a typical value for HDDs). Another way to 
increase the speed is to increase the cache size of h5py in order to more 
efficiently write and read data.

#### Wishlist

* add plotting of phase amplitude maps
* switch h5py to h5py_cache in order to increase cache to fit a whole chunk
=======
#### Wishlist

* permeability calculations
>>>>>>> master
