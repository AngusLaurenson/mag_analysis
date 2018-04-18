# Analysis module
import scipy as sp
import h5py as hd
import pandas as pd
import matplotlib.pyplot as plt
import re
import dask.array as da
from dask.diagnostics import ProgressBar

"""Analyser object is used to analyse the contents of a folder. The intended use case is to analyse time series vector fields from ovf file format, within a jupyter notebook's python environment.

    The idea is to first store the data from the simulation in a hdf5 file with compression.
     ___________
    | OVF files | read ovf files           HDF5 File
    |___________| ------------->       ________|_________
    |___________|                     |                  |
                                   Vector              Meta
                                   fields              data
                            ________|_______          ___|_____
                           |                |        |         |
                        m[x,y,z,c,t]  m[x,y,z,c,w]  [times]  {params}

    The module itself focused around an analyser object

                                      Analyser
                     ____________________|____________________
                    |                                         |
                attributes                                 methods
          __________|____________             ________________|_________
         |           |           |           |          |               |
      list of    number of      time        read       hdf5          Fourier
     ovf files   dimensions    stamps    ovf file    interfaces    Analysis

    The idea is that for one simulation, the simulation is done by one analyser object. To do this all relevant ovf files are passed to an instance. On setup the time stamps and other meta data are collected from the ovf files. Then the user can use various functions to make fourier analysis of the data sets. Ultimately I want to add the visualisation of mode profile and calculation of dispersion relations."""

# Analyser object to contain the analysis in one place
# and avoid cross contamination of analysis of seperate
# datasets in the same jupyter notebook


# setup function takes the list of ovf files for the simulation
# and hdf5 file name. It looks for existing hdf file
# def __init__(self, previous_self = None):
#     super(analyser, self).__init__()
#     pass
# if previous self is defined load that
# self.load(previous_self)

# otherwise just do nothing

# function to load a pickle file of a previous
# version of the analyser object. Useful for
# revisiting analysis
# def load_state(self, previous_self):

# save the state of the analyser object in a
# pickle file
# def save_state(self, fname):
class analyser():

    ''' ####################### METHODS ######################### '''
    # function that reads ovf files and returns your choice of
    # meta data in a dictionary & the raw header in a list
    # and / or the magnetisation data in an array
    def read_ovf(self, fname, target = 'all'):
        # open the file
        with open(fname,'rb') as f:
            # initialise lists for the key value pairs
            keys = []; values = [];
            raw_header = []
            # headers are 28 lines long
            for n in range(28):
                line = f.readline().decode('utf-8')
                raw_header.append(line)
                try:
                    values.append(float(re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', line)[0]))
                    keys.append(re.findall(r'[a-zA-Z]{2,}', line)[-1])
                except:
                    pass

            # create a dictionary holding the meta data
            meta = dict(zip(keys, values))

            # return meta data and raw header if data is not required
            if (target == 'meta'):
                return meta, raw_header

            # check format of numerical data
            check_byte = sp.fromstring(f.read(4), dtype=sp.float32)
            if check_byte != 1234567.0 :
                print('check byte failed!')
                return 1

            # infer the data shape and read in the magnetisation data
            # reshape the data into array of the simulation
            data_shape = sp.array([meta['xnodes'],meta['ynodes'],meta['znodes'],3], dtype = int)
            data = sp.fromstring(f.read(sp.prod(data_shape)*int(meta['Binary'])), dtype=sp.float32)
            data = data.reshape(data_shape, order='c')

            # output according to target type
            # return 1 and print error if target is invalid
            if (target == 'all'):
                return data, meta, raw_header
            elif (target == 'data'):
                return data
            else:
                print('incorrect mode type; \n data, meta and all \n are the only valid modes, \n default is all')
                return 1

    # if the hdf file name is taken, don't overwrite
    # put the ovf files into a list
    # save function to save the state of the analyser object
    # calibration function takes
    def prelims(self, hdfname):
        # get the important parameters of the simulation from
        # the meta data from the ovf files and stuff
        self.times = []
        for n in self.ovfs:
            meta = self.read_ovf(n, 'meta')
            self.times.append(float(list(meta.keys())[15]))

        # put meta data into a group for later
        with hd.File(hdfname, 'a') as f:
            # meta_data = f.create_group('meta')
            times = f['meta'].create_dataset('times', data = self.times)
            parameters = f['meta'].create_dataset('params', data = meta)

    # save the magnetisation data from a list of ovf files
    # into a hdf5 file, along with meta data etc
    def ovf_to_hdf(self, hdf_name, ovf_files = []):

        # calculate the size and shape of the data we are dealing with
        meta, header = self.read_ovf(ovf_files[0], target='meta')
        header_encoded = [n.encode("ascii","ignore") for n in header]

        data_shape = sp.array([meta['xnodes'],meta['ynodes'],meta['znodes'],3,len(ovf_files)])
        data_size = sp.prod(data_shape * int(meta['Binary']))
        time = []
        # create the hdf file
        with hd.File(hdf_name,'w') as f:
            dset = f.create_dataset('mag', data_shape, dtype = sp.dtype('f'+str(int(meta['Binary']))), chunks=True)

            # want to add the meta data in but cannot seem to get it to work
            # There is a problem with data format supported by hdf5

        # go through the ovf files and populate the hdf file with the data
        print('looping through ovf files')
        for n in range(len(ovf_files)):
            print('reading ovf data')
            data, meta, raw = self.read_ovf(ovf_files[n])
            time.append(meta['time'])
            with hd.File(hdf_name,'a') as f:
                print('writing to file')
                f['mag'][:,:,:,:,n] = data

        with hd.File(hdf_name,'a') as f:
            f.create_dataset('time', data = sp.array(time))
            f.create_dataset('header', (len(header_encoded),1),'S10', header_encoded)

        return 0

    # import dask.array as da
    # from dask.diagnostics import ProgressBar
    # perform a FFT along a given access using DASK module
    # which provide lazy evaluation for out of core work
    # useful for large data sets that exceed RAM capacity
    def fft_dask(self, fname, srcdset, destdset, axis):
        # open the hdf5 file
        with hd.File(fname, 'a') as f:

            # get the dimensions of the problem
            dshape = f[srcdset].shape; cshape = f[srcdset].chunks

            # create a destination dataset
            try:
                f.create_dataset(destdset, dshape, chunks=cshape, dtype=complex)
            except:
                pass

            # make a dask array from the dset
            data = da.from_array(f[srcdset], f[srcdset].chunks)

            # weld chunks together to span the fft axis
            newcshape = sp.array(cshape)
            newcshape[axis] = dshape[axis]
            newcshape = tuple(newcshape)

            # rechunk dask array in order to perform fft
            data = da.rechunk(data, newcshape)

            with ProgressBar():
                out = data.compute()

            # fft and write to destination dataset on disk
            fft_data = da.fft.fft(data, axis=axis)
            with ProgressBar():
                fft_data.to_hdf5(fname,destdset, chunks=cshape, compression='lzf')
        return 0

    # perform FFT along given axis, done out of core
    # multiple chunks that span the axis are read into RAM
    # data is processed and then written to disk iteratively
    def fft_no_dask(self, fname, srcdset, destdset, axis):
        # open the hdf5 file
        with hd.File(fname, 'a') as f:

            # get the dimensions of the problem
            dshape = f[srcdset].shape; cshape = f[srcdset].chunks

            # create a new data set for the result to be stored
            try:
                f.create_dataset(destdset, dshape, chunks=cshape, dtype=complex)
            except:
                pass

            # reshape dask array in order to perform fft
            # weld together existing chunks to span the desired axis
            newcshape = sp.array(cshape)
            newcshape[axis] = dshape[axis]
            newcshape = tuple(newcshape)

            # logic to run through each chunk column
            chunkarr = tuple([int(a/b) for (a,b) in zip(dshape, newcshape)])
            for x in sp.ndindex(chunkarr):
                # get subset of array
                index = tuple([slice(int(a*b), int((a+1)*b)) for (a,b) in zip(x,newcshape)])
                chunk_data = f[srcdset][index]
                # perform fft
                fft_chunk = sp.fft(chunk_data, axis = axis)
                # write to disk
                f[destdset][index] = fft_chunk
        return 0
