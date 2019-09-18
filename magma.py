# Analysis module
import scipy as sp
from scipy import fftpack
import scipy.signal as signal
import h5py as hd
import pandas as pd
import matplotlib.pyplot as plt
import re
import dask.array as da
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import os
import colorcet

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
            data_shape = sp.array([meta['znodes'],meta['ynodes'],meta['xnodes'],meta['valuedim']], dtype = int)
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

    ######## INPUT METHODS #######
    
    # this function is slow because concatentation requires finding
    # a contiguous piece of physical memory, better to allocate in advance
    def ovf_to_array(self, ovfs=[]):
        data = []
        for ovf in tqdm(ovfs):
            data.append(self.read_ovf(ovf, target='data'))
        sp.concatenate(data,axis=-1)
        return data

    def ovf_to_array_new(self, ovfs=[]):
        meta, header = self.read_ovf(ovfs[0], target='meta')
        for n in ['xnodes','valuedim']:
            print(meta[n])

        data_shape = sp.array([meta['znodes'],meta['ynodes'],meta['xnodes'],meta['valuedim'],len(ovfs)],dtype=int)
        print(data_shape)
        data = sp.empty(tuple(data_shape),dtype=sp.float32)
        for t in tqdm(range(len(ovfs))):
            data[:,:,:,:,t] = self.read_ovf(ovfs[t],target='data')
        return(data)
 
    def ovf_to_npy(self, dst, ovfs=[]):
        data = []
        for ovf in tqdm(ovfs):
            data.append(self.read_ovf(ovf, target='data'))
        sp.concatenate(data,axis=-1)
        sp.save(dst,data)
        return 0
    
    def ovf_to_hdf(self, hdf_name, ovf_files = []):

        # calculate the size and shape of the data we are dealing with
        meta, header = self.read_ovf(ovf_files[0], target='meta')
        header_encoded = [n.encode("ascii","ignore") for n in header]

        data_shape = sp.array([meta['znodes'],meta['ynodes'],meta['xnodes'],meta['valuedim'],len(ovf_files)])
        data_size = sp.prod(data_shape * int(meta['Binary']))
        time = []
        
        # create the hdf file if it doesnt exist
        #if (os.path.isfile(hdf_name) == True):
        #   print('file exists')
        #    return 1
        with hd.File(hdf_name,'w',libver='latest') as f:
            dset = f.create_dataset('mag', data_shape, dtype = sp.dtype('f'+str(int(meta['Binary']))), chunks=True)

            # want to add the meta data in but cannot seem to get it to work
            # There is a problem with data format supported by hdf5

        # go through the ovf files and populate the hdf file with the data
            chunk_time_length = dset.chunks[-1]
            data_shape = dset.shape
            chunk_number = data_shape[-1] / chunk_time_length
            chunk_shape = dset.chunks

        # Close the hdf5 file at every opportunity
        # According to docs: https://support.hdfgroup.org/HDF5/faq/perfissues.html
        # a memory leak can occur when writing to the same file many times in a loop
            print("creating dataset")
            print("shape ", dset.shape)
            print("chunks ", dset.chunks)
            # prepare an array of all the data in one time chunk
            for c in tqdm(range(int(sp.ceil(chunk_number)))):
                temp_arr = sp.zeros((data_shape[0],data_shape[1],data_shape[2],data_shape[3],chunk_shape[-1]))

                # fill the temp array with data from ovf files
                try:
                    for n in range(chunk_time_length):
                        temp_arr[:,:,:,:,n], meta, raw = self.read_ovf(ovf_files[chunk_time_length*c + n])
                        time.append(meta['time'])
                except:
                    # This catches the unexpected case where the chunk length in time
                    # does not perfectly divide the length of ovf list
                    
                    temp = list(map(self.read_ovf,ovf_files[chunk_time_length*c:]))
                    temp_arr = sp.stack([a[0] for a in temp], axis=-1)
                    
                    
                # open hdf5 file, write the time chunk to disk, close hdf5 file
                with hd.File(hdf_name,'r+',libver="latest") as f:
                    f['mag'][:,:,:,:,chunk_time_length*c:chunk_time_length*(c+1)] = temp_arr

        # Append to the hdf5 file additional meta data
        with hd.File(hdf_name,'a',libver='latest') as f:
            f.create_dataset('time', data = sp.array(time))
            f.create_dataset('header', (len(header_encoded),1),'S30', header_encoded)
            try:
                f.create_dataset('meta', meta.data)
            except:
                print('failed to save metadata to disk')
        
        # change permissions data.hdf5 file to be read only
        os.chmod(hdf_name, 444) 
        
        # then should os.remove() all ovf files to save space on disk
        #for ovf in ovf_files:
        #    os.remove(ovf)
        return 0

    ########## FOURIER ANALYSIS METHODS ###########


    def fft_dask(self, src_fname, src_dset, dst_fname, dst_dset, axis, background_subtraction = True, window = False):
        '''
        # import dask.array as da
        # from dask.diagnostics import ProgressBar
        # perform a FFT along a given access using DASK module
        # which provide lazy evaluation for out of core work
        # useful for large data sets that exceed RAM capacity
        '''
        if (src_fname == dst_fname):
            print('must write to new .hdf5 file')
            return 1
        
        # open the hdf5 files        
        with hd.File(src_fname, 'r', libver='latest') as s:
            with hd.File(dst_fname, 'w', libver='latest') as d:

                # create a destination dataset            
                dshape = s[src_dset].shape; cshape = s[src_dset].chunks
                d.create_dataset(dst_dset, dshape, chunks=cshape, dtype=complex)

            # make a dask array from the dset
            data = da.from_array(s[src_dset], s[src_dset].chunks)
            
            # weld chunks together to span the fft axis
            newcshape = sp.array(cshape)
            newcshape[axis] = dshape[axis]
            newcshape = tuple(newcshape)

            # rechunk dask array in order to perform fft
            data = da.rechunk(data, newcshape)
            
            # make optional background subtraction
            if (background_subtraction == True):
                background = data[:,:,:,:,0]
                data = data - background[:,:,:,:,None]

            # make optional windowing before fourier transform
            if (window != False):
                try:
                    w = eval('signal.'+window+'(data.shape[axis])')
                    dim_arr = sp.ones((1,w.ndim),int).ravel()
                    dim_arr[axis] = -1
                    window_reshaped = w.reshape(dim_arr)
                    data = data * window_reshaped
                except:
                    print('invalid window function, skipping windowing.\nLook up scipy.signal docs')
                    pass
                
            # fft and write to destination dataset on disk
            fft_data = da.fft.fft(data, axis=axis)
            with ProgressBar():
                fft_data.to_hdf5(dst_fname,dst_dset)#, chunks=cshape, dtype=complex, compression='lzf')
        return 0

    # perform FFT along given axis, done out of core
    # multiple chunks that span the axis are read into RAM
    # data is processed and then written to disk iteratively
    def fft_no_dask(self, fname, srcdset, destdset, axis):
        # open the hdf5 file
        with hd.File(fname, 'a', libver='latest') as f:

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

    def calc_dispersion(self, src,dst,axis=2, window=False, save_frequencies=False):
        # t to f
        self.fft_dask(src,'mag','f_mag.hdf5','fft_1',-1,window)
        # x to k
        self.fft_dask('f_mag.hdf5','fft_1','temp2.hdf5','fft_2',axis,window)

        with hd.File('temp2.hdf5','r', libver='latest') as temp:
            disp_arr = da.from_array(temp['fft_2'],chunks=temp['fft_2'].chunks)
            dispersion = da.sum(sp.absolute(disp_arr),
                    axis = tuple([a for a in range(5) if a not in (axis,4)])
                   ) 
            with hd.File(dst,'w', libver='latest') as d:
                pass
            
            dispersion.to_hdf5(dst,'disp')
        

        # delete the intermediary values from longterm memory
        if save_frequencies:
            os.remove('temp2.hdf5')
        else:
            os.remove('temp1.hdf5')
            os.remove('temp2.hdf5')

        # it is possible that this is the wrong approach as Dask Array might have
        # better handling of out of core processes. I can see the name temp.hdf5
        # might clash between simulations if they intrude on one anothers filespace
        return 0
    
    def calc_dispersion_npy(self, src, dst, axis=1):
        data = sp.load(src)
        background = data[0,:,:,:,:]
        data = data - background[None,:,:,:,:]
        disp = sp.sum(
            sp.absolute(
                fftpack.fftshift(
                    fftpack.fft2(data, axes=(0,axis)),
                    axes=(0,axis)
                )
            ),
            axis = tuple([a for a in range(5) if a not in (axis,0)])
        )
        sp.save(dst, disp)
        return 0
    
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
        with hd.File(hdfname, 'a', libver='latest') as f:
            # meta_data = f.create_group('meta')
            times = f['meta'].create_dataset('times', data = self.times)
            parameters = f['meta'].create_dataset('params', data = meta)

    # save the magnetisation data from a list of ovf files
    # into a hdf5 file, along with meta data etc


    def wannier_zeeman(self, data, taxis=0, xaxis=1, zaxis=2):
        temp  = sp.sum(data, axis=zaxis)
        temp  = fftpack.fft(temp, axis=taxis)
        sp.save("freq_x.npy",temp)
        temp = fftpack.fft(temp,axis=xaxis)
        sp.save("disp.npy",temp)
        return 0

    
def phase_amp_filter(data):
    '''Get cyclic color map for phase scaled in intensity by amplitude. Requires complex input'''
    norm = plt.Normalize()
    phase_colors = colorcet.cm['cyclic_mygbm_30_95_c78_s25'](norm(sp.angle(data)))

    # Map the real data to a greyscale colormap
    # print(sp.absolute(raw_data.real).max)
    norm = plt.Normalize()
    amplitude_colors = colorcet.cm['linear_grey_10_95_c0'](norm(sp.absolute(data)))
    
    phase_amp = phase_colors*amplitude_colors
    
    if phase_amp.dtype != sp.uint8:
            phase_amp = (255*phase_amp).astype(sp.uint8)

    return phase_amp