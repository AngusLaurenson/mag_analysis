import scipy as sp
from scipy import fftpack
import sys
sys.path.append('/home/asl203/codebase/mag_analysis')
import os
import magma
from tqdm import tqdm


# first command line argument
# is the name of the folder
folder = str(sys.argv[1])
prefix = str(sys.argv[2])
# list the ovf files within the folder
ovfs = ['/'.join((folder,a)) for a in os.listdir(folder) if 
a.endswith('.ovf') if a.startswith(prefix)]
print(len(ovfs))

# create the file name of the outputs
wannier_fname = '/'.join((folder,'wannier.npy'))
disp_fname = '/'.join((folder,'disp.npy'))

# read in ovf data of first magnetisation file
a = magma.analyser()
trial = a.read_ovf(ovfs[1],target='data')

# reshape to the right ordering for 2D dataset
newshape = [trial.shape[0],trial.shape[2],trial.shape[3]]

# generate an array of zeros to fill with magnetisation data
data_shape = tuple(newshape+[len(ovfs)])
data = sp.zeros(data_shape,dtype=sp.complex64)
print(data.shape)
# go through every ovf file in the folder
# sum across the thickness direction
# fill up an array of m(x,t)
for t in tqdm(range(len(ovfs))):
    temp = a.read_ovf(ovfs[t],target='data')
    #newshape = tuple([temp.shape[1],temp.shape[0],temp.shape[2],temp.shape[3]])
    #temp = sp.reshape(temp, newshape)
    temp = sp.sum(temp, axis=0)
    data[:,:,:,t] = temp

# fourier transform from t --> f
# produces the Wannier-Zeeman ladder (hopefully)
#### ERROR: script can hang somewhere after this comment line ####
wannier  = fftpack.fft(data, axis=-1)
sp.save(wannier_fname,wannier)

# del the data array to save memory
del data

# second fft produces the dispersion
disp = fftpack.fft(wannier.squeeze(),axis=0)
sp.save(disp_fname,disp)
