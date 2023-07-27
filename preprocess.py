print ('######### demo Image#########')
import h5py
f = h5py.File("demoImage.hdf5", 'r')
print (f.keys())
print (f['tstCsm'].size)
print (f['tstCsm'])
print (f['tstMask'])
print (f['tstOrg'])

#######
#print ('######### actual dataset 3GB #########')
#f = h5py.File("dataset.hdf5", 'r')
#print (f.keys())
#print ((f['tstCsm']).size)
#print (f['tstMask'])
#print (f['tstOrg'])



#### opening mmdrza files
print ('######### mmdrza old files #########')

import scipy.io


mat = scipy.io.loadmat('../radar_data_20230514/An.mat')

print (mat.keys())
print (len(mat))
print (mat['__version__'])
print (mat['__header__'])
#print (dir(mat['Hn']))
print (mat['Hn'].shape)
print (mat['Hn'][100][50])
print (len(list((mat.values()))[0]))
print (len(list((mat.values()))[1]))
print (len(list((mat.values()))[2]))
print (len(list((mat.values()))[3]))

print (mat['__version__'])

#print (dir(mat))
print (type(mat))





print ('######### mmdrza new files #########')
mat = scipy.io.loadmat('../CNN_materials/An.mat')
print (mat.keys())
print (len(mat))
print (mat['__version__'])
print (mat['__header__'])
#print (dir(mat['HNNN']))
print (mat['HNNN'].shape)
print (mat['HNNN'][0][100][50])
print (len(list((mat.values()))[0]))
print (len(list((mat.values()))[1]))
print (len(list((mat.values()))[2]))
print (len(list((mat.values()))[3]))

print (mat['__version__'])

#print (dir(mat))
print (type(mat))


print ('### mmdrza new files X ##')

xmat = scipy.io.loadmat('../CNN_materials/x.mat')
print (xmat.keys())
print (len(xmat))
print (xmat['__version__'])
print (xmat['__header__'])
#print (dir(mat['HNNN']))
print (xmat['x'].shape)
print (xmat['x'][0][100][50])
print (len(list((xmat.values()))[0]))
print (len(list((xmat.values()))[1]))
print (len(list((xmat.values()))[2]))
print (len(list((xmat.values()))[3]))

print (xmat['__version__'])

#print (dir(mat))
print (type(xmat))




print ("###########S2P##########")

import skrf as rf

ntwk = rf.Network('../radar_data_20230514/bn/+10.s2p')
s = ntwk.s
print (len(s))
print (s.size)
print (s.shape)
#print (dir(s))
print (type(s))
print (s[0])


import numpy as np

print ("############   DOT product    ############")
print (xmat['x'].shape)
print (mat['HNNN'].shape)

print (type(mat['HNNN'][0]))

## Hadamard product
print (np.sum(np.multiply((mat['HNNN'][20]),xmat['x'][149])))
print (np.multiply((mat['HNNN'][450]),xmat['x'][149]).shape)





## building noise
print ('building noise ^^^^^^^^^^^^^^^^^^^^^')
noise=np.random.randn(5,)
print (noise)
noise=noise*(0.01/np.sqrt(2.))
print (noise)



