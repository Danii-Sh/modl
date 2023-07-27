"""
Created on  2023

This file contains some supporting functions used during training and testing.

@author:Dan
"""
import time
import numpy as np
import h5py as h5
import scipy.io



#%%
def div0( a, b ):
    """ This function handles division by zero """
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    #print('olololo')
    return c


#%% This provide functionality similar to matlab's tic() and toc()
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

#%%
def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==3:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    print (img.dtype,'dtype')
    for i in range(nimg):
        k =div0(img[i]-img[i].min(),img[i].ptp())
        #print ('normalize0', k.dtype,img2[i].dtype)
        img2[i] = k
        #img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)
#%%


def myPSNR(org,recon):
    """ This function calculates PSNR between the original and
    the reconstructed     images"""
    mse=np.sum(np.square( np.abs(org-recon)))/org.size
    psnr=20*np.log10(org.max()/(np.sqrt(mse)+1e-10 ))
    return psnr


#%% Here I am reading the dataset for training and testing from dataset.hdf5 file
# def getA():
#     #mat_radar = scipy.io.loadmat('../radar_data_20230514/An.mat')
#     mat_radar = scipy.io.loadmat('../CNN_materials/An.mat')
#     A_radar= mat_radar['HNNN']
#     print ('A shape',A_radar.shape())
#     #A_radar = np.ones((10,40))
#     return A_radar






#%%

def getTestingData():
    print('Reading the data. Please wait...')
#    filename='demoImage.hdf5' #set the correct path here
    tic()
    mat_radar = scipy.io.loadmat('../CNN_materials/An.mat')
    A_radar = mat_radar['HNNN']
    xmat = scipy.io.loadmat('../CNN_materials/x.mat')
    org_radar = xmat['x']
    toc()
    print('Successfully read the data from file!')
    print('Now doing undersampling....')
    tic()

    
    org_radarSampleNo = 120
    
    
    org_radarSample = np.tile(org_radar[org_radarSampleNo],(1,1,1))  
    A_radar_Iterable = np.tile(A_radar,(1,1,1,1)) 
    #print ('in test', org_radarSample.shape, org_radarSample.dtype)
    
    atb_radar = radar_forward (A_radar,org_radarSample)
    atb_radar=c2r(atb_radar)
    toc()
    print('Successfully undersampled data!')
    return org_radarSample,atb_radar,A_radar_Iterable

#%%

def getData(trnTst='testing',num=100,sigma=.01):
    #num: set this value between 0 to 163. There are total testing 164 slices in testing data

 

    tic()
    print('Reading the data. Please wait...')    
    #mat_radar = scipy.io.loadmat('../radar_data_20230514/An.mat')
    mat_radar = scipy.io.loadmat('../CNN_materials/An.mat')
    #print (mat_radar.keys())
    #print (mat_radar['Hn'].shape)
    A_radar = mat_radar['HNNN']
    #A_radar = np.ones((10,40))
    
    ## deleted functions "normalize & div0" may be needed to apply
    ### creating org data : meaning x in Ax=b
    #org_radar= 0   #  it must be a 4691*1  a+bj with relevant characteristics


    xmat = scipy.io.loadmat('../CNN_materials/x.mat')
    #print (xmat['x'])
    org_radar = xmat['x']
    radar_Test_real = np.random.randn(num, 40, 1 )
    radar_Test_imaginary = np.random.randn(num, 40, 1 )
    #org_radar= radar_Test_real + 1j*radar_Test_imaginary
    #print (org_radar[20][4200][0])
    
    ###some org examples from MRI
    ##(-0.15748274+0.28943804j)     org[0][100][99]
    ##(-0.14326578+0.37293863j)     org[0][99][100]
    ##(-0.13069792+0.23481318j)     org[0][100][100]
    ##(-0.11545675+0.22066297j)     org[0][100][101]
    print (A_radar.shape,xmat['x'].shape[0])
    #A_radar = np.tile(A_radar,(xmat['x'].shape[0],A_radar.shape[0],A_radar.shape[1],A_radar.shape[2]))
    #A_radar = np.repeat(A_radar, [150, 451, 121,61])
    A_radar = np.tile(A_radar,(xmat['x'].shape[0],1,1,1))
    print (A_radar.shape)
    toc()
    
    
    tic()    
    print('Successfully read the data from file!')
    print('Now doing radar forward....')


    atb_radar = radar_forward(A_radar[0],org_radar)    

    toc()
    print('Successfully undersampled data!')

## ??
    if trnTst=='testing':
        atb_radar=c2r(atb_radar)

    return org_radar,atb_radar,A_radar



#%%
def radar_forward(A_radar,org_radar):
    print ("in radar_forward", org_radar.shape, A_radar.shape)
    radar_atb=np.empty(org_radar.shape,dtype=np.complex64)
    nImages,_,_ = org_radar.shape
    for i in range(nImages):
        ## construct relevant noise
        #noise = np.random.randn(10,1)
        AFlat = np.reshape(A_radar, [451, 7381])
        OrgFlat = np.reshape(org_radar[i], [7381, 1])
        Ax = np.matmul(AFlat, OrgFlat)
        noise = np.random.random_sample(Ax.shape)
        b = Ax+noise*(0.01/np.sqrt(2.))
        AT = np.transpose(AFlat)
        #print (AT.shape,b.shape)
        
        aTbFlat = np.matmul(AT, b)
        radar_atb[i] = np.reshape(aTbFlat, [121, 61])
    print ('ATb shape ', radar_atb.shape)    
    return(radar_atb)





#%%
def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    out=np.zeros( inp.shape[0:2],dtype=dtype)
    out=inp[...,0]+1j*inp[...,1]
    return out

def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros( inp.shape+(2,),dtype=dtype)
    out[...,0]=inp.real
    out[...,1]=inp.imag
    return out

#%%
def getWeights(wtsDir,chkPointNum='last'):
    """
    Input:
        wtsDir: Full path of directory containing modelTst.meta
        nLay: no. of convolution+BN+ReLu blocks in the model
    output:
        wt: numpy dictionary containing the weights. The keys names ae full
        names of corersponding tensors in the model.
    """
    tf.reset_default_graph()
    if chkPointNum=='last':
        loadChkPoint=tf.train.latest_checkpoint(wtsDir)
    else:
        loadChkPoint=wtsDir+'/model'+chkPointNum
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as s1:
        saver = tf.train.import_meta_graph(wtsDir + '/modelTst.meta')
        saver.restore(s1, loadChkPoint)
        keys=[n.name+':0' for n in tf.get_default_graph().as_graph_def().node if "Variable" in n.op]
        var=tf.global_variables()

        wt={}
        for key in keys:
            va=[v for v in var if v.name==key][0]
            wt[key]=s1.run(va)

    tf.reset_default_graph()
    return wt

def assignWts(sess1,nLay,wts):
    """
    Input:
        sess1: it is the current session in which to restore weights
        nLay: no. of convolution+BN+ReLu blocks in the model
        wts: numpy dictionary containing the weights
    """

    var=tf.global_variables()
    #check lam and beta; these for for alternate strategy scalars

    #check lamda 1
    tfV=[v for v in var if 'lam1' in v.name and 'Adam' not in v.name]
    npV=[v for v in wts.keys() if 'lam1' in v]
    if len(tfV)!=0 and len(npV)!=0:
        sess1.run(tfV[0].assign(wts[npV[0]] ))
    #check lamda 2
    tfV=[v for v in var if 'lam2' in v.name and 'Adam' not in v.name]
    npV=[v for v in wts.keys() if 'lam2' in v]
    if len(tfV)!=0 and len(npV)!=0:  #in single channel there is no lam2 so length is zero
        sess1.run(tfV[0].assign(wts[npV[0]] ))

    # assign W,b,beta gamma ,mean,variance
    #for each layer at a time
    for i in np.arange(1,nLay+1):
        tfV=[v for v in var if 'conv'+str(i) +str('/') in v.name \
             or 'Layer'+str(i)+str('/') in v.name and 'Adam' not in v.name]
        npV=[v for v in wts.keys() if  ('Layer'+str(i))+str('/') in v or'conv'+str(i)+str('/') in v]
        tfv2=[v for v in tfV if 'W:0' in v.name]
        npv2=[v for v in npV if 'W:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'b:0' in v.name]
        npv2=[v for v in npV if 'b:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'beta:0' in v.name]
        npv2=[v for v in npV if 'beta:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'gamma:0' in v.name]
        npv2=[v for v in npV if 'gamma:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'moving_mean:0' in v.name]
        npv2=[v for v in npV if 'moving_mean:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'moving_variance:0' in v.name]
        npv2=[v for v in npV if 'moving_variance:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
    return sess1
