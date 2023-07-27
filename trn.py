# -*- coding: utf-8 -*-
"""
This is the training code to train the model as described in the following article:

MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

This code solves the following optimization problem:

    argmin_x ||Ax-b||_2^2 + ||x-Dw(x)||^2_2

 'A' can be any measurement operator. Here we consider parallel imaging problem in MRI where
 the A operator consists of undersampling mask, FFT, and coil sensitivity maps.

Dw(x): it represents the residual learning CNN.

Here is the description of the parameters that you can modify below.

epochs: how many times to pass through the entire dataset

nLayer: number of layers of the convolutional neural network.
        Each layer will have filters of size 3x3. There will be 64 such filters
        Except at the first and the last layer.

gradientMethod: MG or AG. set MG for 'manual gradient' of conjuagate gradient (CG) block
                as discussed in section 3 of the above paper. Set it to AG if
                you want to rely on the tensorflow to calculate gradient of CG.

K: it represents the number of iterations of the alternating strategy as
    described in Eq. 10 in the paper.  Also please see Fig. 1 in the above paper.
    Higher value will require a lot of GPU memory. Set the maximum value to 20
    for a GPU with 16 GB memory. Higher the value more is the time required in training.

sigma: the standard deviation of Gaussian noise to be added in the k-space

batchSize: You can reduce the batch size to 1 if the model does not fit on GPU.

Output:

After running the code the output model will be saved in the subdirectory 'savedModels'.
You can give the name of the generated ouput directory in the tstDemo.py to
run the newly trained model on the test data.


@author: Hemant Kumar Aggarwal
"""

# import some librariesw
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from datetime import datetime
from tqdm import tqdm
import supportingFunctions as sf
import model as mm

tf.compat.v1.reset_default_graph()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True

#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLY
nLayers=5
epochs=50
batchSize=1
gradientMethod='AG'
K=1
sigma=0.01
restoreWeights=False
#%% to train the model with higher K values  (K>1) such as K=5 or 10,
# it is better to initialize with a pre-trained model with K=1.
if K>1:
    restoreWeights=True
    restoreFromModel='04Jun_0243pm_5L_1K_100E_AG'

if restoreWeights:
    wts=sf.getWeights('savedModels/'+restoreFromModel)
#--------------------------------------------------------------------------
#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
#print ((datetime.now().strftime("%d%b_%I%M%p_")))
directory=saveDir+datetime.now().strftime("%d%b_%I%M%p_")+ \
 str(nLayers)+'L_'+str(K)+'K_'+str(epochs)+'E_'+gradientMethod

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'


#%% save test model
tf.compat.v1.reset_default_graph()

AT = tf.compat.v1.placeholder(tf.complex64,shape=(None,451,121,61),name='A')
atbT = tf.compat.v1.placeholder(tf.float32,shape=(None,121,61,2),name='atb')



out=mm.makeModel(atbT,AT,False,nLayers,K,gradientMethod)


predTst=out['dc'+str(K)]
predTst=tf.identity(predTst,name='predTst')
sessFileNameTst=directory+'/modelTst'



saver=tf.compat.v1.train.Saver()
with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved:' +savedFile)
#%% read multi-channel dataset
# trnOrg,trnAtb=sf.getData('training')
# trnOrg,trnAtb=sf.c2r(trnOrg),sf.c2r(trnAtb)
trnOrg,trnAtb,trnA=sf.getData('training')
trnOrg,trnAtb=sf.c2r(trnOrg),sf.c2r(trnAtb)
print ('0000000000000',trnOrg.shape,trnAtb.shape,trnA.shape)
#singleA=sf.getA()

#trnA = np.tile(singleA, (100,1,1))
#trnA = np.repeat(singleA[None,:], 100, axis=0)
#trnA = np.empty([])
#for i in range (3):
#  trnA = np.concatenate((singleA, singleA), axis=0)
#print (trnA [0][200][500], trnA [5][200][500])
#m = np.ones((100, 10, 40 ))
#trnA= m + 1j*m
#print('0101011',singleA.shape,trnA.shape)

#%%
tf.compat.v1.reset_default_graph()

AP= tf.compat.v1.placeholder(tf.complex64,shape=(None,None,None,None),name='A')
atbP = tf.compat.v1.placeholder(tf.float32,shape=(None,None,None,2),name='atb')
orgP = tf.compat.v1.placeholder(tf.float32,shape=(None,None,None,2),name='org')
print ('321', AP.shape)

#%% creating the dataset
nTrn=trnOrg.shape[0]
nBatch= int(np.floor(np.float32(nTrn)/batchSize))
nSteps= nBatch*epochs
print ('debugging23232323', nTrn)

#trnData = tf.data.Dataset.from_tensor_slices((orgP,atbP,AP))
trnData = tf.data.Dataset.from_tensor_slices((orgP,atbP,AP))

print (trnData)
#trnData = tf.data.Dataset.from_tensor_slices((trnOrg,trnAtb))
trnData = trnData.cache()
trnData=trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=trnOrg.shape[0])
trnData=trnData.batch(batchSize)
trnData=trnData.prefetch(5)
# iterator=trnData.make_initializable_iterator()
iterator = tf.compat.v1.data.make_initializable_iterator(trnData)


#orgT,atbT,AT = iterator.get_next('getNext')
orgT,atbT ,AT= iterator.get_next('getNext')
#AT=AP

#print (dir(trnData))
print ('000',orgT.shape, atbT.shape,AT.shape)
#%% make training model
#AP= sf.getA()
#print ('321', AP.shape)
#print ('out of model ',AP.dtype)
#AP = tf.cast(AP, tf.complex64)
#print ('2 out of model ',AP.dtype)

out=mm.makeModel(atbT,AT, True,nLayers,K,gradientMethod)

predT=out['dc'+str(K)]
predT=tf.identity(predT,name='pred')
print ('working on loss ',predT.shape,orgT.shape)
loss = tf.reduce_mean(tf.reduce_sum(tf.pow(predT-orgT, 2),axis=0))
tf.summary.scalar('loss', loss)
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

with tf.name_scope('optimizer'):
    optimizer = tf.compat.v1.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    opToRun=optimizer.apply_gradients(capped_gvs)


#%% training code


print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %p"))
print ('parameters are: Epochs:',epochs,' BS:',batchSize,'nSteps:',nSteps,'nSamples:',nTrn)

saver = tf.compat.v1.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = tf.compat.v1.placeholder(tf.float32)
lossSumT = tf.compat.v1.summary.scalar("TrnLoss", lossT)
print ('debugging555555555555555')
with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    if restoreWeights:
        sess=sf.assignWts(sess,nLayers,wts)
    print ('888',orgP.shape,trnOrg.shape,AP.shape,trnA.shape)
    feedDict={orgP:trnOrg,atbP:trnAtb,AP:trnA}
    
    sess.run(iterator.initializer,feed_dict=feedDict)
    
    savedFile=saver.save(sess, sessFileName)
    print ('123456789')
    print("Model meta graph saved in::%s" % savedFile)

    writer = tf.compat.v1.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp,_,_=sess.run([loss,update_ops,opToRun])
            
            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                
                writer.add_summary(lossSum,ep)
                totalLoss=[] #after each epoch empty the list of total loos
        except tf.errors.OutOfRangeError:
            break
    savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
    writer.close()

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %p"))
print ('*************************************************')

#%%
