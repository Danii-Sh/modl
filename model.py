"""
This code will create the model tailored for our specific use

@author: Dan
"""
import tensorflow.compat.v1 as tf
import numpy as np
from os.path import expanduser
home = expanduser("~")
epsilon=1e-5
TFeps=tf.constant(1e-5,dtype=tf.float32)


# function c2r contatenate complex input as new axis two two real inputs
c2r=lambda x:tf.stack([tf.real(x),tf.imag(x)],axis=-1)
#r2c takes the last dimension of real input and converts to complex
r2c=lambda x:tf.complex(x[...,0],x[...,1])

def createLayer(x, szW, trainning,lastLayer):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    """
    W=tf.get_variable('W',shape=szW,initializer=tf.keras.initializers.glorot_normal())
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    xbn=tf.layers.batch_normalization(x,training=trainning,fused=True,name='BN')
    if not(lastLayer):
        return tf.nn.relu(xbn)
    else:
        return xbn

def dw(inp,trainning,nLay):
    """
    This is the Dw block as defined in the Fig. 1 of the MoDL paper
    It creates an n-layer (nLay) residual learning CNN.
    Convolution filters are of size 3x3 and 64 such filters are there.
    nw: It is the learned noise
    dw: it is the output of residual learning after adding the input back.
    """
    lastLayer=False
    nw={}
    nw['c'+str(0)]=inp
    print (inp.shape)
    szW={}
    szW = {key: (3,3,64,64) for key in range(2,nLay)}
    szW[1]=(3,3,2,64)
    szW[nLay]=(3,3,64,2)
    
    for i in np.arange(1,nLay+1):
        if i==nLay:
            lastLayer=True
        with tf.variable_scope('Layer'+str(i)):
            nw['c'+str(i)]=createLayer(nw['c'+str(i-1)],szW[i],trainning,lastLayer)
    print (nw['c0'].shape)
    with tf.name_scope('Residual'):
        shortcut=tf.identity(inp)
        dw=shortcut+nw['c'+str(nLay)]
    print ('!!!', dw.shape)    
    return dw


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, A,lam):
        with tf.name_scope('Ainit'):

            self.A = A
            self.lam=lam

    def myAtA(self,img):
        with tf.name_scope('AtA'):
            #print (self.A.dtype)
            #print (img.dtype)
            #A = tf.cast(self.A, tf.complex64)
            #print (self.A.shape,img.shape)
            
            AFlattened = tf.reshape(self.A, [451, 7381])
            imgFlattened = tf.reshape(img, [1, 7381])
            print ('inside myATA, A img', self.A.shape,img.shape)
            print ('inside myATA, both flat', AFlattened.shape,imgFlattened.shape)
            
            Arhs = tf.matmul(AFlattened, imgFlattened ,transpose_b=True)
            print ('inside myATA, AF Arhs', AFlattened.shape,Arhs.shape)
            #Arhs=self.A*img
            AtArhs=  tf.matmul (AFlattened, Arhs, transpose_a=True) 
            print (AtArhs.shape,imgFlattened.shape,self.lam.shape)
            print (self.lam)
            out =tf.transpose(AtArhs)+self.lam*imgFlattened
            print ('myATA finished', out.shape)
        return out

def myCG(A,rhs):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    rhs=r2c(rhs)
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-10)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap=A.myAtA(p)
            Ap = tf.reshape(Ap, [121, 61])
            alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p)*Ap))
            alpha=tf.complex(alpha,0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.to_float( tf.reduce_sum(tf.conj(r)*r))
            beta = rTrNew / rTr
            beta=tf.complex(beta,0.)
            p = r + beta * p
        return i+1,rTrNew,x,r,p
    print ('rhs shape', rhs.shape)
    x=tf.zeros_like(rhs)
    i,r,p=0,rhs,rhs
    rTr = tf.to_float( tf.reduce_sum(tf.conj(r)*r),)
    loopVar=i,rTr,x,r,p
    print ('kkkkkkkokokokokokok')
    out=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r(out)

def getLambda():
    """
    create a shared variable called lambda.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam1', dtype=tf.float32, initializer=.05)
    return lam

def callCG(rhs):
    """
    this function will call the function myCG on each image in a batch
    """
    G=tf.get_default_graph()
    getnext=G.get_operation_by_name('getNext')
    ## next line might be wrong
    print('irad??? ')
    _,_,A=getnext.outputs
    l=getLambda()
    l2=tf.complex(l,0.)
    def fn(tmp):
        A,r=tmp
        Aobj=Aclass(A,l2)
        y=myCG(Aobj,r)
        return y
    inp=(A ,rhs)
    rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn2' )
    return rec

@tf.custom_gradient
def dcManualGradient(x):
    """
    This function impose data consistency constraint. Rather than relying on
    TensorFlow to calculate the gradient for the conjuagte gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.
    """
    y=callCG(x)
    def grad(inp):
        out=callCG(inp)
        return out
    return y,grad


def dc(rhs,A,lam1):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    #A = tf.cast(A, tf.complex64)
    lam2=tf.complex(lam1,0.)
    def fn( tmp ):
        A ,r=tmp
        #print ('A in',A.dtype)
        #print (r.dtype)
        Aobj=Aclass( A ,lam2 )
        #print ('debugging5555')            
        y=myCG(Aobj,r)
        return y
    #print ('A out',A.dtype)
    inp=(A,rhs)
    print ('???', A.shape, rhs.shape)
    rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn' )
    
    return rec

def makeModel(atb,A,training,nLayers,K,gradientMethod):
    """
    This is the main function that creates the model.

    """
    print ('before dc ',A.dtype)
    out={}
    out['dc0']=atb
    with tf.name_scope('myModel'):
        with tf.variable_scope('Wts',reuse=tf.AUTO_REUSE):
            for i in range(1,K+1):
                j=str(i)
                print ('***',out['dc0'].shape)                
                out['dw'+j]=dw(out['dc'+str(i-1)],training,nLayers)

                lam1=getLambda()
                rhs=atb + lam1*out['dw'+j]
                if gradientMethod=='AG':

                    out['dc'+j]=dc(rhs,A,lam1)
                elif gradientMethod=='MG':
                    if training:
                        out['dc'+j]=dcManualGradient(rhs)
                    else:
                        out['dc'+j]=dc(rhs,lam1)
    return out
