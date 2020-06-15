import cupy as cp
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
import matplotlib.pyplot as plt


class LFD_ProtoNet(object):
    def __init__(self, nb_class_train, nb_class_test,  input_size, dimension, 
                 n_shot, gpu=-1):
        """
        Args
            nb_class_train (int): number of classes in a training episode
            nb_class_test (int): number of classes in a test episode
            input_size (int): dimension of input vector
            dimension (int) : dimension of embedding space
            n_shot (int) : number of shots
        """
        self.nb_class_train = nb_class_train
        self.nb_class_test = nb_class_test
        self.input_size = input_size
        self.dimension = dimension
        self.n_shot = n_shot
        # create chain
        self.chain = self._create_chain()
        self.set_gpu(gpu)


    # Set up methods
    # ---------------
    @property
    def xp(self):
        if self.gpu<0:
            return np
        else:
            return cp

    def set_gpu(self, gpu):
        self.gpu = gpu
        if self.gpu < 0:
            self.chain.to_cpu()
        else:
            self.chain.to_gpu()


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.setup(self.chain)
        self.optimizer.use_cleargrads(use=False)



    def _create_chain(self):
        chain = chainer.Chain(
            l_conv1_1=L.Convolution2D(None,64,(3,3), pad=1),
            l_norm1_1=L.BatchNormalization(64),
            l_conv1_2=L.Convolution2D(64,64,(3,3), pad=1),
  
            l_norm1_2=L.BatchNormalization(64),
            l_conv1_3=L.Convolution2D(64,64,(3,3), pad=1),
            l_norm1_3=L.BatchNormalization(64),
            l_conv1_r=L.Convolution2D(None,64,(3,3), pad=1),
            l_norm1_r=L.BatchNormalization(64),
            
            l_conv2_1=L.Convolution2D(64,128,(3,3), pad=1),
            l_norm2_1=L.BatchNormalization(128),
            l_conv2_2=L.Convolution2D(128,128,(3,3), pad=1),
            l_norm2_2=L.BatchNormalization(128),
            l_conv2_3=L.Convolution2D(128,128,(3,3), pad=1),

            l_norm2_3=L.BatchNormalization(128),  
            l_conv2_r=L.Convolution2D(64,128,(3,3), pad=1),
            l_norm2_r=L.BatchNormalization(128),

            l_conv3_1=L.Convolution2D(128,256,(3,3), pad=1),
            l_norm3_1=L.BatchNormalization(256),
            l_conv3_2=L.Convolution2D(256,256,(3,3), pad=1),
            l_norm3_2=L.BatchNormalization(256),
            l_conv3_3=L.Convolution2D(256,256,(3,3), pad=1),
            l_norm3_3=L.BatchNormalization(256),
            l_conv3_r=L.Convolution2D(128,256,(3,3), pad=1),
            l_norm3_r=L.BatchNormalization(256),
            
            l_conv4_1=L.Convolution2D(256,128,(3,3), pad=1),
            l_norm4_1=L.BatchNormalization(128),
            l_conv4_2=L.Convolution2D(128,128,(3,3), pad=1),
            l_norm4_2=L.BatchNormalization(128),
            l_conv4_3=L.Convolution2D(128,128,(3,3), pad=1),
            l_norm4_3=L.BatchNormalization(128),
            l_conv4_r=L.Convolution2D(256,128,(3,3), pad=1),
            l_norm4_r=L.BatchNormalization(128),
            
            l_phi=L.Linear(self.dimension, self.nb_class_train),
            )
        return chain


    # Train methods
    # ---------------

    def encoder(self, x, batchsize, train=True):
        with chainer.using_config('train', train):
            x2 = F.reshape(x, (batchsize,84,84,3))
            x3 = F.transpose(x2, [0,3,1,2])
            
            c1_r=self.chain.l_conv1_r(x3)
            n1_r=self.chain.l_norm1_r(c1_r)
            
            c1_1=self.chain.l_conv1_1(x3)
            n1_1=self.chain.l_norm1_1(c1_1)
            a1_1=F.relu(n1_1)
            
            c1_2=self.chain.l_conv1_2(a1_1)
            n1_2=self.chain.l_norm1_2(c1_2)
            a1_2=F.relu(n1_2)
            
            c1_3=self.chain.l_conv1_3(a1_2)
            n1_3=self.chain.l_norm1_3(c1_3)
            
            a1_3=F.relu(n1_3+n1_r)
            
            p1=F.max_pooling_2d(a1_3,2)
            p1=F.dropout(p1,ratio=0.3)
            
            c2_r=self.chain.l_conv2_r(p1)
            n2_r=self.chain.l_norm2_r(c2_r)
            
            c2_1=self.chain.l_conv2_1(p1)
            n2_1=self.chain.l_norm2_1(c2_1)
            a2_1=F.relu(n2_1)
            
            c2_2=self.chain.l_conv2_2(a2_1)
            n2_2=self.chain.l_norm2_2(c2_2)
            a2_2=F.relu(n2_2)
            
            c2_3=self.chain.l_conv2_3(a2_2)
            n2_3=self.chain.l_norm2_3(c2_3)
            
            a2_3=F.relu(n2_3+n2_r)
            
            p2=F.max_pooling_2d(a2_3,2)
            p2=F.dropout(p2, ratio=0.2)
            c3_r=self.chain.l_conv3_r(p2)
            n3_r=self.chain.l_norm3_r(c3_r)
            
            c3_1=self.chain.l_conv3_1(p2)
            n3_1=self.chain.l_norm3_1(c3_1)
            a3_1=F.relu(n3_1)
            
            c3_2=self.chain.l_conv3_2(a3_1)
            n3_2=self.chain.l_norm3_2(c3_2)
            a3_2=F.relu(n3_2)
            
            c3_3=self.chain.l_conv3_3(a3_2)
            n3_3=self.chain.l_norm3_3(c3_3)
            
            a3_3=F.relu(n3_3+n3_r)
            
            p3=F.max_pooling_2d(a3_3,2)
            p3=F.dropout(p3,ratio=0.2)
            
            c4_r=self.chain.l_conv4_r(p3)
            n4_r=self.chain.l_norm4_r(c4_r)
            
            c4_1=self.chain.l_conv4_1(p3)
            n4_1=self.chain.l_norm4_1(c4_1)
            a4_1=F.relu(n4_1)
            
            c4_2=self.chain.l_conv4_2(a4_1)
            n4_2=self.chain.l_norm4_2(c4_2)
            a4_2=F.relu(n4_2)
            
            c4_3=self.chain.l_conv4_3(a4_2)
            n4_3=self.chain.l_norm4_3(c4_3)
            
            a4_3=F.relu(n4_3+n4_r)
            
            p4=F.max_pooling_2d(a4_3,2)
            p4=F.dropout(p4, ratio=0.2)
            
            p5=F.average_pooling_2d(p4,6)
            h_t=F.reshape(p5, (batchsize,-1))
        return  h_t

    def Fisher(self,key,label,batchsize,nb_class,convert_dim,dimension):
        label=cp.array(label)
        #label = cp.reshape(cp.array(label),[nb_class,batchsize])
        #key = F.reshape(key,[nb_class,batchsize,-1])
        if(self.n_shot==1):
            Sw=cp.identity(dimension,dtype='float32')
        else:
            Sw=self.local_cov_in_class(key.data,label,nb_class,batchsize)
        Sb=self.local_cov_bet_class(key.data,label,nb_class,batchsize)
        Sw_inv = cp.linalg.inv(Sw)
        Sw_inv_Sb = cp.matmul(Sw_inv,Sb)
        lam,v = cp.linalg.eigh(Sw_inv_Sb)
        eigen_id=cp.argsort(lam)[::-1]
        eigen_value=lam[eigen_id]
        eigen_vector=v[:,eigen_id]
        W=eigen_vector[:,:convert_dim]
        W=cp.reshape(W,[dimension,convert_dim])
        W=F.transpose(W)
        return W

    

    def local_cov_in_class(self,key,label,nb_class,batchsize):
        index = cp.arange(key.shape[0])
        xx,yy = cp.meshgrid(index,index)
        sub = key[xx] - key[yy]

        norm_sub = cp.linalg.norm(sub,axis=2)
        a = cp.exp(-norm_sub*norm_sub/100) 

        lindex = cp.arange(label.shape[0])
        lx,ly = cp.meshgrid(lindex,lindex)
        l = (label[lx]==label[ly])
        a = a*l

        Sw = cp.einsum('ij,ijk,ijl->kl',a,sub,sub,dtype='float32')*0.5*(1.0/batchsize)
        return Sw

    def local_cov_bet_class(self,key,label,nb_class,batchsize):
        index = cp.arange(key.shape[0])
        xx,yy = cp.meshgrid(index,index)
        sub = key[xx] - key[yy]

        norm_sub = cp.linalg.norm(sub,axis=2)
        a1= cp.exp(-norm_sub*norm_sub/100)

        lindex = cp.arange(label.shape[0])
        lx,ly = cp.meshgrid(lindex,lindex)
        l = (label[lx]==label[ly])
        a1 = a1*l*(1.0/(batchsize*nb_class)-1.0/batchsize)

        l2 = (label[lx]!=label[ly])
        a2 = l2*(1.0/(nb_class*batchsize))
        a=a1+a2

        Sb = cp.einsum('ij,ijk,ijl->kl',a,sub,sub,dtype='float32')*0.5
        return Sb           

    def compute_loss(self,label,key,W,batchsize,nb_class,convert_dim):
        W_batch=F.broadcast_to(W,[batchsize,convert_dim,self.dimension])
        keyW_ = F.batch_matmul(W_batch,key)
        keyW = F.reshape(keyW_,[batchsize//nb_class,nb_class,convert_dim])
        mean = F.mean(keyW,axis=0)
        mean = F.reshape(mean,[nb_class,convert_dim])
        mean = F.broadcast_to(mean,[batchsize,nb_class,convert_dim])
        keyW_ = F.reshape(keyW_,[batchsize,convert_dim])
        keyW_ = F.broadcast_to(keyW_,[nb_class,batchsize,convert_dim])
        keyW_ = F.transpose(keyW_,axes=(1,0,2))
        sub = mean - keyW_
        u = -F.sum(sub*sub,axis=2)
        t = chainer.Variable(self.xp.array(label, dtype=self.xp.int32)) 
        return F.softmax_cross_entropy(u,t)

    def compute_accuracy(self,label,key,W,batchsize,nb_class,convert_dim):
        W_batch=F.broadcast_to(W,[batchsize,convert_dim,self.dimension])
        keyW_ = F.batch_matmul(W_batch,key)
        keyW = F.reshape(keyW_,[batchsize//nb_class,nb_class,convert_dim])
        mean = F.mean(keyW,axis=0)
        mean = F.reshape(mean,[nb_class,convert_dim])
        mean = F.broadcast_to(mean,[batchsize,nb_class,convert_dim])
        keyW_ = F.reshape(keyW_,[batchsize,convert_dim])
        keyW_ = F.broadcast_to(keyW_,[nb_class,batchsize,convert_dim])
        keyW_ = F.transpose(keyW_,axes=(1,0,2))
        sub = mean - keyW_
        u = -F.sum(sub*sub,axis=2)

        t_est = self.xp.argmax(F.softmax(u).data, axis=1)      
        return (t_est == self.xp.array(label))   




        
    def train(self, images, labels):
        """
        Train a minibatch of episodes
        """
        images = self.xp.stack(images)
        batchsize = images.shape[0]
        loss = 0

        key = self.encoder(images, batchsize, train=True)
        support_set = key[:self.nb_class_train*self.n_shot,:]
        query_set = key[self.nb_class_train*self.n_shot:,:]
        average_key = F.mean(F.reshape(support_set,[self.n_shot,self.nb_class_train,-1]),axis=0)
        
        batchsize_q = len(query_set.data)

        convert_dim=self.nb_class_train*self.n_shot - 1
        batchsize_s=len(support_set.data)
        batchsize_s=self.n_shot
        W = self.Fisher(support_set,labels[:self.nb_class_train*self.n_shot],batchsize_s,self.nb_class_train,convert_dim,self.dimension)
        loss = self.compute_loss(labels[self.nb_class_train*self.n_shot:],query_set,W,batchsize_q,self.nb_class_train,convert_dim)
                
        self.chain.zerograds()
        loss.backward()
        self.optimizer.update()
        
        return loss.data

        
    def evaluate(self, images, labels):
        """
        Evaluate accuracy score
        """
        nb_class = self.nb_class_test
            
        images = self.xp.stack(images)
        batchsize = images.shape[0]
        accs = []
        
        key= self.encoder(images,batchsize, train=False)
        support_set = key[:nb_class*self.n_shot,:]
        query_set = key[nb_class*self.n_shot:,:]
        average_key = F.mean(F.reshape(support_set,[self.n_shot,nb_class,-1]),axis=0)
        batchsize_q = len(query_set.data)

        convert_dim=self.nb_class_test*self.n_shot - 1
        batchsize_s=len(support_set.data)
        batchsize_s=self.n_shot
        W = self.Fisher(support_set,labels[:self.nb_class_test*self.n_shot],batchsize_s,self.nb_class_test,convert_dim,self.dimension)

        accs_tmp = self.compute_accuracy(labels[nb_class*self.n_shot:],query_set,W,batchsize_q,self.nb_class_test,convert_dim)
        
        accs.append(accs_tmp)

        return accs
    
    def decay_learning_rate(self, decaying_parameter=0.5):
        self.optimizer.alpha=self.optimizer.alpha*decaying_parameter