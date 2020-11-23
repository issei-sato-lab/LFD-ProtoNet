import cupy as cp
import numpy as np
import chainer.cuda
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
import matplotlib.pyplot as plt

class FD_ProtoNet(object):
  def __init__(self, nb_class_train, nb_class_test, input_size, dimension, n_shot, gpu=-1):
    self.nb_class_train = nb_class_train
    self.nb_class_test = nb_class_test
    self.input_size = input_size
    self.dimension = dimension
    self.n_shot = n_shot
    self.chain = self._create_chain()
    self.set_gpu(gpu)
  
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
      l_conv1_1=L.Convolution2D(None,64,(3,3),pad=1),
      l_norm1_1=L.BatchNormalization(64),
      l_conv1_2=L.Convolution2D(64,64,(3,3),pad=1),
      l_norm1_2=L.BatchNormalization(64),
      l_conv1_3=L.Convolution2D(64,64,(3,3),pad=1),
      l_norm1_3=L.BatchNormalization(64),
      l_conv1_r=L.Convolution2D(None,64,(3,3),pad=1),
      l_norm1_r=L.BatchNormalization(64),

      l_conv2_1=L.Convolution2D(64,128,(3,3),pad=1),
      l_norm2_1=L.BatchNormalization(128),
      l_conv2_2=L.Convolution2D(128,128,(3,3),pad=1),
      l_norm2_2=L.BatchNormalization(128),
      l_conv2_3=L.Convolution2D(128,128,(3,3),pad=1),
      l_norm2_3=L.BatchNormalization(128),
      l_conv2_r=L.Convolution2D(64,128,(3,3),pad=1),
      l_norm2_r=L.BatchNormalization(128),

      l_conv3_1=L.Convolution2D(128,256,(3,3),pad=1),
      l_norm3_1=L.BatchNormalization(256),
      l_conv3_2=L.Convolution2D(256,256,(3,3),pad=1),
      l_norm3_2=L.BatchNormalization(256),
      l_conv3_3=L.Convolution2D(256,256,(3,3),pad=1),
      l_norm3_3=L.BatchNormalization(256),
      l_conv3_r=L.Convolution2D(128,256,(3,3),pad=1),
      l_norm3_r=L.BatchNormalization(256),

      l_conv4_1=L.Convolution2D(256,512,(3,3),pad=1),
      l_norm4_1=L.BatchNormalization(512),
      l_conv4_2=L.Convolution2D(512,512,(3,3),pad=1),
      l_norm4_2=L.BatchNormalization(512),
      l_conv4_3=L.Convolution2D(512,512,(3,3),pad=1),
      l_norm4_3=L.BatchNormalization(512),
      l_conv4_r=L.Convolution2D(256,512,(3,3),pad=1),
      l_norm4_r=L.BatchNormalization(512)
    )
    return chain

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
      p2=F.dropout(p2,ratio=0.2)
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

    return h_t

  def Fisher(self,key,label,batchsize,nb_class,convert_dim,dimension,affinity):
    label=cp.array(label)
    if(self.n_shot==1):
      Sw=cp.identity(dimension,dtype='float32')
    else:
      Sw=self.local_cov_in_class(key.data,label,nb_class,batchsize,affinity)
      #Sw=self.local_cov_in_class_NN(key.data,label,nb_class,batchsize,5)
    Sb=self.local_cov_bet_class(key.data,label,nb_class,batchsize,affinity)
    #Sb=self.local_cov_bet_class_NN(key.data,label,nb_class,batchsize,5)
    Sb_Sw=Sb-0.5*Sw
    if(self.n_shot==1):
      Sb_Sw=Sb
    lam,v=np.linalg.eigh(Sb_Sw)
    lam=cp.asarray(lam)
    v=cp.asarray(v)
    eigen_id=cp.argsort(lam)[::-1]
    eigen_value=lam[eigen_id]
    eigen_vector=v[:,eigen_id]
    W=eigen_vector[:,:convert_dim]
    W=cp.reshape(W,[dimension,convert_dim])
    W=W/cp.reshape(cp.linalg.norm(W,axis=0),[1,convert_dim])
    W=F.transpose(W)
    return W

  def local_cov_in_class_NN(self,key,label,nb_class,batchsize,k):
    key_broadcast=cp.broadcast_to(key,(batchsize,batchsize,key.shape[1]))
    key_broadcast_transpose=cp.transpose(cp.broadcast_to(key,(batchsize,batchsize,key.shape[1])),axes=(1,0,2))
    sub_key_broadcast=key_broadcast-key_broadcast_transpose
    norm_sub_broadcast=cp.linalg.norm(sub_key_broadcast,axis=2)
    sorted_d=cp.sort(norm_sub_broadcast,axis=0)
    kth_d=sorted_d[k]
    kth_d=kth_d.reshape([batchsize,1])
    sigma=cp.matmul(kth_d,cp.transpose(kth_d))



    batchsize_per_class=batchsize//nb_class
    index = cp.arange(key.shape[0])
    xx,yy=cp.meshgrid(index,index)
    sub=key[xx]-key[yy]
    norm_sub=cp.linalg.norm(sub,axis=2)
    a=cp.exp(-norm_sub*norm_sub/sigma)
    lindex=cp.arange(label.shape[0])
    lx,ly=cp.meshgrid(lindex,lindex)
    l=(label[lx]==label[ly])
    a=a*l
    a=a.reshape([a.shape[0],a.shape[1],1])
    a_sub=a*sub
    Sw=cp.einsum('ijk,ijl->kl',a_sub,sub,dtype='float32')*0.5*(1.0/batchsize_per_class)
    return Sw

  def local_cov_bet_class_NN(self,key,label,nb_class,batchsize,k):
    key_broadcast=cp.broadcast_to(key,(batchsize,batchsize,key.shape[1]))
    key_broadcast_transpose=cp.transpose(cp.broadcast_to(key,(batchsize,batchsize,key.shape[1])),axes=(1,0,2))
    sub_key_broadcast=key_broadcast-key_broadcast_transpose
    norm_sub_broadcast=cp.linalg.norm(sub_key_broadcast,axis=2)
    sorted_d=cp.sort(norm_sub_broadcast,axis=0)
    kth_d=sorted_d[k]
    kth_d=kth_d.reshape([batchsize,1])
    sigma=cp.matmul(kth_d,cp.transpose(kth_d))

    batchsize_per_class=batchsize//nb_class
    index=cp.arange(key.shape[0])
    xx,yy=cp.meshgrid(index,index)
    sub=key[xx]-key[yy]
    norm_sub=cp.linalg.norm(sub,axis=2)
    a1=cp.exp(-norm_sub*norm_sub/sigma)
    lindex=cp.arange(label.shape[0])
    lx,ly=cp.meshgrid(lindex,lindex)
    l=(label[lx]==label[ly])
    a1=a1*l*(1.0/(batchsize*nb_class)-1.0/batchsize_per_class)
    l2=(label[lx]!=label[ly])
    a2=l2*(1.0/batchsize)
    a=a1+a2
    a=a.reshape([a.shape[0],a.shape[1],1])
    a_sub=a*sub
    Sb=cp.einsum('ijk,ijl->kl',a_sub,sub,dtype='float32')*0.5
    return Sb


    


  def local_cov_in_class(self,key,label,nb_class,batchsize,affinity):
    batchsize_per_class=batchsize//nb_class
    index = cp.arange(key.shape[0])
    xx,yy=cp.meshgrid(index,index)
    sub=key[xx]-key[yy]
    norm_sub=cp.linalg.norm(sub,axis=2)
    a=cp.exp(-norm_sub*norm_sub*affinity)
    lindex=cp.arange(label.shape[0])
    lx,ly=cp.meshgrid(lindex,lindex)
    l=(label[lx]==label[ly])
    a=a*l
    a=a.reshape([a.shape[0],a.shape[1],1])
    a_sub=a*sub
    Sw=cp.einsum('ijk,ijl->kl',a_sub,sub,dtype='float32')*0.5*(1.0/batchsize_per_class)
    return Sw
  
  def local_cov_bet_class(self,key,label,nb_class,batchsize,affinity):
    batchsize_per_class=batchsize//nb_class
    index=cp.arange(key.shape[0])
    xx,yy=cp.meshgrid(index,index)
    sub=key[xx]-key[yy]
    norm_sub=cp.linalg.norm(sub,axis=2)
    a1=cp.exp(-norm_sub*norm_sub*affinity)
    lindex=cp.arange(label.shape[0])
    lx,ly=cp.meshgrid(lindex,lindex)
    l=(label[lx]==label[ly])
    a1=a1*l*(1.0/batchsize-1.0/batchsize_per_class)
    l2=(label[lx]!=label[ly])
    a2=l2*(1.0/batchsize)
    a=a1+a2
    a=a.reshape([a.shape[0],a.shape[1],1])
    a_sub=a*sub
    Sb=cp.einsum('ijk,ijl->kl',a_sub,sub,dtype='float32')*0.5
    return Sb

  def compute_loss(self,label,key,W,W_mean,batchsize,nb_class,convert_dim):
    W_batch=F.broadcast_to(W,[batchsize,convert_dim,self.dimension])
    keyW_=F.batch_matmul(W_batch,key)
    mean=F.reshape(W_mean,[nb_class,convert_dim])
    mean=F.broadcast_to(mean,[batchsize,nb_class,convert_dim])
    keyW_=F.reshape(keyW_,[batchsize,convert_dim])
    keyW_=F.broadcast_to(keyW_,[nb_class,batchsize,convert_dim])
    keyW_=F.transpose(keyW_,axes=(1,0,2))
    sub=mean-keyW_
    u=-F.sum(sub*sub,axis=2)
    t=chainer.Variable(self.xp.array(label,dtype=self.xp.int32))
    return F.softmax_cross_entropy(u,t)

  def compute_accuracy(self,label,key,W,W_mean,batchsize,nb_class,convert_dim):
    W_batch=F.broadcast_to(W,[batchsize,convert_dim,self.dimension])
    keyW_=F.batch_matmul(W_batch,key)
    keyW=F.reshape(keyW_,[batchsize//nb_class,nb_class,convert_dim])
    mean=F.reshape(W_mean,[nb_class,convert_dim])
    mean=F.broadcast_to(mean,[batchsize,nb_class,convert_dim])
    keyW_=F.reshape(keyW_,[batchsize,convert_dim])
    keyW_=F.broadcast_to(keyW_,[nb_class,batchsize,convert_dim])
    keyW_=F.transpose(keyW_,axes=(1,0,2))
    sub=mean-keyW_
    u=-F.sum(sub*sub,axis=2)
    t_est=self.xp.argmax(F.softmax(u).data,axis=1)
    return (t_est==self.xp.array(label))


  def train(self, images, labels):
    labels = cp.array(labels)
    images = self.xp.stack(images)
    batchsize = images.shape[0]
    loss = 0

    key = self.encoder(images, batchsize, train=True)
    support_set = key[:self.nb_class_train*self.n_shot,:]
    query_set = key[self.nb_class_train*self.n_shot:,:]

    support_label = labels[:self.nb_class_train*self.n_shot]

    average_key = F.mean(F.reshape(support_set,[self.n_shot,self.nb_class_train,-1]),axis=0)

    batchsize_q = len(query_set.data)
    batchsize_s = len(support_set.data)

    convert_dim = self.nb_class_train*self.n_shot

    W = self.Fisher(support_set,support_label,batchsize_s,self.nb_class_train,convert_dim,self.dimension,0.01)

    W_batch = F.broadcast_to(W,[self.nb_class_train,convert_dim,self.dimension])
    W_mean = F.batch_matmul(W_batch,average_key)

    loss = self.compute_loss(labels[self.nb_class_train*self.n_shot:],query_set,W,W_mean,batchsize_q,self.nb_class_train,convert_dim)

    self.chain.zerograds()
    loss.backward()
    self.optimizer.update()

    return loss.data

  def evaluate(self, images, labels):
    labels = cp.array(labels)
    nb_class = self.nb_class_test

    images = self.xp.stack(images)
    batchsize = images.shape[0]
    accs = []

    key = self.encoder(images,batchsize,train=False)
    support_set = key[:nb_class*self.n_shot,:]
    query_set = key[nb_class*self.n_shot:,:]

    support_label = labels[:nb_class*self.n_shot]

    average_key = F.mean(F.reshape(support_set,[self.n_shot,nb_class,-1]),axis=0)
    mean_key = F.reshape(average_key,[nb_class,-1])

    batchsize_q = len(query_set.data)
    batchsize_s = len(support_set.data)

    convert_dim = self.nb_class_test*self.n_shot

    W = self.Fisher(support_set,support_label,batchsize_s,nb_class,convert_dim,self.dimension,0.01)

    W_batch = F.broadcast_to(W,[nb_class,convert_dim,self.dimension])
    W_mean = F.batch_matmul(W_batch,average_key)
    accs_tmp = self.compute_accuracy(labels[nb_class*self.n_shot:],query_set,W,W_mean,batchsize_q,nb_class,convert_dim)

    accs.append(accs_tmp)
    return accs
  
  def decay_learning_rate(self, decaying_parameter=0.5):
    self.optimizer.alpha=self.optimizer.alpha*decaying_parameter

    



