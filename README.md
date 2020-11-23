# FD-ProtoNet: Prototypical Network Based on Local Fisher Discriminant Analysis for Few-shot Learning

## Dependencies
* This code is tested on Ubuntu 18.04.3 with Python 3.6.9 and chainer 6.5.0

## Data
### miniImageNet
#Download and unzip "mini-imagenet.tar.gz" from Google Drive link [[mini-ImageNet](https://drive.google.com/file/d/1DvYd7LMa0zvlqTM8oBdCWwQSxpZdf_D5/view?usp=sharing)]
 
#Place ``train.npz``, ``val.npz``, ``test.npz`` files in ``FD_ProtoNet/miniImageNet_LFD_ProtoNet/data``


### tieredImageNet
#Download and unzip "tiered-imagenet.tar.gz" from Google Drive link [[tiered-ImageNet](https://drive.google.com/file/d/1zz7bAYus7EeoMokwUQlLc3OY_eoII8B7/view?usp=sharing)]

#Place images ``.npz`` and labels ``.pkl`` files in ``LFD_ProtoNet/tieredImageNet_LFD_ProtoNet/data``

## Running the code

```
#For miniImageNet experiment

cd  /FD_ProtoNet/miniImageNet/scripts
python train.py --gpu {GPU device number}
                                    --n_shot {n_shot}
                                    --nb_class_train {number of classes in training}
                                    --nb_class_test {number of classes in test}
                                    --n_query_train {number of queries per class in training}
                                    --n_query_test {number of queries per class in test}
                                    --wd_rate {Weight decay rate}

#For tieredImageNet experiment

cd /FD_ProtoNet/tieredImageNet/scripts
python train.py --gpu {GPU device number}
                                    --n_shot {n_shot}
                                    --nb_class_train {number of classes in training}
                                    --nb_class_test {number of classes in test}
                                    --n_query_train {number of queries per class in training}
                                    --n_query_test {number of queries per class in test}
                                    --wd_rate {Weight decay rate}
```
