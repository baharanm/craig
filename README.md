# craig

ICML Paper: Data-efficient Training of Machine Learning Models


### Training on MNIST:
> Change the flags in the code (line 22-23 mnist.py)
>
> Traing on random subsets: subset, random = True, True
>
> Traing on craig subsets: subset, random = True, False  


### Training ResNet on CIFAR10:
> Traing on random subsets: python train_resnet.py -s 0.1 -w -b 512
>
> Traing on craig subsets: python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0

### Training Logistic Regression:
> Traing on random subsets: python logistic.py --data covtype --method sgd -s 0.1 --greedy 0
>
> Traing on craig subsets: python logistic.py --data covtype --method sgd -s 0.1 --greedy 1
>
> You can use -b, -g to specify the learning rate, otherwise the learning rate will be tuned.


Please note that we used the greedy implementation from [summary analythics](https://smr.ai/), and the running times are reported accordingly. To use the provided python implementation, please use the flag smtk=0.
