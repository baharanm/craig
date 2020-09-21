# craig

ICML Paper: Data-efficient Training of Machine Learning Models


Traing on MNIST:
Change the flags in the code (line 22-23 mnist.py)
Traing on random subsets: subset, random = True, True
Traing on craig subsets: subset, random = True, False  


Traing ResNet on CIFAR10:
Traing on random subsets: python train_resnet.py -s 0.1 -w -b 512
Traing on craig subsets: python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0

Train Logistic Regression:
Traing on random subsets: python logistic.py -s 0.1 -w -b 512
Traing on craig subsets: python logistic.py --data covtype -s 0.1 -g --smtk 0
