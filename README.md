# MIXUP-BASED DEEP METRIC LEARNING APPROACHES FOR INCOMPLETE SUPERVISION

By [Luiz H. Buris](http://), [Fabio A. Faria](https://).

UNIFESP SJC -  Instituto de Ciência e Tecnologia

## Introduction
we propose three new approaches in the context of DML. We are particularly interested in NNGK due to its robustness and simplicity. As such, we introduce variants that take advantage of Mixup to cope with metric learning in incomplete supervision scenarios.

- https://github.com/LukeDitria/OpenGAN
- https://github.com/facebookresearch/mixup-cifar10
- https://kevinmusgrave.github.io/pytorch-metric-learning/

## Citation

If you use this method or this code in your paper, then please cite it:

```
@article{buris2022mixup,
  title={Mixup-based Deep Metric Learning Approaches for Incomplete Supervision},
  author={Buris, Luiz H and Pedronette, Daniel CG and Papa, Joao P and Almeida, Jurandy and Carneiro, Gustavo and Faria, Fabio A},
  journal={arXiv preprint arXiv:2204.13572},
  year={2022},
  url={https:https://arxiv.org/pdf/2204.13572.pdf},
}
```

## Code organization

- `train_MbDML1-NNGK_Mixup.py`: This approach is a simple combination between the loss functions of the original approaches (NNGK and Mixup to compose the final loss function of this approach.
- `train_MbDML2_MixupNNGK.py` : In this approach called $MbDML-2$, the images $x_i$ and $x_j$ from the training dataset and their respective labels $y_i$ and $y_j$ within the batch are interpolated by the Mixup method, are passed through the CNN network pre -trained and classified by the $NNGK$ classifier.
- `train_MbDML3_MixupNNGK_NNGK.py` : This approach, called $MbDML-3$, is the simple linear combination between the loss function $MbDML-2$ and the original loss function $NNGK$.
- `train_NNGK.py` :This $NNGK$ approach is the original of the article that was proposed for improvement.


## Requirements and Installation
- Python version 3.6
- A [PyTorch installation](http://pytorch.org/)
- A [Pytorch-Metric-Learning installation](https://kevinmusgrave.github.io/pytorch-metric-learning/#installation)
- pip install -r requirements.txt


## MbDML
you can now carry out "run" the python scrypt with the following command:

```sh
python3 train_MbDML1-NNGK_Mixup.py --max_epochs=200 --name "CIFAR10-MbDML1-NNGK_Mixup" --scale_mixup 2 --alpha 1 --beta 1 --data_dir datasets/CIFAR100K10/train --test datasets/CIFAR100K10/Test --save_dir results/neighbour=200 --num_classes 100 --tsne_graph False --im_ext png --gpu_id 0 --input_size 32

```

```sh
python3 train_MbDML2_MixupNNGK.py --max_epochs=200 --name "CIFAR10-MbDML2_MixupNNGK" --scale_mixup 2 --alpha 1 --alpha 0 --data_dir datasets/CIFAR100K10/train --test datasets/CIFAR100K10/Test --save_dir results/neighbour=200 --num_classes 100 --tsne_graph False --im_ext png --gpu_id 0 --input_size 32

```

```sh
python3 train_MbDML3_MixupNNGK_NNGK.py --max_epochs=200 --name "CIFAR10-MbDML3_MixupNNGK_NNGK" --scale_mixup 2 --alpha 1 --beta 1 --data_dir datasets/CIFAR100K10/train --test datasets/CIFAR100K10/Test --save_dir results/neighbour=200 --num_classes 100 --tsne_graph False --im_ext png --gpu_id 0 --input_size 32

```

## Accuracy curves

Figures show the accuracy curves of each of the proposed supervised approaches $MbDML1$, $MbDML2$ and $MbDML3$ compared to the original NNGK and Mixup approaches during each epoch of the training process and every $5 epochs in the test set, in the four image bases adopted in this experiment (CIfar10, CIfar100, MNIST and Flowers17).

CIFAR10 - Train    |  CIFAR10 - Test
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/Weekly-Learning_DML-Mixup_GAN/blob/main/figure/CIFAR100%20-%20XL10%25%20ACC%20-%20accuracy.png) |  ![](https://github.com/henriqueburis/Weekly-Learning_DML-Mixup_GAN/blob/main/figure/CIFAR100%20-%20XL10%25%20ACC%20-%20test.png) 

## Different Embeddings
Different embeddings defined by each approach: (a) samples in the feature space defined by a pre-trained CNN, (b)
same samples projected onto a Gaussian kernel, (c) samples in the feature space of the pre-trained CNN together with the new
samples created by Mixup, and (d) samples in the feature space by the combination of NNGK and Mixup. Notice that, in this
paper, there are these four possible kinds of feature spaces, therefore the three proposed approaches based on Mixup (MbDML)
are some combination of the existing feature spaces.

![N|Solid](https://github.com/henriqueburis/ICIP2022/blob/main/fig/spaces_b.png?raw=true )

As can be seen in figure the classes are consistently much better separated by the Mixup(NNGK) 

CIFAR10   |   CIFAR100
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/cifar10_tsne.gif) |  ![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/cifar100_tsne.gif) 

## Comparison
Mean accuracies (%) and standard deviation (±) over ten runs using 10% of the training set. Similar and the most accurate results are highlighted.
![N|Solid](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Mean%20accuracies.PNG?raw=true)
