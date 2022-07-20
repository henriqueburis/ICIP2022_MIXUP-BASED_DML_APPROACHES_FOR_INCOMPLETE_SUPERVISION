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

- `train_MbDML1-NNGK_Mixup.py`: Esta abordagem chamada $NNGK+Mixup$ é uma simples combinação entre as funções de perdas das abordagens originais (NNGK e Mixup para compor a função de perda final desta abordagem. Essa  combinação de funções de perdas é utilizada durante o processo de treinamento, calculando-se a perda de ambas abordagens originais em cada lote e a retropropagação do erro é realizado na arquitetura CNN levando-se em consideração a função final combinada definida pela soma dos valores das funções de perda. 
- `train_MbDML2_MixupNNGK.py` : trai.
- `train_MbDML3_MixupNNGK_NNGK.py` : train.
- `train_NNGK.py` : nngk original.


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

## Accuracy Test .....corrigindo.

CIFAR100 - Train    |  CIFAR100 - Test
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
