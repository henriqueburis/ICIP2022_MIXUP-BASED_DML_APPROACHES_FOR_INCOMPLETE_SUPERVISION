import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt_
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt_.get_cmap('Blues')

    plt_.figure(figsize=(62, 57))
    plt_.imshow(cm, interpolation='nearest', cmap=cmap)
    plt_.title(title)
    plt_.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt_.xticks(tick_marks, target_names, rotation=45)
        plt_.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt_.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt_.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt_.tight_layout()
    plt_.ylabel('True label')
    plt_.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt_.savefig(title+'-confusion_matrix.png', dpi=100)
    plt_.close()
    #plt.show()


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 18))

    # We create a scatter plot.
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ax[0, 0].set_title("Sine function")

    # We add the labels for each cluster.
    txts = []
    for i in range(18):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts



def unmount_batch(feature_t,img_t,labels_t):
  feature_img_label = []
  feature = []
  img = []
  labels = []
  for i in range(len(feature_t)):
    for j in range(len(feature_t[i])):
      feature.append(feature_t[i][j])
      img.append(img_t[i][j])
      labels.append(labels_t[i][j])
  return np.array(feature),np.array(img),np.array(labels)

def unmount_batch_v2(feature_t,pred_y,true_l):
  feature_img_label = []
  feature = []
  pred_labels = []
  true_label = []
  for i in range(len(feature_t)):
    for j in range(len(feature_t[i])):
      feature.append(feature_t[i][j])
      pred_labels.append(pred_y[i][j][0])
      true_label.append(true_l[i][j])
  return np.array(feature),np.array(pred_labels),np.array(true_label)


def convert_label_(pred_y,true_l,list_label):
  #feature_img_label = []
  y_true = []
  y_pred = []
  for i in range(len(pred_y)):
    for j in range(len(pred_y[i])):
      y_pred.append(list_label[pred_y[i][j][0]])
      y_true.append(list_label[true_l[i][j]])
  return np.array(y_true),np.array(y_pred)


def CreateDir(path):
        try:
                os.mkdir(path)
        except OSError as error:
                print(error)



def chebyshev(features_u,features_l):
  dist = []
  #print(dist)
  for line in range(features_u.shape[0]):
    dist.append((torch.max(torch.abs(features_u[line]-features_l),dim=1).values).numpy())
  return np.array(dist)


def MCScore(log_prob):
  top2 = torch.topk(log_prob, k=2, dim=1).values[:,1]
  top1 = torch.topk(log_prob, k=2, dim=1).values[:,0]
  score_c = (top2 - top1) / torch.sum(log_prob,dim=1)
  return score_c



class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        #print(x)
        #b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = x*np.log(x)
        b = np.sum(b,axis=1)
        #b = -1.0 * b.sum()
        b = -1.0 * b
        return b



#HLoss = HLoss()


def TopK(final_dist,label_l,true_labels, k):
  final_dist = np.squeeze(np.array(final_dist))
  k = 20
  freq = np.zeros(100)
  topk = np.sort(final_dist)[0:k] # select top k

  for i in range(topk.shape[0]):
     pos = np.where(topk[i]==final_dist)[0]
     freq[label_l[pos]] += 1

  final_index = np.argmax(freq)
  return  final_index

def pairwise_distances_(feature_u, img_u, label_u, feature_l, img_l, label_l, true_labels,topk):
  labels = []
  correct = 0
  erro = 0

  dist_matrix_1 = pairwise_distances(np.array([feature_u]),feature_l , metric = 'euclidean')  #dist_matrix_1.shape(1,5000)
  dist_matrix_scaler_1 = (dist_matrix_1 - dist_matrix_1.min()) / (dist_matrix_1.max() - dist_matrix_1.min())

  dist_matrix_4 = pairwise_distances(np.array([feature_u]),feature_l , metric = 'chebyshev')
  dist_matrix_scaler_4 = (dist_matrix_4 - dist_matrix_4.min()) / (dist_matrix_4.max() - dist_matrix_4.min())

  dist_matrix_5 = pairwise_distances(np.array([feature_u]),feature_l , metric = 'cityblock')
  dist_matrix_scaler_5 = (dist_matrix_5 - dist_matrix_5.min()) / (dist_matrix_5.max() - dist_matrix_5.min())

  final_dist = (1+dist_matrix_scaler_1) * (1+dist_matrix_scaler_5) * (1+dist_matrix_scaler_4)
  
  k = topk
  final_index = TopK(final_dist,label_l, true_labels,k)

  if(true_labels == final_index): #label_l[final_index]):
    correct = correct +1;
  else:
    erro = erro + 1

  return correct,erro

