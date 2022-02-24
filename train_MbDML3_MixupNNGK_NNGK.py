import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as functional
from torchvision.utils import save_image
from neighbours import find_neighbours
from classifier import GaussianKernels
from loader import MultiFolderLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import os
import subprocess
import argparse
import scipy
from copy import deepcopy
from sklearn.manifold import TSNE
from utils import *

parser = argparse.ArgumentParser(description="Train Gaussian kernel classifier using Resnet18 or 50.")
parser.add_argument("--data_dir", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--test", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--save_dir", required=True, type=str, help="Models are saved to this directory.")
parser.add_argument("--num_classes", required=True, type=int, help="Number of training classes to use.")
parser.add_argument("--im_ext", default="jpg", type=str, help="Dataset image file extensions (e.g. jpg, png).")
parser.add_argument("--gpu_id", default=None, type=int, help="GPU ID. CPU is used if not supplied.")
parser.add_argument("--sigma", default=10, type=int, help="Gaussian sigma.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=1e-5, type=int, help="learning_rate")
parser.add_argument("--update_interval", default=5, type=int, help="Stored centres/neighbours are updated every update_interval epochs.")
parser.add_argument("--max_epochs", default=50, type=int, help="Maximum training length (epochs).")
parser.add_argument("--topk", default=20, type=int, help="top k.")
parser.add_argument("--input_size", default=256, type=int, help="input size img.")

parser.add_argument("--name", default=" ", required=True, type=str, help="Dataset file name extensions (e.g. cifar10, cifar100).")
####MIXUP
parser.add_argument('--alpha', default= 1, type=float,help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--scale_mixup', default= 2, type=float,help='scaling the mixup loss')
parser.add_argument('--beta', default= 1, type=float,help='scaling the gauss loss')
#### TSNE GRAPH
parser.add_argument('--tsne_graph', default=True, type=str, help='if true save tsne imagen')
args = parser.parse_args()

seed =args.name+"-EP"+str(args.max_epochs)+"-SM"+str(args.scale_mixup)+"-A"+str(args.alpha)+"-B"+str(args.beta)
print('seed==>',seed)

writer = SummaryWriter(comment="-"+seed)

result_model = list()
result_model.append("SEED::  "+str(seed)+ "\n")
result_model.append("epochs::  "+str(args.max_epochs)+ "scale_mixup::  "+str(args.scale_mixup)+ "alpha::  "+str(args.alpha)+  "beta::  "+str(args.beta)+ "\n")
result_model.append("============================= \n")

"""
Configuration
"""

#Data info
input_size = args.input_size   
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

#Resnet50 model
model = torchvision.models.resnet50(pretrained=True)

#Remove fully connected layer
modules = list(model.children())[:-1]

#--------------------------------------------#
from collections import OrderedDict

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
#--------------------------------------------#

#modules.append(nn.Flatten())
modules.append(Flatten())
model = nn.Sequential(*modules)

kernel_weights_lr = args.learning_rate*1
num_neighbours    = int(args.save_dir.replace('results/neighbour=',''))
eval_interval     = args.update_interval

#Set GPU ID or 'cpu'
if args.gpu_id is None:
	device = torch.device('cpu')
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
	device = torch.device('cuda:0')



def CreateDir(path):
        try:
                os.mkdir(path)
        except OSError as error:
                print(error)


CreateDir(args.save_dir)

"""
Set up DataLoaders
"""

#Transformations/pre-processing operations
train_transforms = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.RandomCrop((input_size,input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

update_transforms = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.CenterCrop((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

test_transforms = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


train_dataset  = MultiFolderLoader(args.data_dir, train_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)
update_dataset = MultiFolderLoader(args.data_dir, update_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)
test_dataset = MultiFolderLoader(args.test, test_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)

#Data loaders to handle iterating over datasets
train_loader  = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=True,  num_workers=3)
update_loader = DataLoader(update_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)

"""
Create Gaussian kernel classifier
"""
model = model.to(device)
#best_state = model.to(device)
#model = model.train()
model = model.eval()

def update_centres():

	#Disable dropout, use global stats for batchnorm
	model.eval()

	#Disable learning
	with torch.no_grad():

		#Update stored centres
		for i, data in enumerate(update_loader, 0):

			# Get the inputs; data is a list of [inputs, labels]. Send to GPU
			inputs, labels, indices = data
			inputs = inputs.to(device)

			#Extract features for batch
			extracted_features = model(inputs)
			#print(extracted_features.shape[0])

			#Save to centres tensor
			idx = i*args.batch_size
			centres[idx:idx + extracted_features.shape[0], :] = extracted_features

	#model.train()
	model.eval()

	return centres


def save_model():
	torch.save(model.state_dict(), args.save_dir + "/"+seed+"model.pt")
	torch.save(kernel_classifier.state_dict(), args.save_dir + "/"+seed+"classifier.pt")
	torch.save(centres, args.save_dir + "/"+seed+"centres.pt")

num_train = len(update_loader.dataset)
print(num_train)

with torch.no_grad():
	num_dims = model(torch.randn(1,3,input_size,input_size).to(device)).size(1)

#Create tensor to store kernel centres
centres = torch.zeros(num_train,num_dims).type(torch.FloatTensor).to(device)
print("Size of centres is {0}".format(centres.size()))

#Create tensor to store labels of centres
centre_labels = torch.LongTensor(update_dataset.get_all_labels()).to(device)

#Create Gaussian kernel classifier
kernel_classifier = GaussianKernels(args.num_classes, num_neighbours, num_train, args.sigma)
kernel_classifier = kernel_classifier.to(device)


"""
Set up loss and optimiser
"""

criterion = nn.NLLLoss()

optimiser = optim.Adam([
                {'params': model.parameters()},
                {'params': kernel_classifier.parameters(), 'lr': kernel_weights_lr}
            ], lr=args.learning_rate)

#exp_lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=step_gamma)

##################################################### MIXUP #######################################################

criterion_mixup = nn.CrossEntropyLoss()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


############################################################################ FIM ####################################

"""
 Test
"""

def test():
    print("Test!")
    running_correct_ = 0
    for i, data in enumerate(tqdm(test_loader), 0):
        inputs, labels, indices = data
        inputs  = inputs.to(device)
        labels  = labels.to(device).view(-1)
        output = model(inputs)
        dist_matrix = torch.cdist(output, centres)
        neighbours_tr = torch.argsort(dist_matrix)[:,0:num_neighbours]
        indices_2 = np.arange(0,output.size(0))
        log_prob, prob_real = kernel_classifier( output , centres, centre_labels, neighbours_tr[indices_2, :] )
        pred = log_prob.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        running_correct_ += correct/args.batch_size

    acc = running_correct_/len(test_loader)

    print('####### ACC_Test_train =',acc)
    return acc


"""
Training
"""
print("Begin training...")
acc_geral = -1
best_epoch = -1
for epoch in range(args.max_epochs):  # loop over the dataset multiple times

	#Update stored kernel centres
	if (epoch % args.update_interval) == 0:

		print("Updating kernel centres...")
		centres = update_centres()
		print("Finding training set neighbours...")
		centres = centres.cpu()
		neighbours_tr = find_neighbours( num_neighbours, centres )
		centres = centres.to(device)
		print("Finished update!")

		if epoch > 0:
                    acc_ataual = test()
                    writer.add_scalar('ACC/test', acc_ataual, epoch)
                    if(acc_geral <= acc_ataual):
                       best_epoch = epoch
                       acc_geral = acc_ataual
                       save_model()
                    #test()

	#Training
	running_loss = 0.0
	running_correct = 0
	for i, data in enumerate(train_loader, 0):
		# Get the inputs; data is a list of [inputs, labels]. Send to GPU
                inputs, labels, indices = data
                inputs  = inputs.to(device)
                labels  = labels.to(device).view(-1)
                indices = indices.to(device)

                inputs_mixup, targets_a, targets_b, lam = mixup_data(inputs, labels,args.alpha, True)
                inputs_mixup, targets_a, targets_b = map(Variable, (inputs_mixup,targets_a, targets_b))
                outputs_mixup,outputs_2 = kernel_classifier( model(inputs_mixup), centres, centre_labels, neighbours_tr[indices, :] )
                loss_mixup = mixup_criterion(criterion_mixup, outputs_mixup, targets_a, targets_b, lam)

		# Zero the parameter gradients
                optimiser.zero_grad()

                log_prob, prob_real = kernel_classifier( model(inputs), centres, centre_labels, neighbours_tr[indices, :])
                loss_gauss = criterion(log_prob, labels) # gaussian loss
                loss = (args.beta * loss_gauss) + (args.scale_mixup * loss_mixup)
                
                loss.backward()
                optimiser.step()
                
                running_loss += loss.item()
                writer.add_scalar('Loss/loss_gauss', loss_gauss, (epoch*len(train_loader.dataset)/32)+i)
                writer.add_scalar('Loss/loss_mixup', loss_mixup, (epoch*len(train_loader.dataset)/32)+i)
                writer.add_scalar('Loss/loss', loss, (epoch*len(train_loader.dataset)/32)+i)
                
                #Get the index of the max log-probability
                pred = log_prob.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                running_correct += correct

	#Print statistics at end of epoch
	if True:
		print('[{0}, {1:5d}] loss: {2:.3f}, accuracy: {3}/{4} ({5:.4f}%)'.format(
			epoch + 1, i + 1, running_loss / len(train_loader.dataset),
                        running_correct, len(train_loader.dataset), 100. * running_correct / len(train_loader.dataset)))
		writer.add_scalar('ACC/accuracy', 100. * running_correct / len(train_loader.dataset), (epoch*len(train_loader.dataset)/32)+i)
		running_loss = 0.0
		running_correct = 0

	#exp_lr_scheduler.step()
	#adjust_learning_rate(optimiser, epoch)




#Update centres final time when done
print("Updating kernel centres (final time)...")
centres = update_centres()

print("Best ACC_Teste_train::  "+str(acc_geral)+ "  best_epoch::  "+str(best_epoch)+ "\n")
result_model.append("============================= \n")
result_model.append("Best ACC_Teste_train::  "+str(acc_geral)+ "  best_epoch::  "+str(best_epoch)+ "\n")

print("########################################################################################")
print("########################################################################################")

############################ Load best state model ######################################
model.load_state_dict(torch.load(args.save_dir + "/"+seed+"model.pt",map_location=device))
kernel_classifier.load_state_dict(torch.load(args.save_dir + "/"+seed+"classifier.pt"))
centres = torch.load(args.save_dir + "/"+seed+"centres.pt")
print(centres)
model = model.eval()
########################################################################################

print("Train")

feature_t= []
labels_t = []
pred_t = []
running_correct = 0
for i, data in enumerate(tqdm(train_loader), 0):
    inputs, labels, indices = data
    inputs  = inputs.to(device)
    labels  = labels.to(device).view(-1)
    indices = indices.to(device)
    output = model(inputs)
    dist_matrix = torch.cdist(output, centres)
    neighbours_tr = torch.argsort(dist_matrix)[:,0:num_neighbours]
    indices = np.arange(0,output.size(0))
    log_prob, prob_real = kernel_classifier( output , centres, centre_labels, neighbours_tr[indices, :] )
    pred = log_prob.argmax(dim=1, keepdim=True)

    feature_t.append(output.data.cpu().numpy())
    labels_t.append(labels.data.cpu().numpy())
    pred_t.append(pred.data.cpu().numpy())
    
    correct = pred.eq(labels.view_as(pred)).sum().item()
    running_correct += correct/args.batch_size


print('####### AAC_Label = ',running_correct/len(train_loader))

result_model.append("============================= \n")
result_model.append("AAC_Label XL::  "+str(running_correct/len(train_loader))+ "\n")

feature_l,pred_l, true_l = unmount_batch_v2(feature_t,pred_t,labels_t)
np.savez(seed+'xl.npz', featuresL=feature_l, predL=pred_l, trueL=true_l)

if(args.tsne_graph == "True"):
  view_tsne = TSNE(random_state=123).fit_transform(feature_l)
  plt.scatter(view_tsne[:,0], view_tsne[:,1], c=pred_l, alpha=0.2, cmap='Set1')
  plt.title(seed+'-tsne-XL',
          fontdict={'family': 'serif',
                    'color' : 'darkblue',
                    #'weight': 'bold',
                    'size': 8})
  plt.savefig(seed+'-tsne-XL.png', dpi=120)


######### Test  ##############
print("Test")

feature_test= []
labels_test = []
pred_test = []

running_correct_ = 0
for i, data in enumerate(tqdm(test_loader), 0):
    inputs, labels, indices = data
    inputs  = inputs.to(device)
    labels  = labels.to(device).view(-1)
    output = model(inputs)
    dist_matrix = torch.cdist(output, centres)
    neighbours_tr = torch.argsort(dist_matrix)[:,0:num_neighbours]
    indices_2 = np.arange(0,output.size(0))
    log_prob, prob_real = kernel_classifier( output , centres, centre_labels, neighbours_tr[indices_2, :] )
    pred = log_prob.argmax(dim=1, keepdim=True)

    feature_test.append(output.data.cpu().numpy())
    labels_test.append(labels.data.cpu().numpy())
    pred_test.append(pred.data.cpu().numpy())


    correct = pred.eq(labels.view_as(pred)).sum().item()
    running_correct_ += correct/args.batch_size

print('####### ACC_Test_pgl =',running_correct_/len(test_loader))

result_model.append("============================= \n")
result_model.append("ACC_Test::  "+str(running_correct_/len(test_loader))+ "\n")

feature_tt,pred_tt, label_tt = unmount_batch_v2(feature_test,pred_test,labels_test)
np.savez(seed+'test.npz', featuresTest=feature_tt, predTest=pred_tt, trueTest=label_tt)

if(args.tsne_graph == "True"):
  view_tsne_u = TSNE(random_state=123).fit_transform(feature_tt)
  plt.scatter(view_tsne_u[:,0], view_tsne_u[:,1], c=label_tt, alpha=0.2, cmap='Set1')
  plt.title(seed+'-tsne_Test',
          fontdict={'family': 'serif',
                    'color' : 'darkblue',
                    #'weight': 'bold',
                    'size': 8})
  plt.savefig(seed+'-tsne_Test.png', dpi=120)


arquivo = open(seed+".txt", "a")
arquivo.writelines(result_model)
arquivo.close()

print("finished")
