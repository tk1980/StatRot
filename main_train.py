import argparse
import os
import random
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from utils.train import validate, train1st, train2nd, adjust_learning_rate, save_checkpoint
import utils.datasets as datasets
from utils.ClassAwareSampler import ClassAwareSampler

import models

parser = argparse.ArgumentParser(description='Imbalanced Learning')
#Data
parser.add_argument('--data', default='./datasets/imagenetlt/', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', default='imagenetlt', type=str,
                    help='dataset name')

#Network
parser.add_argument('--arch', default='ResNet10Feature', type=str,
                    help='CNN architecture (default: ResNet10Feature)')
parser.add_argument('--rot-alpha', default=45, type=float, 
                    help='random rotation angle [degree] (default: 45)')

parser.add_argument('--first-model-file', default=None, type=str,
                    help='path to the 1st-trained model file')

parser.add_argument('--logit-adjust', action='store_true',
                    help='adjust logits based on class probabilities')

#Optimization
parser.add_argument('--lr', default=0.2, type=float) 
parser.add_argument('--epochs', default=180, type=int)  # 30
parser.add_argument('--batch-size', default=256, type=int) 
parser.add_argument('--weight-decay', default=0.0001, type=float) 
parser.add_argument('--momentum', default=0.9, type=float) 

#Utility
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--out-dir', default='./results', type=str,
                    help='path to output directory (default: ./)')
parser.add_argument('--save-all-checkpoints', dest='save_all_checkpoints', action='store_true',
                    help='save all the checkpoints')


def main():
    # performance stats
    stats = {'train_err1': [], 'train_err5': [], 'train_loss': [],
            'test_err1': [],  'test_err5': [],  'test_loss': []}

    # parameters
    args = parser.parse_args()
    
    num_classes = {'imagenetlt':1000, 'inat2018':8142, 'placeslt':365}[args.dataset]

    # output directory
    os.makedirs(args.out_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # 1st training or not?
    Do2ndTrain = args.first_model_file is not None and os.path.isfile(args.first_model_file)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model_feat = models.__dict__[args.arch]()
    if Do2ndTrain:
        print("=> loading 1st-trained model '{}'".format(args.first_model_file))
        checkpoint = torch.load(args.first_model_file, map_location=torch.device('cpu'))
        model_feat.load_state_dict(checkpoint['feat_state_dict'])
        for _, param in model_feat.named_parameters():
            param.requires_grad = False

        model_fc = torch.nn.Linear(model_feat.get_dim(), num_classes, bias=False)
    else:
        model_fc = models.StatRotLinear(model_feat.get_dim(), num_classes, bias=False, alpha=args.rot_alpha)
    print(model_feat)

    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'logs'))
    writer.close()

    # DataParallel will divide and allocate batch_size to all available GPUs
    model_feat = torch.nn.DataParallel(model_feat)
    model_feat.cuda()
    model_fc.cuda()
    criterion.cuda()

    # Data augmentation
    train_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
    test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    # Data loading code
    train_dataset = datasets.ListDataset(args.data, './datasets/{}/train.txt'.format(args.dataset), transform=train_transform)
    val_dataset = datasets.ListDataset(args.data, './datasets/{}/val.txt'.format(args.dataset), transform=test_transform)

    # Data Sampling
    if not Do2ndTrain: # 1st training
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)
    else: # 2nd training
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=ClassAwareSampler(train_dataset,num_samples_cls=4))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Optimizer
    lrs = ( 0.5*args.lr*np.cos(np.pi*np.arange(args.epochs)/args.epochs) + 0.5*args.lr ).tolist()
    model_params = [param for param in model_fc.parameters() if param.requires_grad]
    if not Do2ndTrain:
        model_params += [param for param in model_feat.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(model_params, lr=lrs[0], weight_decay=args.weight_decay, momentum=args.momentum)

    cudnn.benchmark = True

    # Do Train
    if args.logit_adjust:
        model_logit = models.LogitAdjust([train_dataset.targets.count(c) for c in range(num_classes)]).cuda()
        model_whole = torch.nn.Sequential(model_feat, model_fc, model_logit)
    else:
        model_whole = torch.nn.Sequential(model_feat, model_fc)
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lrs)

        # train for one epoch
        if not Do2ndTrain: # 1st training
            trnerr1, trnerr5, trnloss = train1st(train_loader, model_whole, criterion, optimizer, epoch)
        else: # 2nd training
            trnerr1, trnerr5, trnloss = train2nd(train_loader, model_whole, criterion, optimizer, epoch)

        # evaluate on validation set
        valerr1, valerr5, valloss = validate(val_loader, model_whole, criterion)

        # statistics
        stats['train_err1'].append(trnerr1)
        stats['train_err5'].append(trnerr5)
        stats['train_loss'].append(trnloss)
        stats['test_err1'].append(valerr1)
        stats['test_err5'].append(valerr5)
        stats['test_loss'].append(valloss)

        # remember best err@1
        is_best = valerr1 <= min(stats['test_err1'])

        # show and save results
        writer.add_scalar('LearningRate', lr, epoch)
        writer.add_scalar('Loss/train', trnloss, epoch)
        writer.add_scalar('Loss/test', valloss, epoch)
        writer.add_scalar('Error_1/train', trnerr1, epoch)
        writer.add_scalar('Error_1/test', valerr1, epoch)
        writer.add_scalar('Error_5/train', trnerr5, epoch)
        writer.add_scalar('Error_5/test', valerr5, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'feat_state_dict': model_feat.module.state_dict(),
            'fc_state_dict': model_fc.state_dict(),
            'stats': stats,
            'optimizer' : optimizer.state_dict(),
        }, is_best, not args.save_all_checkpoints, filename=os.path.join(args.out_dir, 'checkpoint-epoch{:d}.pth.tar'.format(epoch+1)))


    # show the final results
    minind = stats['test_err1'].index(min(stats['test_err1']))
    print('\n *BEST* Err@1 {:.3f} Err@5 {:.3f}'.format(stats['test_err1'][minind], stats['test_err5'][minind]))
    writer.add_hparams({'dataset':args.dataset, 'arch':args.arch, 'bsize':args.batch_size}, 
                        {'best/err_1':stats['test_err1'][minind], 'best/err_5':stats['test_err5'][minind], 'best/epoch':minind})
    writer.close()


if __name__ == '__main__':
    main()