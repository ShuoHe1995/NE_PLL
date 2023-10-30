import argparse
import math

from model import ResNet18

from dataloaders import PartialDataloader
import os
from model import *
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='train batch size ', default=256, type=int)
parser.add_argument('--dataset', help='specify a dataset', default='cifar10', choices=['cifar10', 'cifar100', 'cifar100-H', 'kmnist', 'fmnist'], type=str)
parser.add_argument('--dataset_root', help='data', default='../../datasets/', type=str)
parser.add_argument('--num_classes', help='the number of class', default=10, type=int)
parser.add_argument('--epochs', help='number of epochs', type=int, default=450)
parser.add_argument('--seed', help='Random seed', default=123, type=int, required=False)
parser.add_argument('--gpu', help='used gpu id', default=0, type=int, required=False)
#######
parser.add_argument('--lr', help='learning rate', default=0.05, type=float)# 0.05
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight optimizer')
parser.add_argument('--partial_rate', help='partial rate', default=-1.0, type=float)
######
parser.add_argument('--eta', type=float, default=0.6, help='selection proportion')
parser.add_argument('--lam', type=float, default=0.99, help='EMA')
parser.add_argument('--warmup_epoch', help='number of epochs', type=int, default=1)
parser.add_argument('--rampup_epoch', help='number of epochs', type=int, default=100)
parser.add_argument('--expand_epoch', help='number of epochs', type=int, default=2)
parser.add_argument('--T', help='sharpen', default=0.5, type=float)

args = parser.parse_args()
print(args)

################### For reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
###################
torch.cuda.set_device(args.gpu)

def train_one_epoch(model1, model2, wd_class_entropy1, wd_class_entropy2, ud_entropy1, ud_entropy2, optimizer):
    model1.train()
    model2.train()
    w = linear_rampup(epoch, args.rampup_epoch)
    print_loss1, print_loss2 = 0, 0
    all_num_selected1, all_num_selected2 = 0, 0
    all_true_selected1, all_true_selected2 = 0, 0
    num_true_wd_expand1, num_true_wd_expand2, num_wd_expand1, num_wd_expand2 = 0, 0, 0, 0
    num_true_ud_expand1, num_true_ud_expand2, num_ud_expand1, num_ud_expand2 = 0, 0, 0, 0
    for i, (images1, images2, labels, true_labels, mask1, mask2, index) in enumerate(train_loader):
        x1, x2, py, ty = images1.cuda(), images2.cuda(), labels.long().cuda(), true_labels
        outputs11 = model1(x1)
        outputs12 = model1(x2)
        outputs21 = model2(x1)
        outputs22 = model2(x2)
        with torch.no_grad():
            sm_outputs11 = torch.softmax(outputs11, dim=1)
            sm_outputs21 = torch.softmax(outputs21, dim=1)
            sm_outputs11 = sm_outputs11 * py
            sm_outputs21 = sm_outputs21 * py
            targets1 = sm_outputs11 / sm_outputs11.sum(dim=1, keepdim=True)
            targets2 = sm_outputs21 / sm_outputs21.sum(dim=1, keepdim=True)

            if epoch >= args.expand_epoch:
                entropy1 = normalized_entropy(targets1, py, 1, False).cpu()
                entropy2 = normalized_entropy(targets2, py, 1, False).cpu()
                #########
                pred_label_matrix1 = F.one_hot(targets1.max(dim=1)[1], args.num_classes).cpu()
                pred_label_matrix2 = F.one_hot(targets2.max(dim=1)[1], args.num_classes).cpu()

                wd_class_threshold1 = (wd_class_entropy1 * pred_label_matrix1).sum(dim=1)
                wd_class_threshold2 = (wd_class_entropy2 * pred_label_matrix2).sum(dim=1)

                pred_label1, pred_label2 = targets1.max(dim=1)[1].cpu(), targets2.max(dim=1)[1].cpu()
                pred_selection = (pred_label1 == pred_label2)

                wd_expand_selection1 = (entropy1 < wd_class_threshold1) * (mask1 == 0)
                wd_expand_selection2 = (entropy2 < wd_class_threshold2) * (mask2 == 0)

                #### remaining ud

                ud_mask1 = (mask1 == 0) * (~wd_expand_selection1)
                ud_mask2 = (mask2 == 0) * (~wd_expand_selection2)

                ud_expand_selection1 = (entropy1 < ud_entropy1) * pred_selection * ud_mask1
                ud_expand_selection2 = (entropy2 < ud_entropy2) * pred_selection * ud_mask2

                num_wd_expand1 += torch.nonzero(wd_expand_selection1).shape[0]
                num_wd_expand2 += torch.nonzero(wd_expand_selection2).shape[0]
                num_true_wd_expand1 += torch.nonzero(pred_label1[wd_expand_selection1] == ty[wd_expand_selection1]).size(0)
                num_true_wd_expand2 += torch.nonzero(pred_label2[wd_expand_selection2] == ty[wd_expand_selection2]).size(0)

                num_ud_expand1 += torch.nonzero(ud_expand_selection1).shape[0]
                num_ud_expand2 += torch.nonzero(ud_expand_selection2).shape[0]
                num_true_ud_expand1 += torch.nonzero(pred_label1[ud_expand_selection1] == ty[ud_expand_selection1]).size(0)
                num_true_ud_expand2 += torch.nonzero(pred_label2[ud_expand_selection2] == ty[ud_expand_selection2]).size(0)

                # expand wd
                mask1[wd_expand_selection1] = 1
                mask2[wd_expand_selection2] = 1
                confidence1[index[wd_expand_selection1]] = targets1[wd_expand_selection1].cpu()
                confidence2[index[wd_expand_selection2]] = targets2[wd_expand_selection2].cpu()

                # expand ud
                mask1[ud_expand_selection1] = 1
                mask2[ud_expand_selection2] = 1
                temp1 = targets1[ud_expand_selection1] ** (1/args.T)
                ud_target1 = temp1 / temp1.sum(dim=1, keepdim=True)
                temp2 = targets2[ud_expand_selection2] ** (1/args.T)
                ud_target2 = temp2 / temp2.sum(dim=1, keepdim=True)
                confidence1[index[ud_expand_selection1]] = ud_target1.detach().clone().cpu()
                confidence2[index[ud_expand_selection2]] = ud_target2.detach().clone().cpu()

                # EMA ud
                avg_ud_entropy1 = entropy1[ud_mask1].mean()
                ud_entropy1 = args.lam * ud_entropy1 + (1-args.lam) * avg_ud_entropy1
                avg_ud_entropy2 = entropy2[ud_mask2].mean()
                ud_entropy2 = args.lam * ud_entropy2 + (1-args.lam) * avg_ud_entropy2

            selected_index1 = torch.nonzero(mask1 == 1).squeeze(dim=1)
            non_selected_index1 = torch.nonzero(mask1 == 0).squeeze(dim=1)
            selected_index2 = torch.nonzero(mask2 == 1).squeeze(dim=1)
            non_selected_index2 = torch.nonzero(mask2 == 0).squeeze(dim=1)

            all_true_selected1 += torch.nonzero(confidence1[index[selected_index1]].max(dim=1)[1] == ty[selected_index1]).size(0)
            all_true_selected2 += torch.nonzero(confidence2[index[selected_index2]].max(dim=1)[1] == ty[selected_index2]).size(0)

            all_num_selected1 += selected_index1.size(0)
            all_num_selected2 += selected_index2.size(0)

            one_hot_label1 = F.one_hot(confidence1[index[selected_index1]].max(dim=1)[1], args.num_classes)
            one_hot_label2 = F.one_hot(confidence2[index[selected_index2]].max(dim=1)[1], args.num_classes)

        #### model1
        ### mix up only for selected
        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        x11 = x1[selected_index1]
        targets11 = one_hot_label1.detach().clone()
        idx = torch.randperm(x11.size(0))
        x11_rand = x11[idx]
        targets11_rand = targets11[idx]
        x11_mix = l * x11 + (1 - l) * x11_rand
        targets11_mix = l * targets11 + (1 - l) * targets11_rand
        outputs11_mix = model1(x11_mix)
        loss1_mix = (-(targets11_mix.cuda() * F.log_softmax(outputs11_mix, dim=1)).sum(dim=1)).mean()

        ### cross-entropy
        loss1_ce1 = (-(confidence1[index[selected_index1]].cuda() * F.log_softmax(outputs11[selected_index1], dim=1)).sum(dim=1)).mean()
        loss1_ce2 = (-(confidence1[index[selected_index1]].cuda() * F.log_softmax(outputs12[selected_index1], dim=1)).sum(dim=1)).mean()
        loss1_add = (-((1-py[non_selected_index1]) * torch.log(1.0000001-F.softmax((outputs11[non_selected_index1]), dim=1))).sum(dim=1)).mean()


        if epoch >= args.expand_epoch:
            loss1 = loss1_ce1 + w * (loss1_ce2 + loss1_mix)
        else:
            if not torch.isnan(loss1_add):
                loss1 = loss1_ce1 + w * (loss1_ce2 + loss1_mix + loss1_add)

        #### model2
        ### mix up only for selected
        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        x12 = x1[selected_index2]
        targets12 = one_hot_label2.detach().clone()
        idx = torch.randperm(x12.size(0))
        x12_rand = x12[idx]
        targets12_rand = targets12[idx]
        x12_mix = l * x12 + (1 - l) * x12_rand
        targets12_mix = l * targets12 + (1 - l) * targets12_rand
        outputs12_mix = model2(x12_mix)
        loss2_mix = (-(targets12_mix.cuda() * F.log_softmax(outputs12_mix, dim=1)).sum(dim=1)).mean()

        ### cross-entropy
        loss2_ce1 = (-(confidence2[index[selected_index2]].cuda() * F.log_softmax(outputs21[selected_index2], dim=1)).sum(dim=1)).mean()
        loss2_ce2 = (-(confidence2[index[selected_index2]].cuda() * F.log_softmax(outputs22[selected_index2], dim=1)).sum(dim=1)).mean()
        loss2_add = (-((1-py[non_selected_index2]) * torch.log(1.0000001-F.softmax((outputs21[non_selected_index2]), dim=1))).sum(dim=1)).mean()

        if epoch >= args.expand_epoch:
            loss2 = loss2_ce1 + w * (loss2_ce2 + loss2_mix)
        else:
            if not torch.isnan(loss2_add):
                loss2 = loss2_ce1 + w * (loss2_ce2 + loss2_mix + loss2_add)

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update labeling confidence
        with torch.no_grad():
            sm_outputs1 = torch.softmax(outputs11, dim=1)
            sm_outputs1 = sm_outputs1 * py
            score1 = sm_outputs1 / sm_outputs1.sum(dim=1, keepdim=True)
            confidence1[index] = score1.cpu()
            sm_outputs2 = torch.softmax(outputs21, dim=1)
            sm_outputs2 = sm_outputs2 * py
            score2 = sm_outputs2 / sm_outputs2.sum(dim=1, keepdim=True)
            confidence2[index] = score2.cpu()

        print_loss1 += loss1.item()
        print_loss2 += loss2.item()

    precision1 = all_true_selected1 / all_num_selected1
    precision2 = all_true_selected2 / all_num_selected2

    print('Train Epoch [{}]: lr:{:.8f} w:{:.4f} loss1:{:.4f} loss2:{:.4f} ud_entropy1:{:.4f} ud_entropy2:{:.4f} all_true_selected1/all_num_selected1:{:d}/{:d}={:.4f}, num_true_wd_expand1/num_wd_expand1={:d}/{:d}, num_true_ud_expand1/num_ud_expand1={:d}/{:d}, all_true_selected2/all_num_selected2:{:d}/{:d}={:.4f}, num_true_wd_expand2/num_wd_expand2={:d}/{:d}, num_true_ud_expand2/num_ud_expand2={:d}/{:d}\n'.format(epoch + 1, optimizer.param_groups[0]['lr'], w, print_loss1/(i+1), print_loss2/(i+1), ud_entropy1, ud_entropy2, all_true_selected1, all_num_selected1, precision1, num_true_wd_expand1, num_wd_expand1, num_true_ud_expand1, num_ud_expand1, all_true_selected2, all_num_selected2, precision2, num_true_wd_expand2, num_wd_expand2, num_true_ud_expand2, num_ud_expand2))
    return

def warmup_one_epoch(model1, model2, optimizer):
    model1.train()
    model2.train()
    print_loss1, print_loss2 = 0, 0
    for i, (image1, image2, labels, true_labels, index) in enumerate(warmup_loader):
        x1, py = image1.cuda(), labels.long().cuda()
        outputs11 = model1(x1)
        outputs21 = model2(x1)
        loss1 = -torch.mean(torch.sum(F.log_softmax(outputs11, dim=1) * confidence1[index].cuda(), dim=1))
        loss2 = -torch.mean(torch.sum(F.log_softmax(outputs21, dim=1) * confidence2[index].cuda(), dim=1))
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ###update label confidence
        with torch.no_grad():
            sm_outputs1 = torch.softmax(outputs11, dim=1) * py
            score1 = sm_outputs1 / sm_outputs1.sum(dim=1, keepdim=True)
            confidence1[index] = score1.cpu()
            sm_outputs2 = torch.softmax(outputs21, dim=1) * py
            score2 = sm_outputs2 / sm_outputs2.sum(dim=1, keepdim=True)
            confidence2[index] = score2.cpu()
        ########
        print_loss1 += loss1.item()
        print_loss2 += loss2.item()
    print('Warmup Epoch [{}]: lr:{:.8f} loss1:{:.4f} loss2:{:.4f}\n'.format(epoch + 1, optimizer.param_groups[0]['lr'], print_loss1/(i+1),print_loss2/(i+1)))

def eval_one_epoch(model1, model2, eval_loader):
    model1.eval()
    model2.eval()
    epoch_entropy1, epoch_entropy2 = torch.zeros(num_data), torch.zeros(num_data)
    epoch_pred_label1, epoch_pred_label2 = torch.zeros(num_data), torch.zeros(num_data)
    with torch.no_grad():
        for batch_idx, (images, labels, true_labels, index) in enumerate(eval_loader):
            x, py = images.cuda(), labels.cuda()
            outputs1 = model1(x)
            outputs2 = model2(x)
            sm_outputs1 = torch.softmax(outputs1, dim=1) * py
            score1 = sm_outputs1 / sm_outputs1.sum(dim=1, keepdim=True)
            pred_label1 = score1.max(dim=1)[1]
            entropy1 = normalized_entropy(score1, py, 1, False)
            sm_outputs2 = torch.softmax(outputs2, dim=1) * py
            score2 = sm_outputs2 / sm_outputs2.sum(dim=1, keepdim=True)
            pred_label2 = score2.max(dim=1)[1]
            entropy2 = normalized_entropy(score2, py, 1, False)
            for i in range(images.size(0)):
                epoch_entropy1[index[i]] = entropy1[i].clone().detach().cpu()
                epoch_entropy2[index[i]] = entropy2[i].clone().detach().cpu()
                epoch_pred_label1[index[i]] = pred_label1[i].clone().detach().cpu()
                epoch_pred_label2[index[i]] = pred_label2[i].clone().detach().cpu()

    selected_mask1, selected_mask2 = torch.zeros(num_data), torch.zeros(num_data)
    num_selected_data1 = int(math.ceil(confidence1.shape[0] * args.eta))
    num_selected_data2 = int(math.ceil(confidence2.shape[0] * args.eta))
    selected_index1 = epoch_entropy1.sort(descending=False)[1][:num_selected_data1]
    selected_index2 = epoch_entropy2.sort(descending=False)[1][:num_selected_data2]
    selected_mask1[selected_index1] = 1
    selected_mask2[selected_index2] = 1
    true_cnt1 = (confidence1.max(dim=1)[1][selected_index1] == all_true_labels[selected_index1]).sum().int()
    true_cnt2 = (confidence2.max(dim=1)[1][selected_index2] == all_true_labels[selected_index2]).sum().int()
    num_selected1 = selected_mask1.sum().int()
    num_selected2 = selected_mask2.sum().int()
    precision1 = true_cnt1 / num_selected1
    precision2 = true_cnt2 / num_selected2

    # updating class threshold

    epoch_pred_label_matrix1 = F.one_hot(epoch_pred_label1.long(), num_class)
    epoch_pred_label_matrix2 = F.one_hot(epoch_pred_label2.long(), num_class)

    wd_class_entropy1 = (epoch_entropy1.unsqueeze(dim=1) * epoch_pred_label_matrix1)[selected_index1].mean(dim=0)
    wd_class_entropy2 = (epoch_entropy2.unsqueeze(dim=1) * epoch_pred_label_matrix2)[selected_index2].mean(dim=0)


    threshold_string1 = ' '.join([str(x.item())[:6] for x in wd_class_entropy1])
    threshold_string2 = ' '.join([str(x.item())[:6] for x in wd_class_entropy2])

    print('Evaluation Epoch [{}]: num_true_selected1/num_selected1:{:d}/{:d}={:.4f} num_true_selected2/num_selected2:{:d}/{:d}={:.4f}\n'.format(epoch + 1, true_cnt1, num_selected1, precision1, true_cnt2, num_selected2, precision2))
    print('WD class_threshold1:{:s} class_threshold2:{:s}'.format(threshold_string1, threshold_string2))
    return selected_mask1.float(), selected_mask2.float(), wd_class_entropy1, wd_class_entropy2

def return_same_idx(a, b):
    uniset, cnt = torch.cat([a, b]).unique(return_counts=True)
    result = torch.nonzero(cnt == 2).squeeze(dim=1)
    return uniset[result]

def test(model1, model2, test_acc_list):
    model1.eval()
    model2.eval()
    correct = 0
    correct2 = 0
    correctmean = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)

            score1, predicted = torch.max(outputs1, 1)
            score2, predicted_2 = torch.max(outputs2, 1)
            outputs_mean = (outputs1 + outputs2) / 2
            _, predicted_mean = torch.max(outputs_mean, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            correct2 += predicted_2.eq(targets).cpu().sum().item()
            correctmean += predicted_mean.eq(targets).cpu().sum().item()
    acc1 = 100. * correct / total
    acc2 = 100. * correct2 / total
    accmean = 100. * correctmean / total
    test_acc_list.append(accmean)
    print('Test  Epoch [{}]: Test Acc1:{:.2f} Test Acc2:{:.2f} Test Acc:{:.2f}\n'.format(epoch + 1, acc1, acc2, accmean))
    return accmean

def normalized_entropy(x, y, dim = -1, keepdim = None):
    p = x * y
    ent = -torch.where(p > 0, p * p.log2(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)
    temp = y.sum(dim=1).log2()
    temp[temp == 0] = 1
    n_ent = ent / temp
    return n_ent

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

partial_labels_file = os.path.join(args.dataset_root, args.dataset+'_pr='+str(args.partial_rate)+'.pt')
loader = PartialDataloader(root=args.dataset_root, dataset=args.dataset, partial_rate=args.partial_rate, partial_labels_file=partial_labels_file, batch_size=args.batch_size, num_workers=8)
warmup_loader, partial_label_matrix, all_true_labels = loader.run('warmup')
test_loader = loader.run('test')

tempY = partial_label_matrix.sum(dim=1).unsqueeze(1).repeat(1, partial_label_matrix.shape[1])
confidence1 = partial_label_matrix.float() / tempY
confidence2 = confidence1.detach().clone()
num_data = confidence1.shape[0]
num_class = confidence1.shape[1]

model1 = ResNet18(num_classes=args.num_classes).cuda()
model2 = ResNet18(num_classes=args.num_classes).cuda()


####
optimizer = torch.optim.SGD([{'params': model1.parameters()}, {'params': model2.parameters()}], args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*(args.lr_decay_rate**3))
####
test_acc_list = []
best_test_acc = 0
all_confidence1 = []
all_confidence2 = []

ud_entropy1 = 0
ud_entropy2 = 0

for epoch in range(args.epochs):

    if epoch < args.warmup_epoch:
        warmup_one_epoch(model1, model2, optimizer)
        eval_loader = loader.run('eval_train')
        all_mask1, all_mask2, wd_class_entropy1, wd_class_entropy2 = eval_one_epoch(model1, model2, eval_loader)
    else:
        eval_loader = loader.run('eval_train')
        all_mask1, all_mask2, wd_class_entropy1, wd_class_entropy2 = eval_one_epoch(model1, model2, eval_loader)
        train_loader = loader.run('train', mask1=all_mask1, mask2=all_mask2)
        train_one_epoch(model1, model2, wd_class_entropy1, wd_class_entropy2, ud_entropy1, ud_entropy2, optimizer)
    #####
    scheduler.step()
    num_true_confidence1 = torch.nonzero(confidence1.max(dim=1)[1] == all_true_labels).shape[0]
    num_true_confidence2 = torch.nonzero(confidence2.max(dim=1)[1] == all_true_labels).shape[0]
    print("acc_confidence1:{:4f} acc_confidence2:{:4f}\n".format(num_true_confidence1 / num_data, num_true_confidence2 / num_data))
    #####
    test_acc = test(model1, model2, test_acc_list)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    all_confidence1.append(confidence1.detach().clone().cpu())
    all_confidence2.append(confidence2.detach().clone().cpu())
    ######

last_test_acc = np.mean(test_acc_list[-5:])
print("best_test_acc:{:.4f} last_test_acc:{:.4f}\n".format(best_test_acc, last_test_acc))



