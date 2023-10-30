import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from augment.cutout import Cutout2
from augment.autoaugment_extra import CIFAR10Policy
import os
import copy
from randaug import *
import numpy as np
from PIL import Image
import torch
import pickle
from sklearn.preprocessing import OneHotEncoder
from model import ResNet18, LeNet
from resnet import resnet
import codecs


def unpickle1(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def unpickle2(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_mnist_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).tolist()
        return parsed

def read_mnist_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(length, num_rows, num_cols)
        return parsed

class PartialDataset(Dataset):
    def __init__(self, root, dataset, mode, transform1, partial_rate=0, transform2=None, partial_labels_file=None, mask1=None, mask2=None):
        self.transform1 = transform1
        self.transform2 = transform2
        self.mode = mode

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle1('%s/test_batch' % root)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100' or dataset == 'cifar100-H':
                test_dic = unpickle1('%s/test' % root)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
            elif dataset == 'kmnist' or dataset == 'fmnist':
                self.test_data = read_mnist_image_file(os.path.join(root, 't10k-images-idx3-ubyte'))
                self.test_label = read_mnist_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))

        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root, n)
                    data_dic = unpickle1(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
            elif dataset == 'cifar100' or dataset == 'cifar100-H':
                train_dic = unpickle1('%s/train' % root)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
            elif dataset == 'kmnist' or dataset == 'fmnist':
                    train_data = read_mnist_image_file(os.path.join(root, 'train-images-idx3-ubyte'))
                    train_label = read_mnist_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))

            self.true_labels = train_label
            self.train_data = train_data

            ### generate partial labels
            if os.path.exists(partial_labels_file):
                partial_label_matrix = torch.load(partial_labels_file)
                print("Load Average Candidate Labels:{:.4f}\n".format(partial_label_matrix.sum() / partial_label_matrix.shape[0]))
            else:
                if partial_rate == -1.0:
                    partial_label_matrix = generate_instance_dependent_candidate_labels(dataset, self.train_data, self.true_labels)
                elif dataset == 'cifar100-H':
                    partial_label_matrix = generate_hierarchical_cv_candidate_labels(dataset, torch.tensor(self.true_labels), partial_rate)
                elif partial_rate > 0:
                    partial_label_matrix = generate_uniform_cv_candidate_labels(torch.tensor(self.true_labels), partial_rate)
                torch.save(partial_label_matrix, partial_labels_file)
            self.partial_label_matrix = partial_label_matrix
            if self.mode == 'train':
                self.mask1 = mask1
                self.mask2 = mask2

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        else:
            return len(self.train_data)

    def __getitem__(self, index):
        if self.mode == 'test':
            each_label = self.test_label[index]
            each_image = self.test_data[index]
            each_image = Image.fromarray(each_image)
            each_image = self.transform1(each_image)
            return each_image, each_label, index
        elif self.mode == 'train':
            each_partial_label = self.partial_label_matrix[index]
            each_true_label = self.true_labels[index]
            each_mask1 = self.mask1[index]
            each_mask2 = self.mask2[index]
            each_image = self.train_data[index]
            each_image = Image.fromarray(each_image)
            each_image1 = self.transform1(each_image)
            each_image2 = self.transform2(each_image)
            return each_image1, each_image2, each_partial_label, each_true_label, each_mask1, each_mask2, index
        elif self.mode == 'warmup':
            each_partial_label = self.partial_label_matrix[index]
            each_true_label = self.true_labels[index]
            each_image = self.train_data[index]
            each_image = Image.fromarray(each_image)
            each_image1 = self.transform1(each_image)
            each_image2 = self.transform2(each_image)
            return each_image1, each_image2, each_partial_label, each_true_label, index
        elif self.mode == 'eval_train':
            each_partial_label = self.partial_label_matrix[index]
            each_true_label = self.true_labels[index]
            each_image = self.train_data[index]
            each_image = Image.fromarray(each_image)
            each_image = self.transform1(each_image)
            return each_image, each_partial_label, each_true_label, index

class PartialDataloader():
    def __init__(self, root, dataset, partial_rate, partial_labels_file, batch_size, num_workers):
        dataset_dir_map = {'cifar10': root + 'cifar-10-batches-py', 'cifar10-LD': root + 'cifar-10-batches-py',
                           'cifar100': root + 'cifar-100-python', 'cifar100-H': root + 'cifar-100-python',
                           'fmnist': root + 'fmnist', 'kmnist': root + 'kmnist'}
        self.dataset_dir = dataset_dir_map[dataset]
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.partial_rate = partial_rate
        self.partial_labels_file = partial_labels_file
        if self.dataset == 'cifar10':
            self.transform_train1 = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.transform_train2 = copy.deepcopy(self.transform_train1)
            self.transform_train2.transforms.insert(0, RandomAugment(3, 5))
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
        elif self.dataset == 'cifar100' or self.dataset == 'cifar100-H':
            self.transform_train1 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_train2 = copy.deepcopy(self.transform_train1)
            self.transform_train2.transforms.insert(0, RandomAugment(3,5))
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
        elif self.dataset =='kmnist':
            self.transform_train1 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 4, padding_mode='reflect'),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            self.transform_train2 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 4, padding_mode='reflect'),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                Cutout2(n_holes=1, length=16),
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            self.transform_test = transforms.Compose([
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
        elif self.dataset == 'fmnist':
            self.transform_train1 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 4, padding_mode='reflect'),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ])
            self.transform_train2 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 4, padding_mode='reflect'),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                Cutout2(n_holes=1, length=16),
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ])

            self.transform_test = transforms.Compose([
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081)),
            ])

    def run(self, mode, mask1=None, mask2=None):
        if mode == 'train':
            train_dataset = PartialDataset(dataset=self.dataset, root=self.dataset_dir, partial_rate=self.partial_rate, partial_labels_file=self.partial_labels_file,
                                           transform1=self.transform_train1, mode="train",
                                            transform2=self.transform_train2, mask1=mask1, mask2=mask2)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.num_workers,
                                      pin_memory=True, drop_last=True)
            return train_loader
        elif mode == 'warmup':
            warmup_dataset = PartialDataset(dataset=self.dataset, root=self.dataset_dir, partial_rate=self.partial_rate, partial_labels_file=self.partial_labels_file,
                                           transform1=self.transform_train1, mode="warmup",
                                            transform2=self.transform_train2)
            warmup_loader = DataLoader(dataset=warmup_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.num_workers,
                                      pin_memory=True, drop_last=True)
            return warmup_loader, warmup_dataset.partial_label_matrix, torch.tensor(warmup_dataset.true_labels)
        elif mode == 'test':
            test_dataset = PartialDataset(dataset=self.dataset, root=self.dataset_dir, transform1=self.transform_test, mode='test')
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

            return test_loader

        elif mode == 'eval_train':
            train_dataset = PartialDataset(dataset=self.dataset, root=self.dataset_dir, partial_labels_file=self.partial_labels_file,
                                           partial_rate=self.partial_rate,
                                           transform1=self.transform_test, mode="eval_train")
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size * 4,
                                      shuffle=False, num_workers=self.num_workers,
                                      pin_memory=True, drop_last=False)
            return train_loader


def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1):
    assert dataname == 'cifar100-H'

    meta = unpickle2('../../datasets/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]: i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]

        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = p_1
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    #print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K):  # for each class
            if jj == train_labels[j]:  # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0
    print("Average Candidate Labels:{:.4f}\n".format(partialY.sum() / n))
    return partialY


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    # partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K)
    # inject label noise if noisy_rate > 0
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    #print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        random_n_j = random_n[j]
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)


    num_avg = partialY.sum() / n
    print("Uniform generation: the average number of candidate Labels:{:.4f}.".format(num_avg))
    return partialY


def binarize_class(y):
        label = y.reshape(len(y), -1)
        enc = OneHotEncoder(categories='auto')
        enc.fit(label)
        label = enc.transform(label).toarray().astype(np.float32)
        label = torch.from_numpy(label)
        return label

def generate_instance_dependent_candidate_labels(dataset, data, targets):
    with torch.no_grad():
        c = max(targets) + 1
        y = binarize_class(torch.tensor(targets, dtype=torch.long))

        batch_size = 2000
        rate = 0.4 if dataset != 'cifar100' else 0.04
        if dataset in ['kmnist', 'fmnist']:
            model = LeNet(out_dim=c, in_channel=1, img_sz=28)
            weight_path = './model_path/' + dataset + '_clean_DA1.pth'
        elif dataset in ['cifar10']:
            model = resnet(depth=32, n_outputs=c)
            weight_path = './model_path/cifar10_original.pt'
        elif dataset in ['cifar100', 'cifar100-H']:
            model = ResNet18(num_classes=100)
            weight_path = './model_path/cifar100_original.pth'
        model.load_state_dict(torch.load(weight_path, map_location='cuda:0'))
        model = model.cuda()
        data = torch.from_numpy(np.copy(data))
        train_X, train_Y = data.cuda(), y.cuda()


        if dataset in ['kmnist', 'fmnist']:
            train_X = train_X.unsqueeze(dim=1).to(torch.float32)
        elif dataset in ['cifar10', 'cifar100', 'cifar100-H']:
            train_X = train_X.permute(0, 3, 1, 2).to(torch.float32)

        train_p_Y_list = []
        step = train_X.size(0) // batch_size
        for i in range(0, step):
            if dataset in ['kmnist', 'fmnist', 'cifar100', 'cifar100-H']:
                outputs = model(train_X[i * batch_size:(i + 1) * batch_size])
            elif dataset in ['cifar10']:
                _, outputs = model(train_X[i * batch_size:(i + 1) * batch_size])
            train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
            partial_rate_array = torch.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()
            train_p_Y[torch.where(z == 1)] = 1.0
            train_p_Y_list.append(train_p_Y)
        train_p_Y = torch.cat(train_p_Y_list, dim=0)
        assert train_p_Y.shape[0] == train_X.shape[0]
    final_y = train_p_Y.cpu().clone()
    print("Average Candidate Labels:{:.4f}\n".format(final_y.sum() / final_y.shape[0]))
    return final_y.cpu()






