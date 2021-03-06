import os
import numpy as np
#from torch_geometric.utils import sparse_to_dense
# from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import radius_graph
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch
import shutil
from torch.nn import Parameter
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix




def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

#graph read write function
def pt_to_gexf(numpyfile, savepath, sample = [0.1, 0.2, 0.3, 0.4 ,0.5 , 0.6, 0.7 ,0.9, 1]):
    # data = torch.load(torchfile)
    # feature = data.x.numpy()
    coordinates = np.load(numpyfile)
    coordinates = coordinates/1500.
    coordinates = torch.from_numpy(coordinates).to(torch.float)
    num_nodes = coordinates.shape[0]
    for sampling_ratio in sample:
        num_sample = int(num_nodes * sampling_ratio)
        choice = np.random.choice(num_nodes, num_sample, replace=True)
        coordinates_new = coordinates[choice]

        edge_index = radius_graph(coordinates_new, 100, None, True, 8)
        adj = sparse_to_dense(edge_index)
        # adj = to_dense_adj(edge_index)
        adj = adj.numpy()
        coordinates_new = coordinates_new.numpy()
        G = nx.from_numpy_matrix(adj)
        x = dict(enumerate(coordinates_new[:, 0].tolist(), 0))
        y = dict(enumerate(coordinates_new[:, 1].tolist(), 0))
        nx.set_node_attributes(G, x, 'x')
        nx.set_node_attributes(G, y, 'y')
        nx.write_gexf(G, os.path.join( savepath, str(sampling_ratio) + numpyfile.split('/')[-1].replace('.npy','.gexf')))


def data_to_gexf(pt_file, savepath):
    data = torch.load(pt_file)
    G = to_networkx(data)
    nx.draw(G)
    plt.savefig(os.path.join(savepath,pt_file.split('/')[-1].replace('.pt','.png')))
    # nx.write_gexf(G, os.path.join(savepath,pt_file.split('/')[-1].replace('.pt','.gexf')))


def output_to_gexf(coordinate, adj, assign_matrix_list, path):
    '''
        Visualize the assignment matrix
    '''
    x = dict(enumerate(coordinate[:,0].tolist(), 0))
    y = dict(enumerate(coordinate[:,1].tolist(), 0))
    assign_matrix_dicts = []
    for idx, assign_matrix in enumerate(assign_matrix_list):
        assign_matrix = np.argmax(assign_matrix, axis=1)
        assign_matrix = dict(enumerate(assign_matrix.flatten(),0))
        assign_matrix = {k: int(v)  for k, v in assign_matrix.items()}
        assign_matrix_dicts.append(assign_matrix)
    mapping_assign_dicts = {}
    mapping_assign_dicts['assign_1'] = assign_matrix_dicts[0].copy()
    if len(assign_matrix_dicts)>1:
        for idx, deeper_assign_matrix in enumerate(assign_matrix_dicts, 1):
            if idx == 1:
                first_assign = assign_matrix_dicts[0].copy()
                continue
            map_assign_matrix = dict([(k, deeper_assign_matrix[v]) for (k,v) in first_assign.items()])
            mapping_assign_dicts['assign_%d'%idx] = map_assign_matrix.copy()
            first_assign = map_assign_matrix.copy()
            # not sparse format
    assert  adj.shape[0] == adj.shape[1], 'the adjacent matrix should have same row and col'
    G = nx.from_numpy_matrix(adj)
    # import pdb;pdb.set_trace()
    for k,v in mapping_assign_dicts.items():
    #     name = namestr[idx]
        nx.set_node_attributes(G, v, k)
    nx.set_node_attributes(G, x, 'x')
    nx.set_node_attributes(G, y, 'y')
    nx.write_gexf(G, path)

# load save functions
def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdirs(os.path.dirname(fpath))
    torch.save(state, fpath)
    #if is_best:
    #    shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'model_best.pth.tar'))

def load_checkpoint(fpath):
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model
# optimizer

def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))


# other temp things
def dataset_analysis(keys):
    train_root_list = ['path-to-dataset-feature',]
    if 'max_node' in keys:
        trainall, train_max_node_number= max_nodes_in_dataset(train_root_list)
        print(train_max_node_number, )
    elif 'normalized' in keys:
        norm1,std1 = normalization(train_root_list)
        return norm1,std1


def max_nodes_in_dataset(pathlist):
    all_number = []
    for path in pathlist:
        lists = os.listdir(path)

        for f in lists:
            g = np.load(os.path.join(path, f))
            nodes = g.shape[0]
            all_number.append(nodes)
    maxn = max(all_number)
    return all_number, maxn


def normalization(pathlist):
    all_number = []
    for path in pathlist:
        lists = os.listdir(path)
        for f in lists:
            g = np.load(os.path.join(path, f)).astype(np.double)
            all_number.append(g)
    allins = np.vstack(all_number)
    norm = np.mean(allins, 0)
    std = np.std(allins, 0)
    return norm, std

class FarthestSampler2:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        farthest_pts = np.zeros((k, 2), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts


class FarthestSampler:
    def __init__(self):
        pass
    def __call__(self, arr, k):
        farthest_pts = np.zeros((k), dtype=np.int32)
        farthest_pts[0] = np.random.randint(arr.shape[0])
        farthest_distance = arr[farthest_pts[0]]
        for i in range(1,k):
            farthest_pts[i] = np.argmax(farthest_distance)
            farthest_distance = np.minimum(farthest_distance, arr[farthest_pts[i]])
        return farthest_pts

def filter_sampled_indice(indice, num):
    total = [i for i in range(num)]
    a = list(filter(lambda x:x not in indice, total))
    return a

@torch.no_grad()
def visualize_network(writer, net, tensor):
    tensor = tensor.to(next(net.parameters()).device)
    writer.add_graph(net, tensor)

def _get_lastlayer_params(net):
    """get last trainable layer of a net
    Args:
        network architectur
    Returns:
        last layer weights and last layer bias
    """
    last_layer_weights = None
    last_layer_bias = None
    layer_name = None
    for name, para in net.named_parameters():
        if 'weight' in name:
            last_layer_weights = para
            layer_name = name.replace('.weight', '')
        if 'bias' in name:
            last_layer_bias = para
            layer_name = name.replace('.bias', '')

    return last_layer_weights, last_layer_bias, layer_name

def visualize_lastlayer(writer, net, n_iter):
    weights, bias, layer_name = _get_lastlayer_params(net)
    writer.add_scalar('LastLayerGradients/grad_norm2_weights({})'.format(layer_name), weights.grad.norm(), n_iter)
    writer.add_scalar('LastLayerGradients/grad_norm2_bias({})'.format(layer_name), bias.grad.norm(), n_iter)

def visualize_scalar(writer, name, scalar, n_iter):
    """visualize scalar"""
    writer.add_scalar(name, scalar, n_iter)


def visualize_param_hist(writer, net, n_iter):
    """visualize histogram of params"""
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, n_iter)

def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    Args:
        dataset: instance of torch.nn.Dataset
    Returns:
        return: mean and std of this dataset
    """

    mean_r = 0
    mean_g = 0
    mean_b = 0

    #opencv BGR channel
    for img, _ in dataset:
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img, _ in dataset:

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std


class Metric:
    def __init__(self, normal_class_id=0, num_classes=3):
        self.normal_class_id = normal_class_id
        self.image_prefix = dict()
        self.image_label = dict()
        self.image_scores = dict()
        self.num_classes = num_classes
        #self.patch_acc = 0
        #self.image_acc_mul = 0
        #self.image_acc_bin = 0

        self.total_patches = 0
        self.correct_patches = 0

        #self.total_images = [0, 0, 0] # grade 1 grade 2 grade 3
        #self.correct_images = [0, 0, 0]
        self.total_images = [0] * num_classes # grade 1 grade 2 grade 3
        self.correct_images = [0] * num_classes

    def _binary_cls(self, cls_id, data_list):
        output = []
        for i in data_list:
            if cls_id == i:
                output.append(1)
            else:
                output.append(0)
        return output
    def _compute_sen_sep(self, preds, labels):
        tn, fp, fn, tp = confusion_matrix(preds, labels).ravel()
        sen = tp / (tp + fn)
        sep = tn / (tn + fp)
        return sen, sep

    def sen_spe(self):
        print('image_sensitivity_specificity....')
        gts = []
        preds = []
        for key, value in self.image_prefix.items():
            pred = value.index(max(value))
            label = self.image_label[key]
            gts.append(label)
            preds.append(pred)

        for cls_id in range(max(gts) + 1):
            sen, sep = self._compute_sen_sep(
                self._binary_cls(cls_id, preds),
                self._binary_cls(cls_id, gts),
            )
            print('class {}, sensitivity: {:02f}, specificity: {:02f}'.format(
                cls_id, sen, sep
            ))







            #correct += pred == label
            #total += 1
            #if pred != label:
                #print(key, value, label)


    def update(self, preds, labels, pathes):

        scores = torch.sigmoid(preds).tolist()
        preds = torch.argmax(preds, dim=-1)

        self.correct_patches += preds.cpu().eq(labels.cpu()).sum().item()
        self.total_patches += len(labels)

        preds = preds.tolist()
        labels = labels.tolist()

        for pred, label, path, score in zip(preds, labels, pathes, scores):
            #prefix = path.split('.')[0]
            #prefix = path.split('_grade_')[0]
            #image_label = int(path.split('_grade_')[1][0]) - 1
            prefix = path.split('_grade_')[0]
            image_label = int(path.split('_grade_')[1][0]) - 1
            # label is patch level label
            # image_label is image level label
            #print(prefix)
            if prefix not in self.image_prefix:
                self.image_prefix[prefix] = [0] * self.num_classes
                self.image_label[prefix] = image_label
                #print(self.num_classes, label)
                assert label == image_label
                assert label < self.num_classes
                assert pred < self.num_classes

            if label == 0:
                s = 1 - score[label]
            else:
                s = score[label]
            self.image_scores[prefix] = s

            assert image_label == label

            # add to image stats
            self.image_prefix[prefix][pred] += 1

            ###############################################################
            #if max(self.image_prefix[prefix]) > 1:
            #    print(max(self.image_prefix[prefix]), path, label, self.image_prefix)
            #    raise Exception('image_prefix[prefix] should be less than 1')
            #assert sum(self.image_prefix[prefix]) <= 1

    def patch_accuracy(self):
        return self.correct_patches / self.total_patches

    def image_acc_three_class(self):

        grade1 = 0
        grade1_correct = 0
        grade2 = 0
        grade2_correct = 0

        correct = 0
        total = 0
        #res = []
        print('image_acc_three....')
        for key, value in self.image_prefix.items():
            pred = value.index(max(value))
            label = self.image_label[key]

            correct += pred == label
            total += 1
            if pred != label:
                print(key, value, label)

            #if label == 0:
            #    if pred == label:
            #        grade1_correct += 1
            #    #else:
            #        #print('grade 1 failed', key)
            #        #tt = key.split('_grade_')[0]
            #        #print(key, 1111)
            #        #print(os.system(
            #        #        'cat /data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops.csv | grep {}'.format(tt)
            #        #))
            #    grade1 += 1

            #elif label == 1:
            #    if pred == label:
            #        grade2_correct += 1
            #    #else:
            #        #tt = key.split('_grade_')[0]
            #        #print(key, 1111)
            #        #print(os.system(
            #        #        'cat /data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Labels/5Crops.csv | grep {}'.format(tt)
            #        #))
            #    #else:
            #        #print('grade 2 failed', key)
            #    grade2 += 1
            #else:
            #    raise ValueError('ffffff')

        #print(grade2_correct / grade2,  grade1_correct / grade1)
        #print(correct / total)
        #print(grade2_correct, grade2,  grade1_correct, grade1)
        #import sys; sys.exit()
        return correct / total

    def auc(self):
        preds = []
        gts = []
        for key, value in self.image_scores.items():
            label = self.image_label[key]
            preds.append(value)
            gts.append(label)
        try:
            return metrics.roc_auc_score(gts, preds)
        except:
            return 1


    def kappa(self):
        preds = []
        gts = []
        for key, value in self.image_prefix.items():
            pred = value.index(max(value))
            label = self.image_label[key]
            preds.append(pred)
            gts.append(label)

        return cohen_kappa_score(gts, preds, weights='quadratic')

    def image_acc_binary_class(self):
        correct = 0
        total = 0

        print('image_bin....')
        for key, value in self.image_prefix.items():
            pred = value.index(max(value))
            label = self.image_label[key]

            if label == self.normal_class_id:
                if pred == self.normal_class_id:
                    correct += 1
            else:
                if pred != self.normal_class_id:
                    correct += 1

            total  += 1
        print('done ....')

        return correct / total


import torch
from torch_sparse import coalesce


def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes


def dense_to_sparse(tensor):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        tensor (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    print('dense_to_sparse.......')
    assert tensor.dim() == 2
    index = tensor.nonzero().t().contiguous()
    value = tensor[index[0], index[1]]
    return index, value


def sparse_to_dense(edge_index, edge_attr=None, num_nodes=None):
    r"""Converts a sparse adjacency matrix given by edge indices and edge
    attributes to a dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor): Edge weights or multi-dimensional edge features.
        num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

        :rtype: :class:`Tensor`
        """
    print('sparse to dense........')
    N = maybe_num_nodes(edge_index, num_nodes)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    adj = torch.sparse_coo_tensor(edge_index, edge_attr, torch.Size([N, N]))
    return adj.to_dense()

class EarlyStopping:
    def __init__(self, patient, metric):
        if metric == 'acc':
            self.acc = 0
        else:
            raise ValueError('wrong metric')

        self.counter = 0
        self.patient = patient
        self.metric = metric


    def update(self, val):
        self.counter += 1
        if self.metric == 'acc':
            if val > self.acc:
                self.acc = val
                self.counter = 0

            if self.counter > self.patient:
                print('early stopping.... ')
                import sys; sys.exit()



if __name__ == '__main__':
    # pt_to_gexf('/data/hdd1/syh/PycharmProjects/CGC-Net/data/proto/coordinate/CRC/fold_4/2_low_grade/Patient_001_02_Low-Grade.npy','/data/hdd1/syh/PycharmProjects/CGC-Net/data/proto/fix_fuse_hover_knn/0/fold_1/1_normal')
    # pt_to_gexf('/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/coordinate/CRC/fold_1/1_normal/H09-00804_A2H_E_1_2_grade_1_2241_4033.npy','/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/gexf')
    pt_to_gexf('/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/coordinate/CRC/fold_1/1_normal/H09-00320_A2H_E_1_2_grade_1_2241_2017.npy','/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/normal_gexf')
    # data_to_gexf('/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/fix_fuse_cia_knn/29/fold_1/H09-00804_A2H_E_1_2_grade_1_2241_4033.pt','/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/gexf2')
