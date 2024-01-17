import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import accuracy_score

import random 

def compute_D_acc(discriminator, test_dataloader, seen_classes, novel_classes, task_no, batch_size=128, opt1='gzsl',
                  opt2='test_seen', psuedo_ft=None, psuedo_lb=None):
    """
    Compute the accuracy of the discriminator on the test set using cosine similarity and identifier projections
    :param discriminator: the discriminator model
    :param test_dataloader: the test dataloader
    :param seen_classes: the seen classes for each tasks
    :param novel_classes: the novel classes for each tasks
    :param task_no: current task number
    :param batch_size: batch size
    :param opt1: the type of the evaluation test space
    :param opt2: the type of the evaluation test set
    :param psuedo_ft: the pseudo features for the current task included (not used in our code)
    :param psuedo_lb: the pseudo labels for the current task included (not used in our code)
    """

    if psuedo_ft is not None:
        data = Data.TensorDataset(psuedo_ft, psuedo_lb)
        test_loader = Data.DataLoader(data, batch_size=batch_size)
    else:
        test_loader = test_dataloader.get_loader(opt2, batch_size=batch_size)
    att = test_dataloader.data['whole_attributes'].cuda()
    if opt1 == 'gzsl':
        search_space = np.arange(att.shape[0])
    if opt1 == 'zsl':
        search_space = test_dataloader.data['unseen_label']

    pred_label = []
    true_label = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.cuda(), labels.cuda()
            features = F.normalize(features, p=2, dim=-1, eps=1e-12)
            if psuedo_ft is None:
                features = features.unsqueeze(1).repeat(1, search_space.shape[0], 1)
            else:
                features = features.squeeze(1).unsqueeze(1).repeat(1, search_space.shape[0], 1)
            semantic_embeddings = discriminator(att).cuda()
            semantic_embeddings = F.normalize(semantic_embeddings, p=2, dim=-1, eps=1e-12)
            cosine_sim = F.cosine_similarity(semantic_embeddings, features, dim=-1)
            predicted_label = torch.argmax(cosine_sim, dim=1)
            predicted_label = search_space[predicted_label.cpu()]
            pred_label = np.append(pred_label, predicted_label)
            true_label = np.append(true_label, labels.cpu().numpy())
    pred_label = np.array(pred_label, dtype='int')
    true_label = np.array(true_label, dtype='int')
    acc = 0
    unique_label = np.unique(true_label)
    for i in unique_label:
        idx = np.nonzero(true_label == i)[0]
        acc += accuracy_score(true_label[idx], pred_label[idx])
    acc = acc / unique_label.shape[0]
    return acc


def next_batch_unseen(unseen_attr, unseen_labels, batch_size):
    """
    Get the next batch of randomized unseen data
    :param unseen_attr: the unseen attributes
    :param unseen_labels: the unseen labels
    :param batch_size: batch size
    """
    idx = torch.randperm(unseen_attr.shape[0])[0:batch_size]
    unsn_at = unseen_attr[idx]
    unsn_lbl = unseen_labels[idx]
    return unsn_at.unsqueeze(1), unsn_lbl


def next_batch(batch_size, attributes, feature, label):
    """
    Get the next batch of randomized data
    :param batch_size: batch size
    :param attributes: the attributes
    :param feature: the features
    :param label: the labels
    """
    idx = torch.randperm(feature.shape[0])[0:batch_size]
    batch_feature = feature[idx]
    batch_label = label[idx]
    batch_attr = attributes[idx]
    return batch_feature, batch_label, batch_attr


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use for random number generators.
    """
    # Seed Python's built-in random module
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Seed PyTorch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)