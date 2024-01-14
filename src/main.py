import argparse
import json
import os
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from torch import optim
from torch.autograd import Variable

from data import DATA_LOADER as dataloader
from rw_loss import compute_rw_imitative_loss, compute_rw_real_loss, compute_rw_creative_loss

cwd = os.path.dirname(os.getcwd())

seen_acc_history = []
unseen_acc_history = []
harmonic_mean_history = []
best_seen_acc_history = []
best_unseen_acc_history = []
best_harmonic_mean_history = []


class Generator(nn.Module):
    def __init__(self, feature_size=2048, att_size=85):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2 * att_size, 1024)
        self.fc2 = nn.Linear(1024, feature_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, noise, att):
        if len(att.shape) == 3:
            h = torch.cat((noise, att), 2)
        else:
            h = torch.cat((noise, att), 1)
        feature = torch.relu(self.fc1(h))
        feature = torch.sigmoid(self.fc2(feature))
        return feature


class Discriminator(nn.Module):
    def __init__(self, feature_size=2048, att_size=85):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(att_size, 1024)
        self.fc2 = nn.Linear(1024, feature_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, att):
        att_embed = torch.relu(self.fc1(att))
        att_embed = torch.relu(self.fc2(att_embed))
        return att_embed


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


class Train_Dataloader:
    def __init__(self, train_feat_seen, train_label_seen, batch_size=32):
        self.data = {'train_': train_feat_seen, 'train_label': train_label_seen}

    def get_loader(self, opt='train_', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt + 'label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=True)
        return data_loader


class Test_Dataloader:
    def __init__(self, test_attr, test_seen_f, test_seen_l, test_seen_a, test_unseen_f, test_unseen_l, test_unseen_a,
                 batch_size=32):
        try:
            labels = torch.cat((test_seen_l, test_unseen_l))
        except:
            labels = test_seen_l
        self.data = {'test_seen': test_seen_f, 'test_seenlabel': test_seen_l,
                     'whole_attributes': test_attr,
                     'test_unseen': test_unseen_f, 'test_unseenlabel': test_unseen_l,
                     'seen_label': np.unique(test_seen_l),
                     'unseen_label': np.unique(test_unseen_l)}

    def get_loader(self, opt='test_seen', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt + 'label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size)
        return data_loader


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


def train(task_no, Neighbors, discriminator, generator, data, seen_classes, novel_classes, replay_feat, replay_lab,
          replay_attr, feature_size=2048, attribute_size=85, dlr=0.005, glr=0.005, batch_size=64,
          unsn_batch_size=8, epochs=50, lambda1=10.0, alpha=1.0, avg_feature=None, all_classes=None, num_tasks=None):
    """
    ####### The training function for each task
    :param task_no: current task number
    :param Neighbors: number of neighbors for lsr gan loss
    :param discriminator: discriminator model
    :param generator: generator model
    :param data: data loader
    :param seen_classes: number of seen classes in each tasks
    :param novel_classes: number of novel classes in each tasks
    :param replay_feat: features of replayed classes
    :param replay_lab: labels of replayed classes
    :param replay_attr: attributes of replayed classes
    :param feature_size: feature size of the feature vector
    :param attribute_size: attribute size of the attribute vector
    :param dlr: learning rate of the discriminator
    :param glr: learning rate of the generator
    :param batch_size: batch size for seen class batches
    :param unsn_batch_size: batch size for unseen class batches
    :param epochs: number of epochs
    :param lambda1: coefficient of the cosine similarity losses 
    :param alpha: coeficient of the classification (cross entropy losses)
    :param avg_feature: average feature vector of the seen classes
    :param all_classes: all classes in the dataset
    :param num_tasks: number of tasks
    """
    if task_no == 1:
        train_feat_seen, train_label_seen, train_att_seen = data.task_train_data(task_no, seen_classes, all_classes,
                                                                                 novel_classes, num_tasks, Neighbors)
        avg_feature = torch.zeros((seen_classes), feature_size).float()
        cls_num = torch.zeros(seen_classes).float()

        # Compute the average features of seen classes
        for i, l1 in enumerate(train_label_seen):
            avg_feature[l1] += train_feat_seen[i]
            cls_num[l1] += 1

        for ul in np.unique(train_label_seen):
            avg_feature[ul] = avg_feature[ul] / cls_num[ul]

        avg_feature = avg_feature.cuda()
        semantic_relation_sn = data.idx_mat
        semantic_values_sn = data.semantic_similarity_seen

    else:
        train_feat_seen, train_label_seen, train_att_seen = data.task_train_data(task_no, seen_classes, all_classes,
                                                                                 novel_classes, num_tasks, Neighbors)
        avg_feature_prev = avg_feature
        avg_feature = torch.zeros((seen_classes) * task_no, feature_size).float()
        cls_num = torch.zeros(seen_classes * task_no).float()
        avg_feature[:seen_classes * (task_no - 1), :] = avg_feature_prev.cpu()

        # Compute average features with previous classes as well when task no > 1
        for i, l1 in enumerate(train_label_seen):
            avg_feature[l1] += train_feat_seen[i]
            cls_num[l1] += 1

        for ul in np.unique(train_label_seen):
            avg_feature[ul] = avg_feature[ul] / cls_num[ul]

        avg_feature = avg_feature.cuda()
        semantic_relation_sn = data.idx_mat
        semantic_values_sn = data.semantic_similarity_seen

    #  Concatenate old task classes
    if task_no == 1:
        whole_feat_seen = train_feat_seen
        whole_labels_seen = train_label_seen

    if task_no > 1:
        whole_feat_seen = torch.cat((train_feat_seen, replay_feat))
        whole_labels_seen = torch.cat((train_label_seen, replay_lab))

    # Initialize loaders and optimizers
    train_loader = Train_Dataloader(whole_feat_seen, whole_labels_seen, batch_size=batch_size)
    print(f'Current seen label {sorted(list(set([i.item() for i in train_label_seen])))},')
    att_per_task = data.attribute_mapping(seen_classes, novel_classes, task_no).cuda()
    attr_seen_exc = att_per_task[0:seen_classes * task_no, :]

    train_data_loader = train_loader.get_loader('train_', batch_size)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=dlr, weight_decay=0.00001)
    G_optimizer = optim.Adam(generator.parameters(), lr=glr, weight_decay=0.00001)

    # Initialize learnable dictionary for attribute generation
    # Attributes are initialized with the heuristic but let to change as the learning goes on
    if opt.attribute_generation_method == "learnable":
        h_gen_att = nn.Embedding(opt.all_classes, data.att_size, max_norm=torch.norm(attr_seen_exc, dim=1).max()).cuda()
        random_idx = torch.randint(0, attr_seen_exc.shape[0], [opt.all_classes, 1], device="cuda")
        random_idx_2 = torch.randint(0, attr_seen_exc.shape[0], [opt.all_classes, 1], device="cuda")
        initialize_alpha = (torch.rand(opt.all_classes) * (.8 - .2) + .2).cuda()
        hallucinated_attributes = attr_seen_exc.squeeze(1)[random_idx_2.squeeze(1)].T * (initialize_alpha) + \
                                  attr_seen_exc[random_idx.squeeze(1)].squeeze(1).T * (1 - initialize_alpha)
        h_gen_att.weight.data = hallucinated_attributes.T
        att_optimizer = optim.Adam(h_gen_att.parameters(), lr=opt.dic_lr, weight_decay=0.00001)
    entory_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # Arrays to hold metrics
    epoch_seen_accuracy_history = []
    epoch_unseen_accuracy_history = []
    epoch_harmonic_accuracy_history = []

    for epoch in range(epochs):
        if epoch % 10 == 0:
            print("Epoch {}/{}...".format(epoch + 1, epochs))

        # Dictionary to hold loss histories
        losses = {
            "d_loss": [],
            "d_imitative_loss": [],
            "d_rw_real_loss": [],
            "g_rw_loss_seen": [],
            "g_seen_loss": [],
            "creativity_loss": [],
            "g_rw_creativity_loss": [],
            "g_rw_imitative_loss": [],
            "g_unseen_loss": [],
        }

        # Mini batch loops
        for feature, label in train_data_loader:
            if feature.shape[0] == 1:
                continue
            _att_sn = attr_seen_exc.unsqueeze(0).repeat([feature.size(0), 1, 1])
            feature, label = feature.cuda(), label.cuda()
            feature_norm = F.normalize(feature, p=2, dim=-1, eps=1e-12)
            D_optimizer.zero_grad()
            if opt.attribute_generation_method == "learnable":
                att_optimizer.zero_grad()

            # Generate fake seen features using generator
            att_bs_seen = att_per_task[label]
            noise = torch.FloatTensor(att_bs_seen.shape[0], att_bs_seen.shape[1], att_bs_seen.shape[2]).cuda()
            noise.normal_(0, 1)
            psuedo_seen_features = generator(noise, att_bs_seen)

            #  Embed seen attributes to the visual feature space
            semantic_embedding = discriminator(att_bs_seen)
            semantic_embed_norm = F.normalize(semantic_embedding, p=2, dim=-1, eps=1e-12)

            #---------Discriminator Update---------

            # Compute discriminator classification loss
            psuedo_seen_features_norm = F.normalize(psuedo_seen_features, p=2, dim=-1, eps=1e-12)
            real_cosine_similarity = lambda1 * F.cosine_similarity(semantic_embed_norm, feature_norm, dim=-1)
            pseudo_cosine_similarity = lambda1 * F.cosine_similarity(semantic_embed_norm, psuedo_seen_features_norm, dim=-1)
            real_cosine_similarity = torch.mean(real_cosine_similarity)
            pseudo_cosine_similarity = torch.mean(pseudo_cosine_similarity)

            # Discriminator regularization loss
            att_task_emb = discriminator(att_per_task[:(seen_classes) * task_no])
            mse_d = mse_loss(avg_feature, att_task_emb)

            # Compute discriminator real-fake loss
            _att_D = discriminator(_att_sn)
            _att_D_norm = F.normalize(_att_D, p=2, dim=-1, eps=1e-12)
            real_features = feature_norm.unsqueeze(1).repeat([1, (seen_classes) * task_no, 1])
            real_cosine_sim = lambda1 * F.cosine_similarity(_att_D_norm, real_features, dim=-1)
            cls_label = label
            classification_losses = entory_loss(real_cosine_sim, cls_label.squeeze())

            # Discriminator optimization step
            d_loss = - torch.log(real_cosine_similarity) + torch.log(
                pseudo_cosine_similarity) + alpha * classification_losses + mse_d
            d_loss.backward(retain_graph=True)
            losses["d_loss"].append(d_loss.item())
            D_optimizer.step()

            # ---------Generator Update---------
            # Generator classification loss
            G_optimizer.zero_grad()
            fake_features = psuedo_seen_features_norm.repeat([1, seen_classes * task_no, 1])
            fake_cosine_sim = lambda1 * F.cosine_similarity(_att_D_norm, fake_features, dim=-1)
            pseudo_classification_loss = entory_loss(fake_cosine_sim, cls_label.squeeze())

            # Generator regularization loss
            Euclidean_loss = Variable(torch.Tensor([0.0]), requires_grad=True).cuda()
            Correlation_loss = Variable(torch.Tensor([0.0]), requires_grad=True).cuda()
            for i in range(seen_classes * task_no):
                sample_idx = (label == i)
                if (sample_idx == 1).sum().item() == 0:
                    Euclidean_loss += 0.0
                if (sample_idx == 1).sum().item() != 0:
                    G_sample_cls = psuedo_seen_features[sample_idx, :]
                    if G_sample_cls.shape[0] > 1:
                        generated_mean = G_sample_cls.mean(dim=0)
                    else:
                        generated_mean = G_sample_cls
                    Euclidean_loss += (generated_mean - avg_feature[i]).pow(2).sum().sqrt()
                    for n in range(Neighbors):
                        generated_mean_norm = F.normalize(generated_mean, p=2, dim=-1, eps=1e-12)
                        avg_norm = F.normalize(avg_feature[semantic_relation_sn[i, n]], p=2, dim=-1, eps=1e-12)
                        Neighbor_correlation = F.cosine_similarity(generated_mean_norm, avg_norm, dim=-1)
                        lower_limit = semantic_values_sn[i, n] - 0.01
                        if opt.dataset == "CUB":
                            upper_limit = semantic_values_sn[i, n] + 0.04
                        else:
                            upper_limit = semantic_values_sn[i, n] + 0.01
                        lower_limit = torch.as_tensor(lower_limit.astype('float')).cuda()
                        upper_limit = torch.as_tensor(upper_limit.astype('float')).cuda()
                        corr = Neighbor_correlation.cuda()
                        margin = (torch.max(corr - corr, corr - upper_limit)) ** 2 + (
                            torch.max(corr - corr, lower_limit - corr)) ** 2
                        Correlation_loss += margin
            Euclidean_loss *= 1.0 / (seen_classes * task_no) * 1
            lsr_seen = (Correlation_loss) * opt.corr_weight
            (lsr_seen).backward(retain_graph=True)

            # Generator seen loss optimization step
            g_loss_seen = - torch.log(pseudo_cosine_similarity) + alpha * pseudo_classification_loss + Euclidean_loss
            g_loss_seen.backward(retain_graph=True)
            losses["g_seen_loss"].append(g_loss_seen.item())
            G_optimizer.step()

            # Generator inductive loss
            if opt.attribute_generation_method == "interpolation":
                hallucinate_1 = att_bs_seen.squeeze(1)
                random_permutations = torch.randperm(att_bs_seen.shape[0])
                hallucinate_1 = hallucinate_1[random_permutations]
                hallucinate_2 = att_bs_seen.squeeze(1)
                random_permutations = torch.randperm(att_bs_seen.shape[0])
                hallucinate_2 = hallucinate_2[random_permutations]
                creative_alpha = (torch.rand(len(label)) * (.8 - .2) + .2).cuda()
                hallucinated_attr = (
                        creative_alpha * hallucinate_1.T + (1 - creative_alpha) * hallucinate_2.T).T.unsqueeze(1)
            elif opt.attribute_generation_method == "learnable":
                random_idx = torch.randint(0, h_gen_att.weight.shape[0], [feature.shape[0], 1], device="cuda")
                hallucinated_attr = h_gen_att(random_idx)
            else:
                raise NotImplementedError


            # Cretivity loss
            noise = torch.FloatTensor(att_bs_seen.shape[0], att_bs_seen.shape[1], att_bs_seen.shape[2]).cuda()
            noise.normal_(0, 1)
            creative_features = generator(noise, hallucinated_attr)
            hallucinated_projections_norm = F.normalize(discriminator(hallucinated_attr), p=2, dim=-1,eps=1e-12)
            creative_features = F.normalize(creative_features, p=2, dim=-1, eps=1e-12)
            creative_features_repeated = creative_features.repeat([1, (seen_classes) * task_no,1])
            creative_cosine_sim = lambda1 * F.cosine_similarity(_att_D_norm, creative_features_repeated,dim=-1)
            G_fake_C = F.log_softmax(creative_cosine_sim, dim=-1)
            entropy_GX_fake = (G_fake_C / G_fake_C.data.size(1)).mean()
            loss_creative = - opt.creative_weight * entropy_GX_fake
            loss_creative_realistic_part = -torch.mean(lambda1 * F.cosine_similarity(hallucinated_projections_norm, creative_features,
                                              dim=-1))
            creativity_loss = (loss_creative + loss_creative_realistic_part)
            creativity_loss.backward(retain_graph=True)
            losses["creativity_loss"].append(creativity_loss.item())

            # Generative random walk loss on discriminator
            discr_rw_imitative_walker_loss, discr_rw_imitative_visit_loss = compute_rw_imitative_loss(rw_config,
                                                                                                      task_no * seen_classes,
                                                                                                      att_per_task,
                                                                                                      discriminator,
                                                                                                      generator)
            discr_rw_imitative_loss = discr_rw_imitative_walker_loss + rw_config.loss_weights.get('visit_loss',
                                                                                                  1.0) * discr_rw_imitative_visit_loss
            discr_rw_imitative_loss = rw_config.loss_weights.discr.imitative * discr_rw_imitative_loss
            discr_rw_imitative_loss.backward(retain_graph=True)
            losses["d_imitative_loss"].append(discr_rw_imitative_loss.item())

            # Generative random walk loss on generator
            discr_rw_real_walker_loss, discr_rw_real_visit_loss = compute_rw_real_loss(rw_config,
                                                                                       task_no * seen_classes,
                                                                                       att_per_task, feature, label,
                                                                                       generator, discriminator)
            discr_rw_real_loss = discr_rw_real_walker_loss + rw_config.loss_weights.get('visit_loss',
                                                                                        1.0) * discr_rw_real_visit_loss
            discr_rw_real_loss = rw_config.loss_weights.discr.real * discr_rw_real_loss
            discr_rw_real_loss.backward(retain_graph=True)
            losses["d_rw_real_loss"].append(discr_rw_real_loss.item())

            gen_rw_imitative_walker_loss, gen_rw_imitative_visit_loss = compute_rw_imitative_loss(rw_config,
                                                                                                  task_no * seen_classes,
                                                                                                  att_per_task,
                                                                                                  discriminator,
                                                                                                  generator)
            gen_rw_imitative_loss = gen_rw_imitative_walker_loss + rw_config.loss_weights.get('visit_loss',
                                                                                              1.0) * gen_rw_imitative_visit_loss
            gen_rw_imitative_loss = rw_config.loss_weights.gen.imitative * gen_rw_imitative_loss
            losses["g_rw_imitative_loss"].append(gen_rw_imitative_loss.item())
            gen_rw_imitative_loss.backward(retain_graph=True)
            gen_rw_creative_walker_loss, gen_rw_creative_visit_loss = compute_rw_creative_loss(rw_config,
                                                                                               task_no * seen_classes,
                                                                                               att_per_task,
                                                                                               discriminator,
                                                                                               creative_features.squeeze(
                                                                                                   1), generator)
            gen_rw_creative_loss = gen_rw_creative_walker_loss + rw_config.loss_weights.get('visit_loss',
                                                                                            1.0) * gen_rw_creative_visit_loss

            gen_rw_creative_loss = opt.grw_creative_weight * gen_rw_creative_loss
            losses["g_rw_creativity_loss"].append(gen_rw_creative_loss.item())
            gen_rw_creative_loss.backward(retain_graph=True)

            G_optimizer.step()

            if opt.attribute_generation_method == "learnable":
                att_optimizer.step()
        if epoch == epochs - 1:

            test_seen_f, test_seen_l, test_seen_a, test_unseen_f, test_unseen_l, test_unseen_a = data.task_test_data_(
                task_no, seen_classes, all_classes, novel_classes, num_tasks)
            print(f'current unseen label {sorted(list(set([i.item() for i in test_unseen_l])))}')
            att_per_task_ = data.attribute_mapping(seen_classes, novel_classes, task_no).cuda()
            test_dataloader = Test_Dataloader(att_per_task_, test_seen_f, test_seen_l, test_seen_a, test_unseen_f,
                                              test_unseen_l, test_unseen_a)
            D_seen_acc = compute_D_acc(discriminator, test_dataloader, seen_classes, novel_classes, task_no,
                                       batch_size=batch_size, opt1='gzsl', opt2='test_seen')
            D_unseen_acc = compute_D_acc(discriminator, test_dataloader, seen_classes, novel_classes, task_no,
                                         batch_size=batch_size, opt1='gzsl', opt2='test_unseen')
            if D_unseen_acc == 0 or D_seen_acc == 0:
                D_harmonic_mean = 0
            else:
                D_harmonic_mean = (2 * D_seen_acc * D_unseen_acc) / (D_seen_acc + D_unseen_acc)

            epoch_seen_accuracy_history.append(D_seen_acc)
            if task_no != opt.num_tasks:
                epoch_unseen_accuracy_history.append(D_unseen_acc)
                epoch_harmonic_accuracy_history.append(D_harmonic_mean)
            print(
                f'Best accuracy at task {task_no} at epoch {epoch}: unseen : {D_unseen_acc:.4f}, seen : {D_seen_acc:.4f}, H : {D_harmonic_mean:.4f}')

        loss_metrics = {
            "epoch": epoch,
            "task_no": task_no,
        }
        for key, value in losses.items():
            if len(value) > 0:
                loss_metrics[f"losses/{key}"] = statistics.mean(value)
        with open(f"logs/{opt.run_name}/losses.json", "a+") as f:
            json.dump(loss_metrics, f)
            f.write("\n")

    seen_acc_history.append(D_seen_acc)
    unseen_acc_history.append(D_unseen_acc)
    harmonic_mean_history.append(D_harmonic_mean)

    final_model_acc_history = []
    forgetting_measure = 0
    if task_no == opt.num_tasks:
        for t in range(1, opt.num_tasks + 1):
            test_seen_f, test_seen_l, test_seen_a, test_unseen_f, test_unseen_l, test_unseen_a = data.task_test_data_(t,
                                                                                                                      seen_classes,
                                                                                                                      all_classes,
                                                                                                                      novel_classes,
                                                                                                                      num_tasks)
            att_per_task_ = data.attribute_mapping(seen_classes, novel_classes, t).cuda()
            test_dataloader = Test_Dataloader(att_per_task_, test_seen_f, test_seen_l, test_seen_a, test_unseen_f,
                                              test_unseen_l, test_unseen_a)
            final_model_acc = compute_D_acc(discriminator, test_dataloader, seen_classes, novel_classes, t,
                                            batch_size=batch_size, opt1='gzsl', opt2='test_seen')
            final_model_acc_history.append(final_model_acc)
        final_model_acc_difference = np.array(final_model_acc_history) - np.array(seen_acc_history)
        forgetting_measure = np.mean(final_model_acc_difference[:-1])

    checkpoint = {
        "task_no": task_no,
        "discriminator": discriminator.state_dict(),
        "generator": generator.state_dict(),
        "optimizer_D": D_optimizer.state_dict(),
        "optimizer_G": G_optimizer.state_dict(),
    }
    if opt.attribute_generation_method == "learnable":
        checkpoint["learned_attributes"] = h_gen_att.state_dict()
    try:
        torch.save(checkpoint, f'checkpoints/{opt.run_name}/checkpoint_task_{task_no}.pth')
        with open(f"logs/{opt.run_name}/metrics.json", "w") as f:
            json.dump({
                "seen_acc_history": seen_acc_history,
                "unseen_acc_history": unseen_acc_history,
                "harmonic_mean_history": harmonic_mean_history,
                "mean_seen_acc": statistics.mean(seen_acc_history),
                "mean_unseen_acc": statistics.mean(unseen_acc_history),
                "mean_harmonic_mean": statistics.mean(harmonic_mean_history),
                "forgetting_measure": forgetting_measure,
                "final_model_acc_history": final_model_acc_history,
            }, f)
    except:
        print("Saving failed")

    # Update replay buffer
    size_for_this = opt.buffer_size // (seen_classes * task_no)
    replay_feature_buffer = []
    replay_label_buffer = []
    replay_attr_buffer = []

    if replay_feat is not None:
        for i in range(0, seen_classes * (task_no - 1)):
            mask = replay_lab.squeeze(1) == i
            replay_feature_buffer.append(replay_feat[mask][:size_for_this])
            replay_label_buffer.append(replay_lab[mask][:size_for_this])
            replay_attr_buffer.append(replay_attr[mask][:size_for_this])

    for i in range(seen_classes * (task_no - 1), seen_classes * task_no):
        mask = train_label_seen.squeeze(1) == i
        selected_seen_feature = train_feat_seen[mask]
        selected_seen_label = train_label_seen[mask]
        selected_seen_attr = train_att_seen[mask]
        replay_feature_buffer.append(selected_seen_feature[:size_for_this])
        replay_label_buffer.append(selected_seen_label[:size_for_this])
        replay_attr_buffer.append(selected_seen_attr[:size_for_this])

    replay_feat = torch.cat(replay_feature_buffer, dim=0)
    replay_lab = torch.cat(replay_label_buffer, dim=0)
    replay_attr = torch.cat(replay_attr_buffer, dim=0)

    print(f'Replay data shape : {replay_feat.shape}')
    print(f'Replay label shape : {replay_lab.shape}')
    print(f'Replay attr shape : {replay_attr.shape}')

    replay_size_comparison = {
        "task_no": task_no
    }
    for i in range(seen_classes * task_no):
        replay_size_comparison[f"{i}"] = replay_feat[(replay_lab == i).squeeze(1)].shape[0]

    return replay_feat, replay_lab, replay_attr, avg_feature


def main(opt):
    data = dataloader(opt)
    discriminator = Discriminator(data.feature_size, data.att_size).cuda()  # Create discriminator
    generator = Generator(data.feature_size, data.att_size).cuda()  # Create generator


    replay_feat = None
    replay_lab = None
    replay_attr = None
    avg_feature = None


    iter_task = opt.num_tasks


    for task_no in range(1, iter_task):
        replay_feat, replay_lab, replay_attr, avg_feature = train(task_no, opt.Neighbors, discriminator, generator,
                                                                  data, opt.seen_classes, opt.novel_classes,
                                                                  replay_feat, replay_lab, replay_attr,
                                                                  feature_size=opt.feature_size,
                                                                  attribute_size=opt.attribute_size,
                                                                  dlr=opt.d_lr,
                                                                  glr=opt.g_lr, batch_size=opt.batch_size,
                                                                  unsn_batch_size=opt.unsn_batch_size,
                                                                  epochs=opt.epochs, lambda1=opt.t, alpha=opt.alpha,
                                                                  avg_feature=avg_feature, all_classes=opt.all_classes,
                                                                  num_tasks=opt.num_tasks)
        if opt.validation:  # Used for hyper-parameter validation quarter of total tasks used for validation
            end = opt.num_tasks // 4
            if task_no == end:
                break


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('please use GPU!')
        exit()
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AWA1', type=str, help="The name of the dataset")
    parser.add_argument('--run_name', default='testjob', type=str, help="The name of the experiment")
    parser.add_argument('--seen_classes', default=8, type=int,
                        help="Number of class seen in each task (automatic if dataset specified) ")
    parser.add_argument('--novel_classes', default=2, type=int,
                        help="Number of class novel/unseen in each task (automatic if dataset specified) ")
    parser.add_argument('--num_tasks', default=5, type=int, help="Number of continual learning tasks")
    parser.add_argument('--all_classes', default=50, type=int,
                        help="Total number of classes in the dataset (automatic if dataset specified) ")
    parser.add_argument('--feature_size', default=2048, type=int,
                        help="Size of the feature vector (automatic if dataset specified) ")
    parser.add_argument('--attribute_size', default=85, type=int,
                        help="Size of the attribute vector (automatic if dataset specified) ")
    # parser.add_argument('--no_of_replay', default=300, type=int,
    #                     help="Number of samples to be used for replay / buffer size")
    parser.add_argument('--data_dir', default=f'{cwd}/data', type=str,
                        help="Path to the data directory , automatically set for current directory/data if not specified")
    parser.add_argument('--d_lr', type=float, default=0.005, help="Discriminator learning rate")
    parser.add_argument('--g_lr', type=float, default=0.005, help="Generator learning rate")
    parser.add_argument('--dic_lr', type=float, default=0.05, help="Dictionary learning rate")
    parser.add_argument('--t', type=float, default=10.0,
                        help="Coefficient for the cosine similarity loss function (lambda1)")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Coefficient for the classification losses (alpha)")
    parser.add_argument('--Neighbors', type=int, default=3,
                        help="Number of neighbors to be used in semantic similarity measure")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for seen classes")
    parser.add_argument('--unsn_batch_size', type=int, default=16, help="Batch size for unseen classes")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--dataroot', default=cwd + '/data', help='path to dataset')
    parser.add_argument('--image_embedding', default='res101', help="Type of features used")
    parser.add_argument('--class_embedding', default='att',
                        help="Specifies the type of embeddings used for semantic information")
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--preprocessing', action='store_true', default=False,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--attribute_generation_method', default='interpolation',
                        choices=["interpolation", "learnable", "none"], type=str,
                        help="Specify the type of attribute generation used")
    parser.add_argument('--creative_weight', type=float, default=0.1, help='weight for creative loss')
    parser.add_argument("--grw_creative_weight", type=float, default=0.1, help="weight for creative loss in grw")
    parser.add_argument("--corr_weight", type=float, default=20, help="Weight of the correlation loss for lsr loss")
    parser.add_argument("--rw_steps", type=int, default=3, help="Number of ranom walk steps")
    parser.add_argument("--seed", type=int, default=2222, help="Random seed")
    parser.add_argument("--buffer_size", type=int, default=5000, help="Memory replay buffer size")
    parser.add_argument("--decay_coef", type=float, default=0.7, help="Decay coefficient for ranom walk")
    parser.add_argument("--load_best_hp", action='store_true', default=True, help="Load the best hyper parameters")
    opt, _ = parser.parse_known_args()
    print("Script has started")
    print(_)

    if opt.dataset in ['AWA1', 'AWA2']:
        opt.seen_classes = 10
        opt.novel_classes = 10
        opt.num_tasks = 5
        opt.all_classes = 50
        opt.attribute_size = 85
        if opt.load_best_hp:
            if opt.dataset == "AWA1":
                if opt.attribute_generation_method == 'interpolation':
                    opt.creative_weight = 10.0
                    opt.grw_creative_weight = 0.5
                else:
                    opt.creative_weight = 1
                    opt.grw_creative_weight = 2
            else:
                if opt.attribute_generation_method == 'interpolation':
                    opt.creative_weight = 1.0
                    opt.grw_creative_weight = 1.0
                else:
                    opt.creative_weight = 10
                    opt.grw_creative_weight = 5
            opt.rw_steps = 3
            opt.decay_coef = 0.7
    if opt.dataset == 'SUN':
        opt.seen_classes = 47
        opt.novel_classes = 47
        opt.num_tasks = 15
        opt.all_classes = 717
        opt.attribute_size = 102
        if opt.load_best_hp:
            if opt.attribute_generation_method == 'interpolation':
                opt.creative_weight = 1.0
                opt.grw_creative_weight = 5.0
            else:
                opt.creative_weight = 1.0
                opt.grw_creative_weight = 1.0
            opt.rw_steps = 5
            opt.decay_coef = 0.7
    if opt.dataset == 'CUB':
        opt.seen_classes = 10
        opt.novel_classes = 10
        opt.num_tasks = 20
        opt.all_classes = 200
        opt.attribute_size = 312
        if opt.load_best_hp:
            if opt.attribute_generation_method == 'interpolation':
                opt.creative_weight = 1
                opt.grw_creative_weight = 2
            else:
                opt.creative_weight = 1
                opt.grw_creative_weight = 2
            opt.rw_steps = 5
            opt.decay_coef = 0.7

    # if opt.deviation_loss == "grawd" or opt.deviation_loss == "both":
    import yaml
    from easydict import EasyDict

    with open("config.yaml", "r") as f:
        rw_config = EasyDict(yaml.load(f, Loader=yaml.Loader))
    rw_config.rw_params.num_steps = opt.rw_steps
    rw_config.loss_weights.gen.creative = opt.grw_creative_weight
    rw_config.rw_params.decay_coef = opt.decay_coef
    # wandb.config.update(rw_config)
    write_config = vars(opt)
    # if opt.deviation_loss == "grawd" or opt.deviation_loss == "both":
    write_config['rw_config'] = vars(rw_config)
    acwd = os.getcwd()
    checkpoint_dir = os.path.join(acwd, 'checkpoints', opt.run_name)
    logs_dir = os.path.join(acwd, 'logs', opt.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    with open(f"logs/{opt.run_name}/config.json", "w") as f:
        json.dump(write_config, f)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    print(opt)
    main(opt)
