import json
import os
from omegaconf import OmegaConf
import statistics
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable
from args import get_args
from utils import compute_D_acc, seed_everything
from model import Discriminator, Generator

from data import DATA_LOADER as dataloader
from rw_loss import compute_rw_imitative_loss, compute_rw_real_loss, compute_rw_creative_loss

cwd = os.path.dirname(os.getcwd())

seen_acc_history = []
unseen_acc_history = []
harmonic_mean_history = []
best_seen_acc_history = []
best_unseen_acc_history = []
best_harmonic_mean_history = []


class Train_Dataloader:
    def __init__(self, train_feat_seen, train_label_seen):
        self.data = {'train_': train_feat_seen, 'train_label': train_label_seen}

    def get_loader(self, opt='train_', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt + 'label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=True)
        return data_loader


class Test_Dataloader:
    def __init__(self, test_attr, test_seen_f, test_seen_l, test_unseen_f, test_unseen_l):

        self.data = {'test_seen': test_seen_f, 'test_seenlabel': test_seen_l,
                     'whole_attributes': test_attr,
                     'test_unseen': test_unseen_f, 'test_unseenlabel': test_unseen_l,
                     'seen_label': np.unique(test_seen_l),
                     'unseen_label': np.unique(test_unseen_l)}

    def get_loader(self, opt='test_seen', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt + 'label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size)
        return data_loader


def train(task_id, Neighbors, discriminator, generator, data, seen_classes, novel_classes, replay_feature, replay_label,
          replay_attribute, feature_size=2048, dlr=0.005, glr=0.005, batch_size=64,
           epochs=50, lambda1=10.0, alpha=1.0, avg_feature= None , all_classes=None, num_tasks=None):
    """
    ####### The training function for each task
    :param task_no: current task number
    :param Neighbors: number of neighbors for lsr gan loss
    :param discriminator: discriminator model
    :param generator: generator model
    :param data: data loader
    :param seen_classes: number of seen classes in each tasks
    :param novel_classes: number of novel classes in each tasks
    :param replay_feature: features of replayed classes
    :param replay_label: labels of replayed classes
    :param replay_attribute: attributes of replayed classes
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
    
    # prepare attribute
    total_seen_classes = seen_classes * task_id
    cur_train_feature_seen, cur_train_label_seen, cur_train_att_seen = data.task_train_data(task_id, seen_classes, all_classes,
                                                                                 novel_classes, num_tasks, Neighbors)
    print(f'Current seen label {sorted(list(set([i.item() for i in cur_train_label_seen])))},')
    
    semantic_relation_seen = torch.tensor(data.idx_mat).cuda()
    semantic_values_seen = torch.tensor(data.semantic_similarity_seen).cuda()
    train_label_seen_tensor = torch.tensor(cur_train_label_seen.clone().detach() - total_seen_classes + seen_classes).squeeze()
    # Prepare tensors for accumulating features and class counts
    accumulated_features = torch.zeros((seen_classes, feature_size))
    class_count = torch.zeros(seen_classes)
    # Use scatter_add to accumulate features for each class
    accumulated_features.index_add_(0, train_label_seen_tensor, cur_train_feature_seen)
    # Count occurrences of each class
    class_count.index_add_(0, train_label_seen_tensor, torch.ones_like(train_label_seen_tensor, dtype=torch.float))
    # Avoid division by zero for classes with no samples
    class_count[class_count == 0] = 1
    # Calculate the average features
    cur_avg_feature = accumulated_features / class_count.unsqueeze(1)
    
    if task_id == 1:
        avg_feature = cur_avg_feature.cuda()
        train_feat_seen = cur_train_feature_seen
        train_label_seen = cur_train_label_seen
    else:
        avg_feature = torch.cat([avg_feature, cur_avg_feature.cuda()],dim=0) # type: ignore
        train_feat_seen = torch.cat((cur_train_feature_seen, replay_feature))
        train_label_seen = torch.cat((cur_train_label_seen, replay_label))
    


    # Initialize loaders and optimizers
    train_loader = Train_Dataloader(train_feat_seen, train_label_seen)
    train_dataloader = train_loader.get_loader('train_', batch_size)
    all_attribute = data.attribute_mapping(seen_classes, novel_classes, task_id).cuda()
    seen_attr = all_attribute[0:total_seen_classes, :]

    
    D_optimizer = optim.Adam(discriminator.parameters(), lr=dlr, weight_decay=0.00001)
    G_optimizer = optim.Adam(generator.parameters(), lr=glr, weight_decay=0.00001)

    # Initialize learnable dictionary for attribute generation
    # Attributes are initialized with the heuristic but let to change as the learning goes on
    if opt.attribute_generation_method == "learnable":
        learnable_attr = nn.Embedding(opt.all_classes, data.att_size, max_norm=torch.norm(seen_attr, dim=1).max()).cuda()
        random_idx = torch.randint(0, seen_attr.shape[0], [opt.all_classes, 1], device="cuda")
        random_idx_2 = torch.randint(0, seen_attr.shape[0], [opt.all_classes, 1], device="cuda")
        initialize_alpha = (torch.rand(opt.all_classes) * (.8 - .2) + .2).cuda()
        hallucinated_attributes = seen_attr.squeeze(1)[random_idx_2.squeeze(1)].T * (initialize_alpha) + \
                                  seen_attr[random_idx.squeeze(1)].squeeze(1).T * (1 - initialize_alpha)
        learnable_attr.weight.data = hallucinated_attributes.T
        att_optimizer = optim.Adam(learnable_attr.parameters(), lr=opt.dic_lr, weight_decay=0.00001)
        
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
            "g_seen_loss": [],
            "creativity_loss": [],
            "g_rw_creativity_loss": [],
            "g_rw_imitative_loss": [],

        }

        # Mini batch loops
        for feature, label in train_dataloader:
            if feature.shape[0] == 1:
                continue
            feature, label = feature.cuda(), label.cuda()
            feature_norm = F.normalize(feature, p=2, dim=-1, eps=1e-12)
            
            
            
                
            # prepare attribute
            batch_repeated_seen_attr = seen_attr.unsqueeze(0).repeat([feature.size(0), 1, 1])
            label_ordered_attr = all_attribute[label]
            
            
            
            

            #---------Discriminator Update---------
            
            D_optimizer.zero_grad()
            
                
            # Embed seen attributes to the visual feature space
            label_ordered_attr_proj = discriminator(label_ordered_attr)
            label_ordered_attr_proj_norm = F.normalize(label_ordered_attr_proj, p=2, dim=-1, eps=1e-12)
            
            # Generate fake seen features using generator
            noise = torch.normal(0, 1, size=label_ordered_attr.shape).cuda()
            pseudo_seen_features = generator(noise, label_ordered_attr)
            pseudo_seen_features_norm = F.normalize(pseudo_seen_features, p=2, dim=-1, eps=1e-12)


            # Compute discriminator real-fake loss
            real_cosine_similarity = lambda1 * F.cosine_similarity(label_ordered_attr_proj_norm, feature_norm, dim=-1)
            pseudo_cosine_similarity = lambda1 * F.cosine_similarity(label_ordered_attr_proj_norm, pseudo_seen_features_norm, dim=-1)
            real_cosine_similarity = torch.mean(real_cosine_similarity)
            pseudo_cosine_similarity = torch.mean(pseudo_cosine_similarity)

            # Discriminator regularization loss
            seen_attr_proj = discriminator(seen_attr)
            mse_d = mse_loss(avg_feature, seen_attr_proj)

            # Compute discriminator classification loss
            batch_repeated_seen_attr_proj = discriminator(batch_repeated_seen_attr)
            batch_repeated_seen_attr_proj_norm = F.normalize(batch_repeated_seen_attr_proj, p=2, dim=-1, eps=1e-12)
            class_repeated_feature_norm = feature_norm.unsqueeze(1).repeat([1, total_seen_classes, 1])
            real_cosine_sim = lambda1 * F.cosine_similarity(batch_repeated_seen_attr_proj_norm, class_repeated_feature_norm, dim=-1)
            classification_losses = entory_loss(real_cosine_sim, label.squeeze())

            # Discriminator optimization step
            d_loss = - torch.log(real_cosine_similarity) + torch.log(
                pseudo_cosine_similarity) + alpha * classification_losses + mse_d
            d_loss.backward(retain_graph=True)
            losses["d_loss"].append(d_loss.item())
            D_optimizer.step()

            # ---------Generator Update---------
            G_optimizer.zero_grad()
            if opt.attribute_generation_method == "learnable":
                att_optimizer.zero_grad()
                
            # Embed seen attributes to the visual feature space
            label_ordered_attr_proj = discriminator(label_ordered_attr)
            label_ordered_attr_proj_norm = F.normalize(label_ordered_attr_proj, p=2, dim=-1, eps=1e-12)
            # Generate fake seen features using generator
            noise = torch.normal(0, 1, size=label_ordered_attr.shape).cuda()
            pseudo_seen_features = generator(noise, label_ordered_attr)
            pseudo_seen_features_norm = F.normalize(pseudo_seen_features, p=2, dim=-1, eps=1e-12)

            # Compute discriminator real-fake loss
            pseudo_cosine_similarity = lambda1 * F.cosine_similarity(label_ordered_attr_proj_norm, pseudo_seen_features_norm, dim=-1)
            pseudo_cosine_similarity = torch.mean(pseudo_cosine_similarity)
                
            # Generator classification loss
            class_repeated_pseudo_seen_features_norm = pseudo_seen_features_norm.repeat([1, total_seen_classes, 1])
            batch_repeated_seen_attr_proj = discriminator(batch_repeated_seen_attr)
            batch_repeated_seen_attr_proj_norm = F.normalize(batch_repeated_seen_attr_proj, p=2, dim=-1, eps=1e-12)
            fake_cosine_sim = lambda1 * F.cosine_similarity(batch_repeated_seen_attr_proj_norm, class_repeated_pseudo_seen_features_norm, dim=-1)
            pseudo_classification_loss = entory_loss(fake_cosine_sim, label.squeeze())

            # Generator regularization loss
            classes = torch.arange(total_seen_classes).cuda()
            class_incremental_label_expanded = label.expand(-1, total_seen_classes)
            mask = class_incremental_label_expanded == classes.unsqueeze(0)

            # Compute means for each class
            class_means = torch.where(mask.unsqueeze(2), pseudo_seen_features.unsqueeze(1), torch.tensor(0.0).cuda()).sum(dim=0)
            sample_counts = mask.sum(dim=0).unsqueeze(1)
            class_means /= (sample_counts + 1e-8)
            

            # Euclidean loss
            euclidean_diff = class_means - avg_feature[classes]
            Euclidean_loss = euclidean_diff.pow(2).sum(dim=-1).sqrt().mean() / total_seen_classes
            
            # Normalize generated_mean and avg_feature for correlation loss
            generated_mean_norm = F.normalize(class_means, p=2, dim=-1, eps=1e-12)
            avg_features_norm = F.normalize(avg_feature[semantic_relation_seen[classes]], p=2, dim=-1, eps=1e-12)

            # Cosine similarity
            cos_sim = F.cosine_similarity(generated_mean_norm.unsqueeze(2), avg_features_norm, dim=-1)

            # Correlation loss
            lower_limits = semantic_values_seen[classes] - 0.01
            upper_limits = semantic_values_seen[classes] + 0.01
            corr_loss_terms = (torch.max(cos_sim - cos_sim, cos_sim - upper_limits))**2 + \
                            (torch.max(cos_sim - cos_sim, lower_limits - cos_sim))**2
            Correlation_loss = corr_loss_terms.mean() 
            
            
            # Generator seen loss optimization step
            g_loss_seen = - torch.log(pseudo_cosine_similarity) + alpha * pseudo_classification_loss + Euclidean_loss + Correlation_loss
            g_loss_seen.backward(retain_graph=True)
            losses["g_seen_loss"].append(g_loss_seen.item())

            # Generator inductive loss
            if opt.attribute_generation_method == "interpolation":
                hallucinate_1 = label_ordered_attr.squeeze(1)
                random_permutations = torch.randperm(label_ordered_attr.shape[0])
                hallucinate_1 = hallucinate_1[random_permutations]
                hallucinate_2 = label_ordered_attr.squeeze(1)
                random_permutations = torch.randperm(label_ordered_attr.shape[0])
                hallucinate_2 = hallucinate_2[random_permutations]
                creative_alpha = (torch.rand(len(label)) * (.8 - .2) + .2).cuda()
                hallucinated_attr = (
                        creative_alpha * hallucinate_1.T + (1 - creative_alpha) * hallucinate_2.T).T.unsqueeze(1)
            elif opt.attribute_generation_method == "learnable":
                random_idx = torch.randint(0, learnable_attr.weight.shape[0], [feature.shape[0], 1], device="cuda")
                hallucinated_attr = learnable_attr(random_idx)
            else:
                raise NotImplementedError
            
            


            # Cretivity loss
            noise = torch.normal(0, 1, size=hallucinated_attr.shape).cuda()
            creative_features = generator(noise, hallucinated_attr)
            hallucinated_projections_norm = F.normalize(discriminator(hallucinated_attr), p=2, dim=-1,eps=1e-12)
            creative_features_norm = F.normalize(creative_features, p=2, dim=-1, eps=1e-12)
            creative_features_repeated = creative_features_norm.repeat([1, total_seen_classes,1])
            creative_cosine_sim = lambda1 * F.cosine_similarity(batch_repeated_seen_attr_proj_norm, creative_features_repeated,dim=-1)
            G_fake_C = F.log_softmax(creative_cosine_sim, dim=-1)
            entropy_GX_fake = (G_fake_C / G_fake_C.data.size(1)).mean()
            loss_creative = - opt.creative_weight * entropy_GX_fake
            loss_creative_realistic_part = -torch.mean(lambda1 * F.cosine_similarity(hallucinated_projections_norm, creative_features,
                                              dim=-1))
            creativity_loss = (loss_creative + loss_creative_realistic_part)
            creativity_loss.backward(retain_graph=True)
            losses["creativity_loss"].append(creativity_loss.item())

            # Generative random walk loss on discriminator
            discr_rw_imitative_walker_loss, discr_rw_imitative_visit_loss = compute_rw_imitative_loss(opt,
                                                                                                      total_seen_classes,
                                                                                                      all_attribute,
                                                                                                      discriminator,
                                                                                                      generator)
            discr_rw_imitative_loss = discr_rw_imitative_walker_loss + opt.loss_weights.visit_loss * discr_rw_imitative_visit_loss
            discr_rw_imitative_loss = opt.loss_weights.discr.imitative * discr_rw_imitative_loss
            discr_rw_imitative_loss.backward(retain_graph=True)
            losses["d_imitative_loss"].append(discr_rw_imitative_loss.item())

            # Generative random walk loss on generator
            discr_rw_real_walker_loss, discr_rw_real_visit_loss = compute_rw_real_loss(opt,
                                                                                       total_seen_classes,
                                                                                       all_attribute, feature, label,
                                                                                       generator, discriminator)
            discr_rw_real_loss = discr_rw_real_walker_loss + opt.loss_weights.visit_loss * discr_rw_real_visit_loss
            discr_rw_real_loss = opt.loss_weights.discr.real * discr_rw_real_loss
            discr_rw_real_loss.backward(retain_graph=True)
            losses["d_rw_real_loss"].append(discr_rw_real_loss.item())

            gen_rw_imitative_walker_loss, gen_rw_imitative_visit_loss = compute_rw_imitative_loss(opt,
                                                                                                  total_seen_classes,
                                                                                                  all_attribute,
                                                                                                  discriminator,
                                                                                                  generator)
            gen_rw_imitative_loss = gen_rw_imitative_walker_loss + opt.loss_weights.visit_loss * gen_rw_imitative_visit_loss
            gen_rw_imitative_loss = opt.loss_weights.gen.imitative * gen_rw_imitative_loss
            losses["g_rw_imitative_loss"].append(gen_rw_imitative_loss.item())
            gen_rw_imitative_loss.backward(retain_graph=True)
            gen_rw_creative_walker_loss, gen_rw_creative_visit_loss = compute_rw_creative_loss(opt,
                                                                                               total_seen_classes,
                                                                                               all_attribute,
                                                                                               discriminator,
                                                                                               creative_features.squeeze(
                                                                                                   1), generator)
            gen_rw_creative_loss = gen_rw_creative_walker_loss + opt.loss_weights.visit_loss * gen_rw_creative_visit_loss

            gen_rw_creative_loss = opt.grw_creative_weight * gen_rw_creative_loss
            losses["g_rw_creativity_loss"].append(gen_rw_creative_loss.item())
            gen_rw_creative_loss.backward(retain_graph=True)

            G_optimizer.step()


            if opt.attribute_generation_method == "learnable":
                att_optimizer.step()
                
        if epoch == epochs - 1:
            test_seen_f, test_seen_l, test_seen_a, test_unseen_f, test_unseen_l, test_unseen_a = data.task_test_data_(
                task_id, seen_classes, all_classes, novel_classes, num_tasks)
            print(f'current unseen label {sorted(list(set([i.item() for i in test_unseen_l])))}')
            att_per_task_ = data.attribute_mapping(seen_classes, novel_classes, task_id).cuda()
            test_dataloader = Test_Dataloader(att_per_task_, test_seen_f, test_seen_l, test_unseen_f,
                                              test_unseen_l)
            D_seen_acc = compute_D_acc(discriminator, test_dataloader, seen_classes, novel_classes, task_id,
                                       batch_size=batch_size, opt1='gzsl', opt2='test_seen')
            D_unseen_acc = compute_D_acc(discriminator, test_dataloader, seen_classes, novel_classes, task_id,
                                         batch_size=batch_size, opt1='gzsl', opt2='test_unseen')
            if D_unseen_acc == 0 or D_seen_acc == 0:
                D_harmonic_mean = 0
            else:
                D_harmonic_mean = (2 * D_seen_acc * D_unseen_acc) / (D_seen_acc + D_unseen_acc)

            epoch_seen_accuracy_history.append(D_seen_acc)
            if task_id != opt.num_tasks:
                epoch_unseen_accuracy_history.append(D_unseen_acc)
                epoch_harmonic_accuracy_history.append(D_harmonic_mean)
            print(
                f'Best accuracy at task {task_id} at epoch {epoch}: unseen : {D_unseen_acc:.4f}, seen : {D_seen_acc:.4f}, H : {D_harmonic_mean:.4f}')

        loss_metrics = {
            "epoch": epoch,
            "task_no": task_id,
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
    if task_id == opt.num_tasks:
        for t in range(1, opt.num_tasks + 1):
            test_seen_f, test_seen_l, test_seen_a, test_unseen_f, test_unseen_l, test_unseen_a = data.task_test_data_(t,
                                                                                                                      seen_classes,
                                                                                                                      all_classes,
                                                                                                                      novel_classes,
                                                                                                                      num_tasks)
            att_per_task_ = data.attribute_mapping(seen_classes, novel_classes, t).cuda()
            test_dataloader = Test_Dataloader(att_per_task_, test_seen_f, test_seen_l, test_unseen_f,
                                              test_unseen_l)
            final_model_acc = compute_D_acc(discriminator, test_dataloader, seen_classes, novel_classes, t,
                                            batch_size=batch_size, opt1='gzsl', opt2='test_seen')
            final_model_acc_history.append(final_model_acc)
        final_model_acc_difference = np.array(final_model_acc_history) - np.array(seen_acc_history)
        forgetting_measure = np.mean(final_model_acc_difference[:-1])

    checkpoint = {
        "task_no": task_id,
        "discriminator": discriminator.state_dict(),
        "generator": generator.state_dict(),
        "optimizer_D": D_optimizer.state_dict(),
        "optimizer_G": G_optimizer.state_dict(),
    }
    if opt.attribute_generation_method == "learnable":
        checkpoint["learned_attributes"] = learnable_attr.state_dict()
    try:
        torch.save(checkpoint, f'checkpoints/{opt.run_name}/checkpoint_task_{task_id}.pth')
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
    size_for_this = opt.buffer_size // (seen_classes * task_id)
    replay_feature_buffer = []
    replay_label_buffer = []
    replay_attr_buffer = []

    if replay_feature is not None:
        for i in range(0, seen_classes * (task_id - 1)):
            mask = replay_label.squeeze(1) == i
            replay_feature_buffer.append(replay_feature[mask][:size_for_this])
            replay_label_buffer.append(replay_label[mask][:size_for_this])
            replay_attr_buffer.append(replay_attribute[mask][:size_for_this])

    for i in range(seen_classes * (task_id - 1), seen_classes * task_id):
        mask = cur_train_label_seen.squeeze(1) == i
        selected_seen_feature = cur_train_feature_seen[mask]
        selected_seen_label = cur_train_label_seen[mask]
        selected_seen_attr = cur_train_att_seen[mask]
        replay_feature_buffer.append(selected_seen_feature[:size_for_this])
        replay_label_buffer.append(selected_seen_label[:size_for_this])
        replay_attr_buffer.append(selected_seen_attr[:size_for_this])

    replay_feature = torch.cat(replay_feature_buffer, dim=0)
    replay_label = torch.cat(replay_label_buffer, dim=0)
    replay_attribute = torch.cat(replay_attr_buffer, dim=0)

    print(f'Replay data shape : {replay_feature.shape}')
    print(f'Replay label shape : {replay_label.shape}')
    print(f'Replay attr shape : {replay_attribute.shape}')

    replay_size_comparison = {
        "task_no": task_id
    }
    for i in range(seen_classes * task_id):
        replay_size_comparison[f"{i}"] = replay_feature[(replay_label == i).squeeze(1)].shape[0]

    return replay_feature, replay_label, replay_attribute, avg_feature


def main(opt):
    data = dataloader(opt)
    discriminator = Discriminator(data.feature_size, data.att_size).cuda()  # Create discriminator
    generator = Generator(data.feature_size, data.att_size).cuda()  # Create generator


    replay_feat = None
    replay_lab = None
    replay_attr = None
    avg_feature = None

    # Hyper-parameter validation with quarter of total tasks
    if opt.validation:
        iter_task = opt.num_tasks // 4
    else:
        iter_task = opt.num_tasks


    for task_id in range(1, iter_task):
        replay_feat, replay_lab, replay_attr, avg_feature = train(task_id, opt.Neighbors, discriminator, generator,
                                                                  data, opt.seen_classes, opt.novel_classes,
                                                                  replay_feat, replay_lab, replay_attr,
                                                                  feature_size=opt.feature_size,
                                                                  dlr=opt.d_lr,
                                                                  glr=opt.g_lr, batch_size=opt.batch_size,
                                                                  epochs=opt.epochs, lambda1=opt.t, alpha=opt.alpha,
                                                                  avg_feature=avg_feature, all_classes=opt.all_classes,
                                                                  num_tasks=opt.num_tasks)
          



if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('please use GPU!')
        exit()
    cwd = os.getcwd()
    opt = get_args(cwd)
    print("Script has started")
    print()
    
    seed_everything(opt.seed)


    # log
    checkpoint_dir = os.path.join(cwd, 'checkpoints', opt.run_name)
    logs_dir = os.path.join(cwd, 'logs', opt.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    with open(f"logs/{opt.run_name}/config.json", "w") as f:
        json.dump(OmegaConf.to_yaml(opt), f)
    if opt.wandb_log:
        wandb.init(project='Random_Walk_CGZSL', name=opt.run_name, config=vars(opt))
        
        
    print(opt)
    main(opt)
