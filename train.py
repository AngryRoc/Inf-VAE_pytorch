import numpy as np

from utils.flags import *
import operator
import time
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.cascades_data import Cascades
from torch.utils.data import DataLoader
from models.models import InfVAESocial, InfVAECascades
from utils.preprocess import *
from utils.flags import *
from eval.eval_metrics import *
from torch import clone


def main(argv):
    # hyper-params
    # General params.
    dataset = FLAGS.dataset
    epochs = FLAGS.epochs
    pretrain_epochs = FLAGS.pretrain_epochs
    max_gradient_norm = FLAGS.max_gradient_norm
    cuda_device = FLAGS.cuda_device

    test_freq = FLAGS.test_freq
    val_freq = FLAGS.val_freq
    early_stopping = FLAGS.early_stopping

    batch_queue_threads = FLAGS.batch_queue_threads

    graph_AE = FLAGS.graph_AE
    use_feats = FLAGS.use_feats

    k_list = [10, 50, 100]
    # Model Hyper-parameters.
    lambda_s = FLAGS.lambda_s
    lambda_r = FLAGS.lambda_r
    latent_dim = FLAGS.latent_dim
    pos_weight = FLAGS.pos_weight

    # Evaluation parameters.
    max_seq_length = FLAGS.max_seq_length
    test_min_percent = FLAGS.test_min_percent
    test_max_percent = FLAGS.test_max_percent

    # Co-Attention model parameters.
    lambda_a = FLAGS.lambda_a
    cascade_lr = FLAGS.cascade_lr
    cascade_batch_size = FLAGS.cascade_batch_size

    # VAE model parameters.
    hidden1_dim = FLAGS.hidden1_dim
    hidden2_dim = FLAGS.hidden2_dim
    hidden3_dim = FLAGS.hidden3_dim
    vae_dropout_rate = FLAGS.vae_dropout_rate
    vae_loss_function = FLAGS.vae_loss_function
    vae_batch_size = FLAGS.vae_batch_size
    vae_lr = FLAGS.vae_lr
    vae_weight_decay = FLAGS.vae_weight_decay
    vae_pos_weight = FLAGS.vae_pos_weight

    score = float('-inf')
    scores_metrics = None

    device = torch.device("cuda:" + cuda_device if torch.cuda.is_available() else "cpu")
    print(device)

    # load data
    A, A_norm, A_label = load_graph(dataset)
    A = A.to(device)
    A_norm = A_norm.to(device)
    A_label = A_label.to(device)
    features = load_feats(dataset).to(device)
    train_data = Cascades(dataset, max_seq_length, mode='train')
    val_data = Cascades(dataset, max_seq_length, mode='val')
    test_data = Cascades(dataset, max_seq_length, mode='test')
    num_nodes = A_norm.shape[0]
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)
    norm = A.shape[0] * A.shape[0] / float(vae_pos_weight * A.sum() + (A.shape[0] * A.shape[0]) - A.sum())

    VGAE = InfVAESocial(features.shape[1], hidden1_dim, hidden2_dim, hidden3_dim, latent_dim, device,
                        vae_dropout_rate).to(
        device)
    CoAtt = InfVAECascades(num_nodes + 1, latent_dim, max_seq_length, device).to(device)

    optimizer_VGAE = optim.Adam(VGAE.parameters(), lr=vae_lr, weight_decay=vae_weight_decay)
    optimizer_CoAtt = optim.Adam(CoAtt.parameters(), lr=cascade_lr)
    z_vae_embeds = torch.zeros([num_nodes + 1, latent_dim]).to(device)
    print("======VAE Pre-train=======")
    # Step 0: Pre-training using simple VAE on social network.
    for epoch in range(pretrain_epochs):
        # Training step
        VGAE.train()
        optimizer_VGAE.zero_grad()
        A_pred = VGAE(A_norm, features)
        likelihood_loss = -norm * torch.mean(
            vae_pos_weight * A_label.view(-1)
            * torch.log(
                torch.maximum(A_pred.view(-1),
                              torch.ones((A_pred.shape[0], A_pred.shape[1])).view(-1).to(device) * (1e-10)))
            + (1 - A_label.view(-1))
            * torch.log(
                torch.maximum((1 - A_pred.view(-1)),
                              torch.ones((A_pred.shape[0], A_pred.shape[1])).view(-1).to(device) * (1e-10)))
        )

        kl_divergence = 0.5 * torch.mean(
            torch.sum(torch.square(VGAE.mean) + torch.exp(VGAE.logstd) - VGAE.logstd - 1, 1))

        social_loss = likelihood_loss + kl_divergence
        social_loss.backward()
        optimizer_VGAE.step()
        epoch_loss = social_loss.item()
        print("Mean VAE loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))
    print("Pre-training completed")
    # Initial run to get embeddings.
    print("Initial run to get embeddings")
    for i in range(VGAE.mean.shape[0]):
        z_vae_embeds[i] = VGAE.mean[i].detach()

    sender_embeds = z_vae_embeds.clone().to(device)
    receiver_embeds = z_vae_embeds.clone().to(device)

    val_loss_all = []
    for epoch in range(epochs):

        # Train
        # Step 1: VAE on Social Network.
        VGAE.train()
        optimizer_VGAE.zero_grad()
        A_pred = VGAE(A_norm, features)
        likelihood_loss = -norm * torch.mean(
            vae_pos_weight * A_label.view(-1)
            * torch.log(
                torch.maximum(A_pred.view(-1),
                              torch.ones((A_pred.shape[0], A_pred.shape[1])).view(-1).to(device) * (1e-10)))
            + (1 - A_label.view(-1))
            * torch.log(
                torch.maximum((1 - A_pred.view(-1)),
                              torch.ones((A_pred.shape[0], A_pred.shape[1])).view(-1).to(device) * (1e-10)))
        )

        kl_divergence = 0.5 * torch.mean(
            torch.sum(torch.square(VGAE.mean) + torch.exp(VGAE.logstd) - VGAE.logstd - 1, 1))

        sender_loss = 0.5 * lambda_s * torch.mean(
            torch.sum(
                torch.square(sender_embeds - torch.cat((VGAE.mean, torch.zeros(1, VGAE.mean.shape[1]).to(device)))), 1
            ))

        receiver_loss = 0.5 * lambda_r * torch.mean(
            torch.sum(
                torch.square(receiver_embeds - torch.cat((VGAE.mean, torch.zeros(1, VGAE.mean.shape[1]).to(device)))), 1
            ))
        social_loss = likelihood_loss + kl_divergence + sender_loss + receiver_loss
        social_loss.backward()
        optimizer_VGAE.step()
        epoch_loss = social_loss.item()
        print("Mean VAE loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))
        for i in range(VGAE.mean.shape[0]):
            z_vae_embeds[i] = VGAE.mean[i].detach()

        # Step 2: Diffusion Cascades
        CoAtt.train()
        losses = []
        for i, (examples, lengths, targets, masks, examples_times, targets_times) in enumerate(train_loader, 0):
            examples = examples.to(device)
            masks = masks.to(device)
            outputs = CoAtt(examples, masks).to(device)
            labels_k_hot = torch.max(one_hot(targets, num_nodes + 1), dim=1)
            cascade_loss = torch.mean(pos_weight * labels_k_hot.values.to(device).view(-1) * (-torch.log(
                torch.maximum(torch.sigmoid(outputs).view(-1),
                              torch.ones((outputs.shape[0], outputs.shape[1])).view(-1).to(device) * (1e-10)))) + (
                                                  1 - labels_k_hot.values.to(device).view(-1)) * (-torch.log(
                torch.maximum((1 - torch.sigmoid(outputs).view(-1)),
                              torch.ones((outputs.shape[0], outputs.shape[1])).view(-1).to(device) * (
                                  1e-10))))) + lambda_a * torch.sum(CoAtt.co_attn_wts ** 2) / 2

            sender_loss = 0.5 * lambda_s * torch.mean(
                torch.sum(torch.square(CoAtt.sender_embeddings - z_vae_embeds), 1
                          ))

            receiver_loss = 0.5 * lambda_r * torch.mean(
                torch.sum(torch.square(CoAtt.receiver_embeddings - z_vae_embeds), 1
                          ))
            diffusion_loss = cascade_loss + sender_loss + receiver_loss
            losses.append(diffusion_loss.item())
            diffusion_loss.backward()
            optimizer_CoAtt.step()
        sender_embeds = CoAtt.sender_embeddings.detach()
        receiver_embeds = CoAtt.receiver_embeddings.detach()
        epoch_loss = np.mean(losses)
        print("Mean Attention loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))

        # Testing
        if epoch % test_freq == 0:
            VGAE.eval()
            CoAtt.eval()
            VGAE(A_norm, features)
            for i in range(VGAE.mean.shape[0]):
                z_vae_embeds[i] = VGAE.mean[i].detach()

            total_samples = 0
            num_eval_k = len(k_list)
            avg_map_scores, avg_recall_scores = [0.] * num_eval_k, [0.] * num_eval_k
            for i, (examples, lengths, targets, masks, examples_times, targets_times) in enumerate(test_loader, 0):
                examples = examples.to(device)
                masks = masks.to(device)
                targets = targets.to(device)
                outputs = CoAtt(examples, masks)
                top_k = torch.topk(outputs, 200, dim=1).indices
                top_k_filter = remove_seeds(top_k.detach().cpu().numpy(), examples.cpu().numpy())
                masks = get_masks(top_k_filter, examples.cpu())
                relevance_scores_all = get_relevance_scores(top_k_filter, targets.cpu().numpy())
                m = torch.sum(torch.max(one_hot(targets.cpu(), num_nodes + 1), dim=1).values, dim=-1).numpy()
                relevance_scores = masked_select(relevance_scores_all, masks)

                recall_scores = [mean_recall_at_k(relevance_scores, k, m) for k in k_list]
                map_scores = [MAP(relevance_scores, k, m) for k in k_list]
                num_samples = relevance_scores.shape[0]
                avg_map_scores = list(
                    map(operator.add, map(operator.mul, map_scores,
                                          [num_samples] * num_eval_k), avg_map_scores))
                avg_recall_scores = list(map(operator.add, map(operator.mul, recall_scores,
                                                               [num_samples] * num_eval_k), avg_recall_scores))
                total_samples += num_samples
            avg_map_scores = list(map(operator.truediv, avg_map_scores, [total_samples] * num_eval_k))
            avg_recall_scores = list(map(operator.truediv, avg_recall_scores, [total_samples] * num_eval_k))
            metrics = dict()
            for k in range(0, num_eval_k):
                K = k_list[k]
                metrics["MAP@%d" % K] = avg_map_scores[k]
                metrics["Recall@%d" % K] = avg_recall_scores[k]
            if avg_map_scores[0] > score:
                score = avg_map_scores[0]
                scores_metrics = metrics

        # Validation
        losses = []
        if epoch % val_freq == 0:
            VGAE.eval()
            CoAtt.eval()
            VGAE(A_norm, features)
            for i in range(VGAE.mean.shape[0]):
                z_vae_embeds[i] = VGAE.mean[i].detach()
            for i, (examples, lengths, targets, masks, examples_times, targets_times) in enumerate(val_loader, 0):
                examples = examples.to(device)
                masks = masks.to(device)
                outputs = CoAtt(examples, masks)

                labels_k_hot = torch.max(one_hot(targets, num_nodes + 1), dim=1)
                cascade_loss = torch.mean(pos_weight * labels_k_hot.values.to(device).view(-1) * (-torch.log(
                    torch.maximum(torch.sigmoid(outputs).view(-1),
                                  torch.ones((outputs.shape[0], outputs.shape[1])).view(-1).to(device) * (1e-10)))) + (
                                                  1 - labels_k_hot.values.to(device).view(-1)) * (-torch.log(
                    torch.maximum((1 - torch.sigmoid(outputs).view(-1)),
                                  torch.ones((outputs.shape[0], outputs.shape[1])).view(-1).to(device) * (
                                      1e-10))))) + lambda_a * torch.sum(CoAtt.co_attn_wts ** 2) / 2

                sender_loss = 0.5 * lambda_s * torch.mean(
                    torch.sum(torch.square(CoAtt.sender_embeddings - z_vae_embeds), 1
                              ))

                receiver_loss = 0.5 * lambda_r * torch.mean(
                    torch.sum(torch.square(CoAtt.receiver_embeddings - z_vae_embeds), 1
                              ))
                diffusion_loss = cascade_loss + sender_loss + receiver_loss
                losses.append(diffusion_loss.item())
            epoch_loss = np.mean(losses)

            print("Validation Attention loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))
            if len(val_loss_all) >= early_stopping and val_loss_all[-1] > np.mean(
                    val_loss_all[-(early_stopping + 1):-1]):
                print("Early stopping at epoch: %04d" % (epoch + 1))
                break
    pprint(scores_metrics)


if __name__ == '__main__':
    app.run(main)
