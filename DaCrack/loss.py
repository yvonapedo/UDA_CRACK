import torch
import torch.nn.functional as F
import torch.nn as nn


def domain_discriminator_loss(D_Fs, D_Ft):
    loss_fn = nn.BCEWithLogitsLoss()


    # loss1 = loss_fn(logits1, target1)  # Compute the first loss
    # loss2 = loss_fn(logits2, target2)  # Compute the second loss

    # Now you can add the computed losses
    # total_loss = loss1 + loss2  # Correct
    """
    Calculate the domain discriminator loss.

    Parameters:
        D_Fs (torch.Tensor): Output of the discriminator for source domain features (Fs).
        D_Ft (torch.Tensor): Output of the discriminator for target domain features (Ft).

    Returns:
        torch.Tensor: The calculated loss.
    """
    # Domain labels: source domain = 1, target domain = 0
    source_labels = torch.ones_like( D_Fs)  # Label for source domain
    target_labels = torch.zeros_like(D_Ft)  # Label for target domain
    # print(D_Ft)
    # print(D_Fs)
    # Binary Cross-Entropy Loss for source and target domains
    # loss_source = F.binary_cross_entropy(D_Fs, source_labels)
    loss_source = loss_fn(D_Fs, source_labels)
    loss_target = loss_fn(D_Ft, target_labels)

    target_labels = torch.ones_like(D_Ft)  # Pretend the target domain is source domain (1)
    # print(target_labels)
    # Binary Cross-Entropy Loss for fooling the discriminator
    loss_ada = loss_fn(D_Ft, target_labels)

    # Total loss is the sum of the source and target domain losses
    loss = loss_source + loss_target

    return loss, loss_ada


def adaptation_loss(D_E_Xt):
    """
    Calculate the adaptation loss for the encoder to fool the domain discriminator.

    Parameters:
        D_E_Xt (torch.Tensor): Output of the discriminator for target domain features processed by the encoder (E(Xt)).

    Returns:
        torch.Tensor: The calculated adaptation loss.
    """
    # The goal is to fool the discriminator, so we want the target features to be classified as the source domain (label = 1)
    print(D_E_Xt)
    target_labels = torch.ones_like(D_E_Xt)  # Pretend the target domain is source domain (1)
    # print(target_labels)
    # Binary Cross-Entropy Loss for fooling the discriminator
    loss = F.binary_cross_entropy(D_E_Xt, target_labels)

    return loss
