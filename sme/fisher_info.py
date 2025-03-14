"""
Module for computing Fisher information and standard errors.
Computes the Fisher information by averaging the outer products of gradients over the dataloader.
"""
import torch

def calculate_fisher_information(model, dataloader, device):
    """
    Calculate the Fisher information matrix by accumulating gradients.
    
    Args:
        model: The SMEModel containing the encoder, emulator, and loss function.
        dataloader: A DataLoader providing batches of input data as tuples (phi, Y).
        device: The device used for computation.

    Returns:
        A tensor representing the Fisher information matrix.
    """
    model.eval()
    fisher_information = None

    for batch in dataloader:
        # Expecting batch as a tuple (phi, Y)
        phi_batch, Y_batch = batch
        phi_batch = phi_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Zero out gradients for encoder and emulator separately.
        model.encoder.zero_grad()
        model.emulator.zero_grad()

        # Forward pass through encoder and emulator.
        f_output = model.encoder(Y_batch)
        g_output = model.emulator(phi_batch)

        # Compute the loss used in training.
        loss = model.loss_fn(f=f_output, g=g_output)

        # Use negative loss as a proxy for log-likelihood.
        log_likelihood = -loss
        log_likelihood.backward()

        # Collect gradients from both encoder and emulator parameters.
        gradients = []
        for param in model.encoder.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        for param in model.emulator.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))

        if gradients:
            gradients = torch.cat(gradients)
            current_outer = torch.outer(gradients, gradients)
            if fisher_information is None:
                fisher_information = current_outer
            else:
                fisher_information += current_outer

    if fisher_information is not None and len(dataloader) > 0:
        fisher_information /= len(dataloader)
    return fisher_information

def calculate_standard_errors(fisher_information):
    """
    Calculate standard errors from the Fisher information matrix.

    Args:
        fisher_information: Tensor representing the Fisher information matrix.

    Returns:
        Tensor of standard errors (sqrt of the diagonal of the inverted Fisher matrix).
    """
    # Add a small regularization to the diagonal to avoid singular matrix issues.
    jitter = 1e-6
    diag_jitter = jitter * torch.eye(fisher_information.size(0), device=fisher_information.device)
    fisher_information_regularized = fisher_information + diag_jitter

    fisher_information_inv = torch.inverse(fisher_information_regularized)
    standard_errors = torch.sqrt(torch.diag(fisher_information_inv))
    return standard_errors