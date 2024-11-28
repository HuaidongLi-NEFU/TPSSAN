import numpy as np
import torch
import torch.nn.functional as F



def augmentation_torch(volume, aug_factor):
    # volume is numpy array of shape (C, D, H, W)
    noise = torch.clip(torch.randn(*volume.shape) * 0.1, -0.2, 0.2).cuda()
    return volume + aug_factor * noise#.astype(np.float32)


def cut_module(inputs, unlabel_data, eval_net, num_augments, alpha, mixup_modes, aug_factor):
    # Get the batch size for inputs and unlabeled data
    num_inputs = len(inputs)
    num_unlabeled = len(unlabel_data)

    # Step 1: Data augmentation with random noise
    augmented_inputs = [(augmentation_torch(input[0], aug_factor), input[1]) for input in inputs]

    # Step 2: Expand unlabeled data and add noise
    expanded_unlabeled = unlabel_data.repeat(num_augments, 1, 1, 1, 1)  # [num_augments * batch_size, 1, D, H, W]
    expanded_unlabeled += torch.clamp(torch.randn_like(expanded_unlabeled) * 0.1, -0.2, 0.2)  # Adding noise

    # Step 3: Predict labels for the unlabeled data
    with torch.no_grad():
        pred_unlabeled, _, _, _ = eval_net(
            expanded_unlabeled)  # The model outputs 4 values, but we only use the first one
        pred_unlabeled = F.softmax(pred_unlabeled, dim=1)  # Apply softmax to get probabilities

    # Step 4: Generate pseudo-labels
    pseudo_labels = torch.zeros(unlabel_data.shape).repeat(1, num_augments, 1, 1, 1).cuda()  # Initialize pseudo-labels
    for i in range(num_augments):
        pseudo_labels += pred_unlabeled[i * num_unlabeled:(i + 1) * num_unlabeled]
    pseudo_labels /= num_augments  # Average the predictions to get the final pseudo-labels

    pseudo_labels = pseudo_labels.repeat(num_augments, 1, 1, 1, 1)
    pseudo_labels = torch.argmax(pseudo_labels, dim=1)  # Get the final pseudo-labels (class with max probability)

    # Step 5: Combine augmented data with pseudo-labels
    augmented_unlabeled = list(zip(expanded_unlabeled, pseudo_labels))

    # Step 6: Apply Mixup
    x_mixup_mode, u_mixup_mode = mixup_modes[0], mixup_modes[1]

    if x_mixup_mode == '_':
        inputs_prime = augmented_inputs
    else:
        raise ValueError('Invalid mixup_mode for inputs')

    if u_mixup_mode == 'x':
        idxs = np.random.permutation(range(num_unlabeled * num_augments)) % num_inputs
        unlabeled_prime = [cutmix(augmented_unlabeled[i], augmented_inputs[idxs[i]], alpha) for i in
                           range(num_unlabeled * num_augments)]
    else:
        raise ValueError('Invalid mixup_mode for unlabeled data')

    return inputs_prime, unlabeled_prime, pseudo_labels


def cutmix(input1, input2, alpha):
    x1, p1 = input1
    x2, p2 = input2

    # Get the image size (C, D, H, W)
    _, depth, height, width = x1.shape

    # Generate the cut ratio based on a Beta distribution
    lambda_val = np.random.beta(alpha, alpha)
    lambda_val = max(lambda_val, 1 - lambda_val)  # Ensure lambda is at least 0.5

    # Randomly select the starting point for the cut
    rand_h = np.random.randint(0, height)
    rand_w = np.random.randint(0, width)

    # Calculate the size of the cut area
    cut_h = int(height * np.sqrt(1 - lambda_val))
    cut_w = int(width * np.sqrt(1 - lambda_val))

    # Create a mask for the cut area
    mask = np.zeros((height, width))
    mask[rand_h:rand_h + cut_h, rand_w:rand_w + cut_w] = 1

    # Swap the selected cut area between the two images
    x1[:, rand_h:rand_h + cut_h, rand_w:rand_w + cut_w] = x2[:, rand_h:rand_h + cut_h, rand_w:rand_w + cut_w]

    # Update the pseudo-labels based on the mixing
    p = lambda_val * p1 + (1 - lambda_val) * p2

    return (x1, p)