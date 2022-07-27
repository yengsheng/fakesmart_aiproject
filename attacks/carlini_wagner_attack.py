from . import attack_common
import os
import numpy as np
import time
import torch
from torch import optim, Tensor
from torch.autograd import Variable
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import datetime

def l2_loss_func(x, w):
    return torch.dist(x,((torch.tanh(w) + 1)/2), p=2)

def carlini_wagner_attack(model, img: Tensor, ori_label, target=False, num_classes=43, target_label=-1, max_itr = 1000):
    binary_search_itr = 10
    c = 0.001
    c_upper_bound = 1e10
    c_lower_bound = 0
    lr = 0.01
    min_loss = 1e10
    best_attack_img = img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not target:
        target_label = ori_label
    if torch.cuda.is_available():
        identity_matrix = Variable(torch.from_numpy(np.eye(num_classes)[target_label]).cuda().float())
    else:
        identity_matrix = Variable(torch.from_numpy(np.eye(num_classes)[target_label]).float())
    
    # delta = 0.5(tanh(w) + 1 ) - x
    # Why multiple by 1-0.0001 no one knows
    w = (img*2).sub(1).mul(1 - 1e-5).atanh()
    modifier =torch.zeros_like(w, requires_grad=True).to(device).float()

    for b_step in range(binary_search_itr):
        optimizer = optim.Adam([modifier], lr=lr)
        found_success = False
        for i in range(max_itr):
            optimizer.zero_grad()

            #0.5*tanh(w+pert)+0.5
            adv_img = (torch.tanh(w + modifier) + 1) / 2
            l2_loss = l2_loss_func(adv_img, w) 
            new_pred = model(adv_img)
            new_loss = c*f6(new_pred, identity_matrix, target, device)
            total_loss = l2_loss + new_loss

            # Minimise loss
            total_loss.backward(retain_graph=True)
            optimizer.step()
            # if i % 200 == 0:
            #      print('Itr {}: total loss: {}, l2_loss:{}, new_loss:{}'
            #            .format(i, total_loss.item(), l2_loss.item(), new_loss.item()))
            
            # Could also add a confidence level to make the image further away from original
            pred_result = new_pred.argmax(1, keepdim=True).item()
            
            # Success in attack
            if min_loss > l2_loss and \
                ((target and pred_result==target_label) or \
                    (not target and pred_result!=ori_label)):
                    min_loss = l2_loss
                    best_attack_img = adv_img
                    found_success = True

        if found_success:
            c_upper_bound = min(c_upper_bound, c)
        else:
            c_lower_bound = max(c_lower_bound, c)
            if c_upper_bound > 1e9:
                c = c*10
        if c_upper_bound < 1e9:
            c = (c_lower_bound + c_upper_bound)/2

    return best_attack_img[0]

# objective function 6 as seen in the paper
def f6(pred, tlab, target, device, k=0):
    pred_max = torch.max(pred *tlab)
    other_pred =torch.max((1 - tlab) * pred)

    multiplier = 1 if target else -1
    return torch.max(multiplier * (other_pred - pred_max), torch.Tensor([-k]).to(device))
  


# Return the accuracy after the attack and the adversarial examples
def test_cw_attack(model, device, test_loader, num_tests, num_classes, target_label = -1):
    model.eval()

    accuracy_counter = 0
    
    # List of successful adversarial samples
    attack_sample_list = []

    for i, (image, labels) in enumerate(test_loader):
        if image is None:
          continue
        if i %20 == 0:
            print("images attacked:", i, datetime.datetime.now())

        # label = torch.unsqueeze(label, 0)
        _, label = torch.max(labels.data, 1)

        # Skip image if the target label is the same as the current label
        if target_label == label.item():
            accuracy_counter += 1
            continue

        # Send the data and label to the device
        image, label = image.to(device), label.to(device)

        image.requires_grad = True
        image.requires_grad = True

        # Get initial prediction before attack
        output = model(image)

        # Get the index of the max log-probability
        init_pred = output.max(1, keepdim = True)[1]
        
        # If the initial prediction is wrong, do not bother attacking, skip current image
        if init_pred.item() != label.item():
            continue

        # Attack
        attack_image = carlini_wagner_attack(model, image, label.item(), target=target_label!=-1, num_classes=num_classes, target_label=target_label)

        attack_image_unsqueezed = torch.unsqueeze(attack_image, 0)
        # Get new prediction after attack
        attack_output = model(attack_image_unsqueezed)

        # Get the index of the max log-probability
        _, attack_pred = torch.max(attack_output.data, 1)
        attack_pred = attack_pred[0]

        # Check if attck label different from init pred
        if attack_pred.item() == label.item():
            accuracy_counter += 1            
        else:
            # Save some attack images for display
            if len(attack_sample_list) < 10:
                adv_ex = attack_image.squeeze().detach().cpu().numpy()
                attack_sample_list.append((init_pred.item(), attack_pred.item(), adv_ex))
                print("predictions:", len(attack_sample_list), attack_output.data)
                # grid = make_grid(attack_image_unsqueezed.data.cpu(), normalize=True).permute(1,2,0).numpy()
                # plt.imshow(grid)
                # plt.savefig(os.path.join(attack_common.VISUALISATIONS_DIR, "attack_image" + str(int(time.time()))))
                # plt.show()
    final_accuracy = accuracy_counter/float(num_tests)
    
    # Display for progress
    print("C-W Attack: Test Accuracy = {}/{} = {}".format(accuracy_counter, \
                                                                num_tests, \
                                                                final_accuracy))

    return final_accuracy, attack_sample_list


def test_and_visualise_cw_attack(model, test_loader, num_tests, num_classes, target_label=1):
    accuracies = []
    examples = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the original accuracy and example images
    print("Getting original accuracies")
    accuracy_counter = 0
    # List of successful adversarial samples
    sample_list = []
    
    for image, label in test_loader:
        _, label = torch.max(label.data, 1)
        output = model(image)
        pred = output.max(1, keepdim = True)[1]

        # Check if pred correct
        if pred.item() == label.item():
            accuracy_counter += 1    
            if len(sample_list) < 10:
                ex = image.squeeze().detach().cpu().numpy()
                sample_list.append((pred.item(), pred.item(), ex))

    accuracies.append(accuracy_counter/float(num_tests))
    examples.append(sample_list)
    print("original accuracy:", accuracies[0])

    # Run attack
    print("Carlini Wagner: running attack...")
    acc, ex = test_cw_attack(model, device, test_loader, num_tests, num_classes,  target_label)
    accuracies.append(acc)
    examples.append(ex)

    title = "targeted_carlini_wagner_attack"
 
    # Show some example attack images
    examples_count = 0

    # Initialize figure
    plt.figure(figsize = (30, 30))

    # Browse through epsilon values and adversarial examples
    for i in range(len(accuracies)):
        for j in range(len(examples[i])):
            examples_count += 1
            plt.subplot(len(accuracies), len(examples[0]), examples_count)
            
            # Remove x-axis and y-axis ticks from plot
            plt.xticks([], [])
            plt.yticks([], [])
            
            if j == 0:
                if i == 0:
                    plt.ylabel("Original Accuracy: {}".format(accuracies[i]), fontsize = 14)
                else:
                    plt.ylabel("Attack Accuracy: {}".format(accuracies[i]), fontsize = 14)

            # Labels for each image subplot
            orig, adv, attack_img = examples[i][j]
            plt.title("ori: {}, attack: {}".format(orig, adv))

            img = attack_img.swapaxes(0,1)
            new_img = img.swapaxes(1,2)
            # Display image
            plt.imshow(new_img, cmap = "gray")

    plt.tight_layout()
    plt.savefig(os.path.join(attack_common.VISUALISATIONS_DIR, "example_images_" + title))
    plt.show()