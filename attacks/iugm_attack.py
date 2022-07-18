import torch
import torch.nn.functional as F
import numpy as np

def iugm_attack(image, epsilon, model, original_label, iter_num = 10):
    # print("attacking...")
    eps_image = image
   
    # The image does not change if epsilon is 0
    if epsilon != 0:
        # Run the model on the original image to find the least probable class
        image.grad = None

        output = model(image)
        _, least_probable_class = output.data.min(1)
        
        for i in range(iter_num):
            image.grad = None

            # Backpropagate
            pred_loss = F.nll_loss(output, least_probable_class)
            pred_loss.backward(retain_graph = True)

            # Gradients of image
            img_grad = image.grad.data

            # Generate new image with noise
            eps_image = image - epsilon*img_grad
            eps_image.retain_grad()
            
            # Clip eps_image to maintain pixel values in [0, 1] range
            eps_image = torch.clamp(eps_image, 0, 1)
            # eps_image_unsqueezed = torch.unsqueeze(eps_image, 0)

            # Check if the new prediction is the least probable class
            new_output = model(eps_image)
            _, new_pred = new_output.data.max(1)

            # If new_pred is already the least probable, we can stop
            if new_pred == least_probable_class:
              # print("least probable achieved")
              break
            else:
                image = eps_image
                image.retain_grad()
                output = new_output

    return eps_image[0]

# Return the accuracy after the attack and the adversarial examples
def test_iugm_attack(model, device, test_loader, epsilon, num_tests):
    # Counter for correct values (used for accuracy)
    accuracy_counter = 0

    model.eval() 

    attack_sample_list = []

    for image, label in test_loader:
        if image is None:
          continue
        # # Model uses batch normalisation
        # image_unsqueezed = torch.unsqueeze(image, 0)

        # Send the data and label to the device
        image_unsqueezed, label = image.to(device), label.to(device)

        image.requires_grad = True
        image_unsqueezed.requires_grad = True

        # Get initial prediction before attack
        outputs = model(image_unsqueezed)

        # Get the index of the max log-probability
        _, init_pred = torch.max(outputs, 1)
        _, label = torch.max(label.data, 1)

        # Skip image if the model already incorrectly classifies it
        if init_pred.item() != label.item():
          # print("already wrong")
          continue
            
        # Attack!
        eps_image = iugm_attack(image_unsqueezed, epsilon, model, label)

        eps_image_unsqueezed = torch.unsqueeze(eps_image, 0)
        eps_images = torch.cat((eps_image_unsqueezed, eps_image_unsqueezed), 0)

        # Get new prediction after attack
        attack_output = model(eps_images)
        
        # Get the index of the max log-probability
        _, eps_pred = torch.max(attack_output.data, 1)
        eps_pred = eps_pred[0]

        # Check if attck label different from init pred
        if eps_pred.item() == label.item():
            accuracy_counter += 1
            # If epsilon is 0, then we keep some examples anyway
            if (epsilon == 0) and (len(attack_sample_list) < 5):
                adv_ex = eps_image.squeeze().detach().cpu().numpy()
                attack_sample_list.append((init_pred.item(), eps_pred.item(), adv_ex))
        else:
            # Save some attack images for display
            if len(attack_sample_list) < 5:
                adv_ex = eps_image.squeeze().detach().cpu().numpy()
                attack_sample_list.append((init_pred.item(), eps_pred.item(), adv_ex))

    final_accuracy = accuracy_counter/float(num_tests)
    
    # Display for progress
    print("Epsilon: {} - Model Accuracy (under attack) = {}/{} = {}".format(epsilon, \
                                                                            accuracy_counter, \
                                                                            num_tests, \
                                                                            final_accuracy))

    return final_accuracy, attack_sample_list