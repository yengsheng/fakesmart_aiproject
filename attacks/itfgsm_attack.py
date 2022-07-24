from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np

TARGET_CLASS = 1

def itfgsm_attack(image, epsilon, model, orig_class, target_class, iter_num = 10):
    
    # The image does not change if epsilon is 0
    eps_image = image
    worked = False

    if epsilon != 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        target_class_variable = Variable(torch.from_numpy(np.asarray([target_class])))
        target_class_torch = target_class_variable.type(torch.LongTensor).to(device)
        
        for i in range(iter_num):
            image.grad = None

            pred = model(image)
            # loss
            pred_loss = F.nll_loss(pred, target_class_torch)
            pred_loss.backward(retain_graph = True)
            
            # Gradients of image
            img_grad = image.grad.data

            # Add noise to processed image
            eps_image = image - epsilon*torch.sign(img_grad)
            eps_image.retain_grad()
            
            # Clip eps_image to maintain pixel values in [0, 1] range
            eps_image = torch.clamp(eps_image, 0, 1)
            
            # Check if the new prediction is the target class
            new_output = model(eps_image)
            _, new_pred = new_output.data.max(1)
            
            if new_pred == target_class_torch:
                worked = True
                break
            else:
                image = eps_image
                image.retain_grad()
            
    return eps_image[0], worked


# Return the accuracy after the attack and the adversarial examples
def test_itfgsm_attack(model, device, test_loader, epsilon, num_tests):
    model.eval()

    accuracy_counter = 0
    
    # List of successful adversarial samples
    attack_sample_list = []

    for image, label in test_loader:
        if image is None:
          continue

        # label = torch.unsqueeze(label, 0)
        _, label = torch.max(label.data, 1)

        # Skip image if the target label is the same as the current label
        if TARGET_CLASS == label.item():
            accuracy_counter += 1
            continue

        # Model uses batch normalisation
        # image_unsqueezed = torch.unsqueeze(image, 0)

        # Send the data and label to the device
        image_unsqueezed, label = image.to(device), label.to(device)

        image.requires_grad = True
        image_unsqueezed.requires_grad = True

        # Get initial prediction before attack
        output = model(image_unsqueezed)

        # Get the index of the max log-probability
        init_pred = output.max(1, keepdim = True)[1]

        
        # print("initial eval before attack:",init_pred, label)
        
        # If the initial prediction is wrong, do not bother attacking, skip current image
        if init_pred.item() != label.item():
            continue

        # Attack!
        eps_image, worked = itfgsm_attack(image_unsqueezed, epsilon, model, label, TARGET_CLASS)
        # print("eps_image.size()",eps_image.size())
        eps_image_unsqueezed = torch.unsqueeze(eps_image, 0)

        # Get new prediction after attack
        attack_output = model(eps_image_unsqueezed)

        # Get the index of the max log-probability
        _, eps_pred = torch.max(attack_output.data, 1)
        eps_pred = eps_pred[0]

        # Check if attck label different from init pred
        if eps_pred.item() == label.item():
            accuracy_counter += 1            
            # If epsilon is 0, then we keep some examples anyway
            if (epsilon == 0) and (len(attack_sample_list) < 10):
                adv_ex = eps_image.squeeze().detach().cpu().numpy()
                attack_sample_list.append((init_pred.item(), eps_pred.item(), adv_ex))
        else:
            # Save some attack images for display
            if len(attack_sample_list) < 10:
                adv_ex = eps_image.squeeze().detach().cpu().numpy()
                attack_sample_list.append((init_pred.item(), eps_pred.item(), adv_ex))

    final_accuracy = accuracy_counter/float(num_tests)
    
    # Display for progress
    print("Epsilon: {} - Test Accuracy = {}/{} = {}".format(epsilon, \
                                                            accuracy_counter, \
                                                            num_tests, \
                                                            final_accuracy))

    return final_accuracy, attack_sample_list
