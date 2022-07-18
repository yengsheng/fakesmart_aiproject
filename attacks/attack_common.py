
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

VISUALISATIONS_DIR = os.path.join('.', 'attacks', 'visualisations')


def test_and_visualise_basic_attack(model, test_attack_func, test_loader, num_tests, title):
  epsilons = [0, .005, .01, .015, .02, .025, .03, .05, 0.1]
  accuracies = []
  examples = []

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # Run test_attack_func() function for each epsilon
  for eps in epsilons:
      print(title, "epsilon:", eps)
      acc, ex = test_attack_func(model, device, test_loader, eps, num_tests)
      accuracies.append(acc)
      examples.append(ex)

  # Initialize figure
  plt.figure(figsize = (10, 7))

  # Display accuracy vs. Epsilon values plot
  plt.plot(epsilons, accuracies, "o-")

  # Adjust x-axis and y-axis labels and ticks
  plt.yticks(np.arange(0, 1.1, step = 0.1))

  plt.title("Accuracy vs. Epsilon")
  plt.xlabel("Epsilon")
  plt.ylabel("Accuracy")

  # Display
  plt.savefig(os.path.join(VISUALISATIONS_DIR, "accuracy_episilon_" + title))
  plt.show()
  
  # Show some example attack images
  cnt = 0

  # Initialize figure
  plt.figure(figsize = (30, 30))

  # Browse through epsilon values and adversarial examples
  for i in range(len(epsilons)):
      for j in range(len(examples[i])):
          cnt += 1
          plt.subplot(len(epsilons), len(examples[0]), cnt)
          
          # Remove x-axis and y-axis ticks from plot
          plt.xticks([], [])
          plt.yticks([], [])
          
          if j == 0:
              plt.ylabel("Eps: {}".format(epsilons[i]), fontsize = 14)
              
          # Labels for each image subplot
          orig, adv, attack_img = examples[i][j]
          plt.title("{} --> {}".format(orig, adv))

          img = attack_img.swapaxes(0,1)
          new_img = img.swapaxes(1,2)
          # Display image
          plt.imshow(new_img, cmap = "gray")

  plt.tight_layout()
  plt.savefig(os.path.join(VISUALISATIONS_DIR, "example_images_" + title))
  plt.show()
