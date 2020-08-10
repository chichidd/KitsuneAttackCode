# KitsuneAttackCode
This repository contains python code in form of Jupyter Notebook for adversarial attack on Kitsune model. The experiments are conducted on dataset Mirai (code in "exp-mirai") and VideoInjection (code in "exp-videoinjection") of \[1\].
Note: Copying the folder "data" and "model" (can be found [here](https://drive.google.com/drive/folders/1_GPJrO0drKq6qbL1GKi0ebOBu_g208tj?usp=sharing.) under respective folder.

# Run the test for dataset Mirai (in "exp-mirai")
  - TrainTorchFE.ipynb : train and save the entire Kitsune model.
  - AdversarialTest.ipynb : Find the optimal threshold for Kitsune classifier and compute saliency maps for benign and malicious inputs.
  - AttackerInTraining.ipynb : Simulate an attacker monitoring a small portion (5%) of network packets to create a local Kitsune model and launching adversarial attack based on his local model.
  
# Run the test for dataset VideoInjection (in "exp-videoinjection")
  - TrainTorchFE.ipynb : train and save the entire Kitsune model.
  - AdversarialTest.ipynb: Find the optimal treshold and show the impact of packet loss to Kitsune. To run the code directly, you need to download files under folder "videoinj" of the link above whose name starts with "RMSEs_packet_loss" into the folder "Kitsune". Also, you can run the Python script "Kitsune/packet_loss.py" to regenerate the same files. Note that you may change the packet loss probability as indicated in the script.


# Reference
[1]. Kitsune Python implementation https://github.com/ymirsky/Kitsune-py
