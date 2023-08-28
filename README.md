# The-code-for-our-cross-modal-super-resolution-reconstruct-work.
This is the code for the cross-modal super-resolution reconstruction.

## **Seup
We run the program on a desktop using python. Environment requirements:

Torch                   1.8.0+cu111
Torchvision             0.9.0+cu111
Tensorboard             2.6.0
scipy                   1.2.1

## **Train the model:

	**Firstly, conduct the training for Stage 1, which involves independently running the following several programs.
### Train the HR encoder and decoder network.
	···python s1_1_hr_fenlei.py
	···python s1_train_split.py
 
Then, conduct the training for stage 2, which involves independently running the following several programs.
### Train the LR and Haptic encoder network.
  ···python s2_1_lr_fenlei.py
  ···python s2_2t_fenlei.py

### Train the mapping network.
 ···python s2_train_mapping_lr-t.py
  
### Train the whole network.
 ···python s2_fusion_network.py
 ···python s2_train_fine-tuning.py
  
## **Evaluation:
	After the completion of model training, you can utilize the saved encoding networks for tactile signals, encoding network for low-resolution visual signals, fusion network, and the final generative network for evaluation.

## Data:

The original dataset is from LMT-108-Surface-Material database which can be downloaded at https://zeus.lmt.ei.tum.de/downloads/texture/.
The dataset we constructed for our practical cross-modal communication platform  is available at https://github.com/dadajuan/VisTouchdataset.
