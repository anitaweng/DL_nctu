# DCGAN & DQN
1. For problem 1, implementing DCGAN to reconstruct human faces.
   The related files are in GAN_source_code folder.
   Setting the parameters in main.py.
   ```
   parser.add_argument('--dataroot', default='data/celeba', type=str)
   parser.add_argument('--batch_size', default=128, type=int)
   parser.add_argument('--image_size', default=64, type=int)
   parser.add_argument('--num_epochs', default=5, type=int)
   parser.add_argument('--lr', default=0.0002, type=float)
   ```
   The simluated results are shown as follow:
   
   ![image](https://user-images.githubusercontent.com/42642215/132985990-ad2d6351-3ee0-447b-8c16-569c0fbcfe90.png)
   
2. For problem 2, only slightly modify Deep Q Network from TA's code.
   The related files are in DQN_source_code folder.
   The simluated result is shown as follow:
   
   ![image](https://user-images.githubusercontent.com/42642215/132986251-94c222c4-5cba-49b9-86ac-d4900ade0255.png)

   
