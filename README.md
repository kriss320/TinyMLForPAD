# TinyMLForPAD
This project uses neural networks to classify images into **Bona Fide** and **Presentation Attacks**. The leading software used is Pytorch and Timm. All code has been created with the help of [Pytorch documentation](https://docs.pytorch.org/docs/stable/index.html), where specific resources have been used to make the code cited above the function. ChatGPT has been used to help with bugfixing, increased readability and formatting of the code. 
The choice of different parameters is explained in more detail in the thesis. 

The code is organized into separate files for the methods described in the thesis.
All files can be run individually. The training process takes a while to complete. Due to inherent randomness in the training, it will produce different results over time. Therefore, it is recommended to run the training process numerous times to get the best possible model.

Image locations are expected to be in the following format:

    dataset/
    ├── BonaFide/      (containing all BonaFide images, may have subfolders)
    └── PAs/           (containing all PA images, may have subfolders)


