# TinyMLForPAD
This project aims to classify images into **Bona Fide** and **Presentation Attacks** using neural networks. The main software used is Pytorch and Timm. All code has been created with the help of [Pytorch documentation](https://docs.pytorch.org/docs/stable/index.html), where spesific resources have been used to make the code it is sited above the function. Copilot has been used after the code was created to help with increased readability and formating of the code. 

The code is organized into separate files for the different methods described in the thesis.
All files are able to be run individually. Training process takes a while to complete. Due to inherent randomness in the trainig it will produce different results over time, therefore it is recomended to run the training process numerous times to get the best possible model

Image locations are expected to be in the following format:

    dataset/
    ├── BonaFide/      (containing all BonaFide images, may have subfolders)
    └── PAs/           (containing all PA images, may have subfolders)


