# AFR-ConvMixer
Our aim is to develop an advanced Automated Face Recognition system that excels in recognizing faces even in challenging situations, such as when sunglasses or hats partially obscure faces. The algorithm's configurations and settings are crucial in achieving this goal.

We define key parameters, including three ConvMixer blocks for robust feature extraction and five iterations in the AdaBoost algorithm for precision enhancement. Leveraging pre-trained CNN models like ResNet-50, Inception-v3, and DenseNet-161, we tap into their learned features to aid in identifying facial attributes.

The preprocessing phase ensures data uniformity by resizing and normalizing pixel values. Initializing pre-trained CNN models, configuring ConvMixer architecture, and integrating skip connections enhance feature capture and learning.

The AdaBoost component combines ConvMixer predictions and iteratively refines sample weights. Pre-trained CNN models extract features for face recognition, and AdaBoost's ensemble learning produces predictions.

Performance evaluation involves metrics like accuracy, precision, recall, and F1-score, which indicate the algorithm's efficacy in recognizing faces even under complex conditions.

After meticulous training and tuning, the Automated Face Recognition system achieves an impressive 97% accuracy, uniting ConvMixer's feature extraction, pre-trained CNN models' expertise, and AdaBoost's ensemble predictions. This approach demonstrates its potential to elevate facial recognition, yielding higher accuracy and resilience against real-world challenges.
