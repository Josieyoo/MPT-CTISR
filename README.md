# MPT-CTISR
The existing prior-guided methods for text image super-resolution are usually for English, and the fusion of text semantic prior increases the amount of network parameters and computational costs dramatically, which limits the application of these methods on mobile devices. Aiming at this problem, a Chinese scene dataset suitable for super-resolution and a lightweight Chinese scene image super-resolution network incorporating text prior based on MobileViTv3(MPT-CTISR) are proposed. Firstly, convolutional layers and recognizer are employed to extract low super-resolution image features and text sequences, respectively. Then, principal component analysis and Transformer are incorporated into the feature fusion block to associate important text information with image features. Subsequently, the MVT3+ sequential residual blocks are constructed to learn high-dimensional deep features, capturing the details and complex relationships after feature fusion. Finally, image super-resolution reconstruction is achieved through sub-pixel convolution. In addition, a binary gradient loss function is put forward to guide the reconstruction to pay attention to text details by filtering noise. The experiments on the proposed dataset show that the quality of our reconstructed image is superior to existing methods. To further validate the model performance, comparative experiments are conducted with three classical recognizers, and our method achieves higher accuracy than other algorithms.

Pre-trained recognizer：https://github.com/FudanVI/benchmarking-chinese-text-recognition

Raw dataset：https://aistudio.baidu.com/aistudio/competition/detail/8/0/task-definition

CSID-SR：https://pan.baidu.com/s/1xi9rhuV3pXHqD9qF5u6Zjg?pwd=f5e2 
