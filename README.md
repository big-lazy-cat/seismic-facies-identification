# Seismic Facies Identification Challenge - Mengdi

There are 4 scripts: "train_inference_class0123.ipynb", "train_inference_class4.ipynb", "train_inference_class5.ipynb" and "binary_ensemble.ipynb".

I used Google Colab to run the scripts (with GPU option enabled). The training data was put in the directory "data". 

Note that the initial labels were from 1 to 6. I added -1 for each label. So in my script, labels become from 0 to 5. 

Since the metric is weighted F1 score with much more weight on class 4 and class 5, I used the script "train_inference_class0123.ipynb" to train class 0, 1, 2 and 3, and 2 other scripts to train class 4 and class 5.
"train_inference_class0123.ipynb", "train_inference_class4.ipynb" and "train_inference_class5.ipynb" are very similar, with 2 differences: the number of classes in image segmentation, and the class weights given to each class during training.
* "train_inference_class0123.ipynb" merges classes 4 and 5 into class 1, so the number of classes is 4. The 4 classes have the same weight. 
* "train_inference_class4.ipynb" keeps all of the classes during training, so the number of classes is 6. Class 4 has more class weight than other classes. 
* "train_inference_class5.ipynb" keeps all of the classes during training, so the number of classes is 6. Class 5 has more class weight than other classes. 
* "binary_ensemble.ipynb" combines the inference results of the first 3 scripts and generates the final submission file.

I added utils.set_global_seed(SEED) and utils.prepare_cudnn(deterministic=True) at the beginning of the scripts to decrease training randomness. Nevertheless, there are still other randomness in training. At each run, the training result may be slightly different.

To replicate the results:
1. Run "train_inference_class0123.ipynb". It will train an image segmentation model, do inference and save inference results in a .npz file in the folder of "binary_predictions".
2. Run "train_inference_class4.ipynb". It will train an image segmentation model, do inference and save inference results in a .npz file in the folder of "binary_predictions".
3. Run "train_inference_class5.ipynb". It will train an image segmentation model, do inference and save inference results in a .npz file in the folder of "binary_predictions".
4. Run "binary_ensemble.ipynb". It will use the 3 .npz files previously generated to generate the final submission.
