# Cats vs Dogs Classification with Pretrained ResNet-50 and Grad-CAM

Classifies cat and dog images using a pretrained ResNet-50 (no training from scratch)
and explains its predictions with Grad-CAM heatmaps. Built with PyTorch on Google Colab.

## Dataset
[Cats and Dogs (Kaggle)](https://www.kaggle.com/datasets/yashdogra/cats-and-dogs),
downloaded in the notebook with `kagglehub`. A balanced random subset (100 per class) is used.

## Method
1. **Preprocessing:** resize to 256, center-crop 224, ImageNet normalization.
2. **Model:** torchvision ResNet-50 with ImageNet weights (inference only).
3. **Cat/dog mapping:** ImageNet classes 151-268 (dog breeds) and 281-285 (domestic cats)
   are grouped; the group with the larger summed softmax probability is the prediction.
4. **Evaluation:** accuracy, classification report, confusion matrix.
5. **Visualization:** grids of correctly and incorrectly classified images.
6. **Grad-CAM:** gradients of the top class score with respect to the last `layer4`
   block's feature maps are averaged to get channel weights, then a weighted sum with
   ReLU gives a heatmap overlaid on the image.

## Run it
1. Open `cats_dogs_resnet_gradcam.ipynb` in Google Colab.
2. Set the runtime to GPU.
3. Run all cells (Kaggle credentials may be needed for the download).

## Results
Accuracy: The pre-trained ResNet-50 was tested on 200 images (100 cats and 100 dogs) without any training. It classified 190 images correctly and 10 incorrectly, giving an accuracy of 95%.

The confusion matrix shows that all 100 dogs were classified correctly, while 10 of the 100 cats were wrongly predicted as dogs. This means dog recall was 1.00 and cat recall was 0.90. Cat precision was 1.00, because every time the model said “cat” it was right. Dog precision was 0.91, because 10 of its 110 “dog” predictions were actually cats.

All 10 mistakes were the same type (cat predicted as dog). The model is therefore slightly biased towards predicting “dog”. A likely reason is that ImageNet, the dataset ResNet was trained on, contains about 118 dog classes but only 5 domestic cat classes.

Accuracy: 0.9500

              precision    recall  f1-score   support

         cat       1.00      0.90      0.95       100
         dog       0.91      1.00      0.95       100

    accuracy                           0.95       200
   macro avg       0.95      0.95      0.95       200
weighted avg       0.95      0.95      0.95       200

Confusion matrix:
 [[ 90  10]
 [  0 100]]

## Observations
Correct vs incorrect images section:
In the correctly classified grid, dogs were mostly predicted with a confidence of about 1.00, while cats had lower confidence (around 0.80 to 0.90). So the model is more certain about dogs than cats.
The incorrectly classified grid contains 10 cats. By appearance, 6 are hairless Sphynx cats and 4 are Abyssinian cats. Sphynx cats have wrinkled skin, large ears and no fur, so they look quite different from a typical cat. Some Abyssinians were also hard cases, for example one standing in a side view and one lying down with its body covering most of the image. Several wrong predictions had very high confidence (0.99, 0.98, 0.93), which shows the model can be confidently wrong. Others were near 0.50, meaning the model was almost undecided.

Grad-CAM observations section:
Original vs heatmap: Grad-CAM shows which parts of the image the model used for its decision. Red and yellow areas mean strong influence, and blue areas mean little influence.

Correctly classified images:
The Abyssinian cat’s heat is strongest on its eyes and the centre of its face.
For the dogs, the heat sits on the face, nose and muzzle (for example the Samoyed’s face).
In the Samoyed image, the chairs and the cluttered room in the background are blue, so the model correctly ignored the background and focused on the dog.
This suggests that for correct predictions the model looks at the animal’s face, which is a sensible thing to do.

Misclassified images:
For the hairless cats, the heat is on the cheek, forehead or neck/wrinkled skin rather than on a cat-like face. The large ears are not highlighted. The model seems to rely on features that it also associates with some dog breeds.
For the Abyssinian cats, the heat falls on the body and the white bedding instead of the face. In the sleeping cat, the face has almost no heat, and in the standing cat the head is cut off at the edge of the image. Without a clear face, the model had no strong cat evidence.
The heatmaps for the wrong predictions are also more scattered and less focused on the face than those for the correct ones.

Conclusion:
The pre-trained ResNet-50 classifies cats and dogs with 95% accuracy without any training. Grad-CAM showed that it normally focuses on the animal’s face. Mistakes happened mostly with unusual-looking cats (hairless Sphynx) and images where the face was hidden or not visible. Fine-tuning the last layer on cat images would likely fix many of these errors.

## Limitations
Zero-shot mapping depends on ImageNet's label set. Fine-tuning the final layer
would likely improve accuracy further.

## Requirements
torch, torchvision, matplotlib, scikit-learn, pillow, kagglehub# cats-dogs-resnet-gradcam
