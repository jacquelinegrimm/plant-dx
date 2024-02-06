# Plant Disease Classifier

This repository contains resources to train a model for classifying images of diseased and healthy plant leaves.

## Files

- `plant-disease-classifier-v1.ipynb`: Jupyter Notebook for training the classification model.
- `app.py`: Code for deploying a demo using Gradio.

## Dataset

The model is based on the `vit_small_patch16_224` architecture and fine-tuned using a dataset uploaded to Kaggle by Samir Bhattarai. The dataset can be found [here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

### Classes

The model was trained to classify images into 38 classes:

1. Apple - Apple Scab
2. Apple - Black Rot
3. Apple - Cedar Apple Rust
4. Apple - Healthy
5. Blueberry - Healthy
6. Cherry - Powdery Mildew
7. Cherry - Healthy
8. Corn - Gray Leaf Spot
9. Corn - Common Rust
10. Corn - Northern Leaf Blight
11. Corn - Healthy
12. Grape - Black Rot
13. Grape - Esca (Black Measles)
14. Grape - Isariopsis Leaf Spot
15. Grape - Healthy
16. Orange - Citrus Greening
17. Peach - Bacterial Spot
18. Peach - Healthy
19. Bell Pepper - Bacterial Spot
20. Bell Pepper - Healthy
21. Potato - Early Blight
22. Potato - Late Blight
23. Potato - Healthy
24. Raspberry - Healthy
25. Soybean - Healthy
26. Squash - Powdery Mildew
27. Strawberry - Leaf Scorch
28. Strawberry - Healthy
29. Tomato - Bacterial Spot
30. Tomato - Early Blight
31. Tomato - Late Blight
32. Tomato - Leaf Mold
33. Tomato - Septoria Leaf Spot
34. Tomato - Spider Mites
35. Tomato - Target Spot
36. Tomato - Yellow Leaf Curl Virus
37. Tomato - Tomato Mosaic Virus
38. Tomato - Healthy

## Demo

Check out the demo at [HuggingFace Spaces](https://huggingface.co/spaces/jacquelinegrimm/plant-dx).
