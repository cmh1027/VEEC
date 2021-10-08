# VEEC (Vector-based Emoji Emotion classifier)

## Introduction
Conventional transformer that transforms a human face to the corresponding emoji uses method that trains data with labeled, for example assign numbers to each emotion.
And corresponding emojis are subordinated to the number. However, we used a different approach : namely, we train emojis seperately like human faces. By doing that, we can pick utmost similar emoji by using cosine-similarity between human-face vector and emoji vector.

## Advantage
1. We can add much more emojis than 7 emotions, like "superhappy", which can be exampled by testing superhappy.jpg
2. We can assign several emojis to the same emotion so that more similar emoji is matched to the human face

## Data augmentation
Because our emoji dataset is too small, we used several data augmentation methods, and it helps the overfitting problem.

## Dependencies
* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)


