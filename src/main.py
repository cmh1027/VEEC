from emotions import expVec_face
from emotions_emoji import expVec_emoji
import os
import numpy as np
import argparse

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
args = parser.parse_args()

faces = expVec_face(args.name)

data = {}

for filename in os.listdir("test/"):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        exp_emoji = expVec_emoji("test/" + filename)
        data[filename] = exp_emoji

for idx, face in enumerate(faces):
    max_similarity = None
    max_filename = None
    print("face : ", face)
    for filename, emoji in data.items():
        print(filename, emoji)
        if max_similarity is None and max_filename is None:
            max_similarity = cosine_similarity(face, emoji)
            max_filename = filename
        else:
            similarity = cosine_similarity(face, emoji)
            if max_similarity < similarity:
                max_similarity = similarity
                max_filename = filename
    print("Face %d : %s" % (idx, max_filename))         
