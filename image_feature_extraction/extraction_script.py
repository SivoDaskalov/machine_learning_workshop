import os

import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def auto_crop(image):
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayimage, 128, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])
        cropped_image = grayimage[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        # cv2.imshow('image', cropped_image)
        # cv2.waitKey(0)
        return cropped_image

    return None


def extract_features(image_path, vector_size=32):
    image = cv2.imread(image_path)
    image = auto_crop(image)
    cv2.imwrite(os.path.join(os.curdir, 'cropped', os.path.basename(image_path)), image)
    try:
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: %s' % e)
        return None
    return dsc


def batch_extractor(images_path, vector_size=32):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f, vector_size=vector_size)
    return result


def build_cosine_similarity_matrix(features):
    images = features.keys()
    similarities = np.empty((len(images), len(images)), dtype=object)
    for row_idx, row_label in enumerate(images):
        for col_idx, col_label in enumerate(images):
            similarity = cosine_similarity(features[row_label].reshape(1, -1), features[col_label].reshape(1, -1))[0, 0]
            similarities[row_idx][col_idx] = np.round(similarity, 4)
    images = [name.rsplit('\\', 1)[1].split('.')[0].upper() for name in images]
    similarities = pd.DataFrame(data=similarities.astype(float), index=images, columns=images)
    return similarities


def plot_similarities(dataframe, title=None, cmap=plt.cm.Blues, vmin=0.5, vmax=1.0, label_color_thresh=None):
    figure = plt.figure()
    if title:
        plt.title(title, y=1.12)
    if not label_color_thresh:
        label_color_thresh = vmin + 0.5 * (vmax - vmin)

    plt.imshow(dataframe.as_matrix(), interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.yticks(np.arange(len(dataframe.index)), dataframe.index)
    plt.xticks(np.arange(len(dataframe.columns)), dataframe.columns, rotation=0)
    plt.gca().xaxis.tick_top()

    for i, j in itertools.product(range(dataframe.shape[0]), range(dataframe.shape[1])):
        plt.text(j, i, "%.2f" % dataframe.iloc[i, j],
                 horizontalalignment="center", color="white" if dataframe.iloc[i, j] > label_color_thresh else "black")

    plt.tight_layout()
    return figure


vector_sizes = [16, 32, 64, 128, 256, 512, 1024]

for vector_size in vector_sizes:
    features = batch_extractor(os.path.join(os.curdir, 'images'), vector_size=vector_size)
    similarities = build_cosine_similarity_matrix(features)
    similarities.to_csv('similarities_%d.csv' % vector_size)
    # figure = plot_similarities(similarities, title='Cosine similarity between %d KAZE features' % vector_size)
    figure = plot_similarities(similarities, vmin=0.0)
    figure.savefig("%s/similarities_%d.png" % (os.curdir, vector_size))
