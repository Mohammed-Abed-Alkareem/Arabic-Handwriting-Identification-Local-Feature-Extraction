
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans


def sift(img , test=False):
    """
    SIFT algorithm implementation
    :param img: input image
    :return: keypoints, descriptors
    """

    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    if test:
        image_keypoints = cv.drawKeypoints(img, keypoints, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(image_keypoints)

    # print("Number of keypoints Detected: ", len(keypoints))


    
        # SIFT implementation
    return keypoints, descriptors


def surf(img, hessianThreshold=400, test=False):
    """
    SURF algorithm implementation
    :param img: input image
    :return: keypoints, descriptors
    """

    surf = cv.xfeatures2d.SURF_create(hessianThreshold)  # Hessian Threshold is the threshold to filter out weak keypoints
    keypoints, descriptors = surf.detectAndCompute(img, None)
    
    # print("Number of keypoints Detected: ", len(keypoints))

    if test:
        image_keypoints = cv.drawKeypoints(img, keypoints, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(image_keypoints)



    return keypoints, descriptors



   
    
def get_images(path):
    """
    Get all images in a directory that may be in another directory
    :param path: directory path
    :return: images
    """
    # get all images
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                images.append(os.path.join(root, file))

    return images






def get_feature_vectors(images_path, algorithm='sift', hessianThreshold=400 , numOfClusters=100):
    
    # Extract BOW descriptors for all images
    vbowDescriptors = vBow_extract_all(images_path, algorithm, hessianThreshold)

    print(f"Finished extracting BOW descriptors for {len(vbowDescriptors)} images ^_^")

    
    # Cluster BOW descriptors
    kmeans = cluster_vbow(vbowDescriptors, numOfClusters)

    print(f"Finished clustering BOW descriptors into {numOfClusters} clusters ^_^")


    
    # Get BOW histograms
    histograms = get_vbow_histograms(vbowDescriptors, kmeans)

    print(f"Finished getting BOW histograms for {len(histograms)} images ^_^")
    
    return histograms



def vBow_extract_all(path, algorithm='sift', hessianThreshold=400):
    imgs = get_images(path)
    # Pre-allocate array
    VbowDescriptors = np.empty(len(imgs), dtype=object)
    
    for i, img_path in enumerate(imgs):
        img = cv.imread(img_path, 0)
        if algorithm == 'sift':
            _, descriptors = sift(img)
        elif algorithm == 'surf':
            _, descriptors = surf(img, hessianThreshold)
        VbowDescriptors[i] = descriptors
    
    return VbowDescriptors

def get_vbow_histograms(vbowDescriptors, kmeans):
    n_images = len(vbowDescriptors)
    histograms = np.zeros((n_images, kmeans.n_clusters))
    
    for i, descriptors in enumerate(vbowDescriptors):
        # Predict all clusters at once
        clusters = kmeans.predict(descriptors)
        # Use bincount for faster histogram creation
        hist = np.bincount(clusters, minlength=kmeans.n_clusters)
        histograms[i] = hist
        
    return histograms

def cluster_vbow(vbowDescriptors, k=100):
    # Calculate total number of descriptors

    #make sure that the descriptors in float64
    for i in range(len(vbowDescriptors)):
        vbowDescriptors[i] = np.float64(vbowDescriptors[i])

    total_desc = sum(desc.shape[0] for desc in vbowDescriptors)
    descriptors = np.empty((total_desc, vbowDescriptors[0].shape[1]))
    
    # Fill array more efficiently
    idx = 0
    for desc in vbowDescriptors:
        n_desc = desc.shape[0]
        descriptors[idx:idx + n_desc] = desc
        idx += n_desc
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(descriptors)
    return kmeans

def create_inverted_file(histograms):
    # Convert to sparse format for efficiency
    histograms = np.array(histograms)
    inverted_file = {}
    
    # Use numpy operations for finding non-zero elements
    non_zero_indices = np.nonzero(histograms)
    for img_idx, word_idx in zip(non_zero_indices[0], non_zero_indices[1]):
        if word_idx not in inverted_file:
            inverted_file[word_idx] = []
        inverted_file[word_idx].append(img_idx)
    
    return inverted_file