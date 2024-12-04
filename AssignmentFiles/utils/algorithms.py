
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
import faiss
import time


def sift(img , test=False):
    """
    SIFT algorithm implementation
    :param img: input image
    :return: keypoints, descriptors
    """

    #if the image is colored, convert it to gray
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    if test:
        image_keypoints = cv.drawKeypoints(img, keypoints, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(image_keypoints)

    return keypoints, descriptors


def orb(img, test=False):

    #if the image is colored, convert it to gray
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if test:
        image_keypoints = cv.drawKeypoints(img, keypoints, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(image_keypoints)

    return keypoints, descriptors


def extract_features(
    images, 
    labels, 
    algorithm='sift', 
    numOfClusters=100, 
    Test=False, 
    kMeans=None, 
    Compare=False
):  
    """
    Extract features from images and optionally cluster them into a Bag of Visual Words representation.

    :param images: List of input images.
    :param labels: List of corresponding labels for the images.
    :param algorithm: Feature extraction algorithm ('sift' or 'orb').
    :param numOfClusters: Number of clusters for visual words.
    :param Test: If True, skips clustering and directly forms feature vectors using precomputed kMeans.
    :param kMeans: Pre-trained kMeans object (used in Test mode).
    :param Compare: If True, measures and outputs time for feature extraction, clustering, and feature formation.
    :return: Feature vectors, labels, and optionally the kMeans object.
    """
    all_descriptors = []  # List to store all descriptors
    processed_labels = []  # List to store labels for processed images

    if Compare:
        time_extraction = []
        num_keypoints = 0

    for img, label in zip(images, labels):
        # Ensure the image is in the correct format
        if len(img.shape) == 3:  # Convert colored images to grayscale
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if Compare:
            start_time = time.time()

        # Extract descriptors using the specified algorithm
        if algorithm == 'sift':
            _, descriptors = sift(img)
        elif algorithm == 'orb':
            _, descriptors = orb(img)
        else:
            raise ValueError("Unsupported algorithm. Use 'sift' or 'orb'.")

        if Compare:
            end_time = time.time()
            time_extraction.append(end_time - start_time)

        # If descriptors are extracted, append them
        if descriptors is not None:
            all_descriptors.append(descriptors)
            processed_labels.append(label)

            if Compare:
                num_keypoints += descriptors.shape[0]

    print("Number of processed images: ", len(all_descriptors))

    if Compare:
        print("Average number of keypoints per image: ", num_keypoints / len(all_descriptors))
        print("Average time for feature extraction: ", np.mean(time_extraction))

    # If in Test mode, form feature vectors using the precomputed kMeans
    if Test:
        feature_vectors = get_vbow_featureVectors(all_descriptors, kMeans)
        return feature_vectors, processed_labels

    # Otherwise, perform clustering to generate visual words
    if Compare:
        clustering_start = time.time()

    print("Clustering descriptors...")
    kmeans = cluster_vbow(all_descriptors, numOfClusters, use_gpu=True)

    if Compare:
        clustering_time = time.time() - clustering_start
        print(f"Clustering time: {clustering_time:.2f} seconds")

    # Form feature vectors using the clustered visual words
    if Compare:
        forming_start = time.time()

    print("Forming feature vectors...")
    feature_vectors = get_vbow_featureVectors(all_descriptors, kmeans)

    if Compare:
        forming_time = time.time() - forming_start
        print(f"Forming time: {forming_time:.2f} seconds")
        return time_extraction, clustering_time, forming_time

    return feature_vectors, processed_labels, kmeans



def get_vbow_featureVectors(imagesDiscreptors, kmeans):
    n_images = len(imagesDiscreptors)
    n_clusters = kmeans.k  # Use the correct attribute for the number of clusters
    featureVectors = np.zeros((n_images, n_clusters), dtype=np.float32)
    
    for i, descriptors in enumerate(imagesDiscreptors): # loop over all images
        # Predict all clusters at once
        featureVectors[i] = get_vbow_featureVector(descriptors, kmeans)

    featureVectors = compute_tfidf_weights(featureVectors)
        
    return featureVectors # return the feature vectors of all images


def get_vbow_featureVector(imagesDiscreptor, kmeans):
    """
    Compute raw feature vector (term counts) for a single image using FAISS KMeans.
    :param imagesDiscreptor: Descriptors of the image.
    :param kmeans: Trained FAISS KMeans object.
    :return: Raw feature vector (term counts).
    """
    imagesDiscreptor = np.float32(imagesDiscreptor)  # Ensure float32
    featureVector = np.zeros(kmeans.k, dtype=np.float32)
    _, clusters = kmeans.index.search(imagesDiscreptor, 1)  # Find the nearest cluster for each descriptor
    for cluster in clusters:
        featureVector[cluster[0]] += 1  # Increment the count for each cluster

    return featureVector  # Return raw term counts



def compute_tfidf_weights(feature_vectors):
    """
    Compute TF-IDF weights for visual Bag-of-Words feature vectors.
    :param feature_vectors: Array of shape (n_images, n_clusters), 
                            where each entry is the count of a cluster in an image.
    :return: TF-IDF weighted feature vectors.
    """
    # Term Frequency (TF)
    tf = feature_vectors / np.sum(feature_vectors, axis=1, keepdims=True)  # Normalize by image total

    # Document Frequency (DF): Number of images containing each cluster
    df = np.sum(feature_vectors > 0, axis=0)  # Count images where a cluster appears

    # Inverse Document Frequency (IDF)
    n_images = feature_vectors.shape[0]
    idf = np.log((n_images + 1) / (df + 1)) + 1  # Adding 1 to avoid division by zero

    # TF-IDF Weighting
    tfidf = tf * idf

    return tfidf




def cluster_vbow(imagesDiscreptors, k=100, use_gpu=False):
    """
    Cluster descriptors using FAISS with optional GPU acceleration.
    :param imagesDiscreptors: List of descriptors from all images.
    :param k: Number of clusters (default=100).
    :param use_gpu: Whether to use GPU acceleration (default=False).
    :return: Trained FAISS KMeans object.
    """
    print(f"Using GPU: {use_gpu}")

    # Ensure all descriptors are in float32 format (required for FAISS)
    for i in range(len(imagesDiscreptors)):
        imagesDiscreptors[i] = np.float32(imagesDiscreptors[i])

    # Calculate total number of descriptors
    total_desc = sum(desc.shape[0] for desc in imagesDiscreptors)  # Total number of keypoints in all images
    descriptors = np.empty((total_desc, imagesDiscreptors[0].shape[1]), dtype=np.float32)  # Pre-allocate array

    # Fill array efficiently
    idx = 0
    for desc in imagesDiscreptors:
        n_desc = desc.shape[0]  # Number of keypoints in the current image
        descriptors[idx:idx + n_desc] = desc  # Fill array with keypoints of the current image
        idx += n_desc

    d = descriptors.shape[1]  # Dimension of descriptors

    # Create FAISS KMeans
    kmeans = faiss.Kmeans(d=d, k=k, niter=20, verbose=True, gpu=use_gpu)

    # Train FAISS KMeans
    kmeans.train(descriptors)

    print(f"Clustering complete. Number of clusters: {k}")

    return kmeans

    
   