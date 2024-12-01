
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
    imagesDiscreptors = vBow_extract_all(images_path, algorithm, hessianThreshold)

    print(f"Finished extracting BOW descriptors for {len(imagesDiscreptors)} images ^_^")

    
    # Cluster BOW descriptors
    kmeans = cluster_vbow(imagesDiscreptors, numOfClusters)

    print(f"Finished clustering BOW descriptors into {numOfClusters} clusters ^_^")


    
    # Get BOW featureVectors
    featureVectors = get_vbow_featureVectors(imagesDiscreptors, kmeans)

    print(f"Finished getting BOW featureVectors for {len(featureVectors)} images ^_^")

    invertedFile = create_inverted_file(imagesDiscreptors , kmeans)
    
    return featureVectors , kmeans , invertedFile



def vBow_extract_all(path, algorithm='sift', hessianThreshold=400):
    imgs = get_images(path)
    # Pre-allocate array
    imagesDiscreptors = np.empty(len(imgs), dtype=object)
    
    for i, img_path in enumerate(imgs):
        img = cv.imread(img_path, 0)
        if algorithm == 'sift':
            _, descriptors = sift(img)
        elif algorithm == 'surf':
            _, descriptors = surf(img, hessianThreshold)
        imagesDiscreptors[i] = descriptors
    
    return imagesDiscreptors

def get_vbow_featureVectors(imagesDiscreptors, kmeans):
    n_images = len(imagesDiscreptors)
    featureVectors = np.zeros((n_images, kmeans.n_clusters))
    
    for i, descriptors in enumerate(imagesDiscreptors): # loop over all images
        # Predict all clusters at once
        clusters = kmeans.predict(descriptors) # get the cluster of each keypoint in the image
        # Count occurrences of each cluster
        for cluster in clusters: 
            featureVectors[i, cluster] += 1 # increment the count of the cluster in the feature vector of the image

    return featureVectors # return the feature vectors of all images

def cluster_vbow(imagesDiscreptors, k=100):
    # Calculate total number of descriptors

    #make sure that the descriptors in float64
    for i in range(len(imagesDiscreptors)):
        imagesDiscreptors[i] = np.float64(imagesDiscreptors[i])

    total_desc = sum(desc.shape[0] for desc in imagesDiscreptors) # number of all keypoints in all images
    descriptors = np.empty((total_desc, imagesDiscreptors[0].shape[1])) # pre-allocate array (number of all keypoints, 128)
    
    # Fill array more efficiently
    idx = 0
    for desc in imagesDiscreptors:
        n_desc = desc.shape[0] # number of keypoints in the current image
        descriptors[idx:idx + n_desc] = desc # fill the array with the keypoints of the current image
        idx += n_desc

    # the result is an array of shape (total number of keypoints, 128) containing all the keypoints of all images
    
    kmeans = KMeans(n_clusters=k) # create KMeans object
    kmeans.fit(descriptors) # fit the KMeans object to the descriptors
    return kmeans


def create_inverted_file(vBowDiscriptors , kMeans):
    
    invertedFile = {}
    for i, descriptors in enumerate(vBowDiscriptors): # loop over all images
        # Predict all clusters at once
        clusters = kMeans.predict(descriptors) # get the cluster of each keypoint in the image
        # Count occurrences of each cluster
        for j, cluster in enumerate(clusters): 

            if cluster not in invertedFile: # if the cluster is not in the inverted file, add it
                invertedFile[cluster] = []

            invertedFile[cluster].append((i,j))  # add the image index and the keypoint index in the image to the cluster in the inverted file
    
    return invertedFile
        

    

    
       
        
    
   