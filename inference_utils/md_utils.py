import numpy as np
import scipy as sp
import torch
from scipy.stats import chi2

def pca(embeddings):
    
    embeddings = np.array(embeddings)
    mean_values = np.mean(embeddings.T, axis=1)
    C = embeddings - mean_values
    covariance_mat = np.cov(C.T)
    eig_val, eig_vec = np.linalg.eig(covariance_mat)
    sorted_idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_idx]
    eig_vec = eig_vec.T[sorted_idx]
    transformed_pts = transform_features(eig_vec, embeddings, mean_values)

    return eig_val, eig_vec, transformed_pts


def transform_features(eig_vec, data_points, mean_values):

    data_points = np.array(data_points)
    C = data_points - mean_values
    transformed_points = eig_vec.T.dot(C.T).T
    return transformed_points

def mahalanobis(x, pc, label):
    label = np.array(label)
    n_classes = np.max(label) + 1
    mean_values = []
    distance_matrix = np.empty((x.shape[0], n_classes))
    num_eig = pc.shape[1]

    for i in range(n_classes):
        index_i = np.argwhere(label==i)
        feats = pc[index_i].squeeze(1)
        mean = np.mean(feats,axis=0)
        x_minus_mu = x - np.mean(feats, axis=0)
        cov = np.cov(pc.T) + np.diag([1e-20]*num_eig)
        inv_cov = sp.linalg.inv(cov)
        left_term = np.dot(x_minus_mu, inv_cov)
        mahal = np.dot(left_term, x_minus_mu.T)
        if isinstance(mahal, np.float):
            dist = mahal
        else:
            dist = mahal.diagonal()

        dist = np.sqrt(dist)
        distance_matrix[:,i] = dist
    
    return distance_matrix

def extract_features(model, dataset):
    label_database = []
    emb_database = []
    path_database = []

    with torch.no_grad():
        for it, (img, label, _) in enumerate(dataset):
            
            b_images = img.cuda()
            b_labels = label.cuda()

            emb, logits = model(b_images)
            label_database.extend(label.detach().cpu().numpy())
            emb_database.extend(emb.detach().cpu().numpy())

    return emb_database, label_database

def get_md_prob(mahalanobis_distance, num_eig):
    p_values = 1-chi2.cdf(mahalanobis_distance**2, num_eig)
    return p_values
