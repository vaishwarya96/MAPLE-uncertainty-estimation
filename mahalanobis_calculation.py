import pickle
import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import scipy
import scipy.stats as stats


from config import get_cfg_defaults
from models.wide_resnet import WideResNet 
from dataset_utils import utils, load_dataset
from inference_utils import md_utils, metrics


cfg = get_cfg_defaults()

checkpoint = cfg.MODEL.CHECKPOINT_DIR

#Load the id map with the class label information
label_dict = pickle.load(open(os.path.join(checkpoint, 'label_dict.pkl'), 'rb'))
n_classes = len(label_dict)

#Load the model
model_path = os.path.join(checkpoint, cfg.MODEL.EXPERIMENT)
model = WideResNet(num_classes = n_classes)
model.classification_layer = nn.Linear(1280, n_classes)
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.eval()

#Load the data
id_map = utils.get_id_map(cfg.DATASET.ID_MAP_PATH)
ood_id_map = utils.get_id_map(cfg.INF.OOD_ID_MAP_PATH)

train_X, train_y = np.load(os.path.join(checkpoint, 'train_img_list.npy')), np.load(os.path.join(checkpoint, 'train_label_list.npy'))
train_data_loader = load_dataset.LoadDataset(train_X, train_y)
train_data = data.DataLoader(train_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

id_X, id_y, _ = utils.get_dataset(cfg.INF.ID_TEST_DATASET, id_map)
id_data_loader = load_dataset.LoadDataset(id_X, id_y)
id_data = data.DataLoader(id_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

ood_X, ood_y, _ = utils.get_dataset(cfg.INF.OOD_TEST_DATASET, ood_id_map)
ood_data_loader = load_dataset.LoadDataset(ood_X, ood_y)
ood_data = data.DataLoader(ood_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)


#Extract train features and perform PCA
train_emb, train_label = md_utils.extract_features(model, train_data)
mean_train_emb = np.mean(np.array(train_emb).T, axis=1)
eigen_values, eigen_vectors, transformed_pts = md_utils.pca(train_emb)
explained_variances = eigen_values / np.sum(eigen_values)
cumsum = np.cumsum(explained_variances)
num_eig = int(np.argwhere(cumsum>cfg.INF.EXP_VAR_THRESHOLD)[0])

print("Number of principal components is %d" %(num_eig))

selected_eig_vectors = eigen_vectors[:,:num_eig]
pc = transformed_pts[:,:num_eig]



###ID analysis###
id_emb, id_label = md_utils.extract_features(model, id_data)
#Convert to PCA frame
transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)

#ID Metrics
pred_class = np.argmin(id_md_matrix, axis=1)
actual_pred_class = [label_dict[k] for k in pred_class]
accuracy = metrics.get_accuracy(actual_pred_class, id_label) 
print("Accuracy: %f" %(accuracy))
ece = metrics.get_expected_calibration_error(id_pred_prob, id_label, num_bins=cfg.INF.NUM_BINS)
print("ECE score: %f" %(ece))

###OOD analysis###
ood_emb, _ = md_utils.extract_features(model, ood_data)
transformed_ood_data = md_utils.transform_features(selected_eig_vectors, ood_emb, mean_train_emb)
ood_md_matrix = md_utils.mahalanobis(transformed_ood_data, pc, train_label)
ood_pred_prob = md_utils.get_md_prob(ood_md_matrix, num_eig)


unc_in = 1 - np.max(id_pred_prob, axis=1)
unc_out = 1 - np.max(ood_pred_prob, axis=1)
auroc = metrics.get_auroc_score(unc_in, unc_out)
print("AUROC: %f" %(auroc))




