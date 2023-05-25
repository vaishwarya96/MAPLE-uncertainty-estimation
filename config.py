from yacs.config import CfgNode as CN

_C = CN()


###System settings###
_C.SYSTEM = CN()
#Number of workers
_C.SYSTEM.NUM_WORKERS = 4
#Random seed number
_C.SYSTEM.RANDOM_SEED = 1


###Model settings###
_C.MODEL = CN()
#Checkpoint directory path
_C.MODEL.CHECKPOINT_DIR = 'checkpoint/'
#Saved model name
_C.MODEL.EXPERIMENT = 'network.pth'


###Dataset parameters###
_C.DATASET = CN()
#ID map path
_C.DATASET.ID_MAP_PATH = 'PATH TO TRAIN ID CSV FILE'
#Dataset path
_C.DATASET.DATASET_PATH = 'PATH TO TRAIN DATASET' 
#Fraction of data for validation
_C.DATASET.VAL_SIZE = 0.2
#Image size
_C.DATASET.IMG_SIZE = (32,32)
#Image mean and std
_C.DATASET.IMG_MEAN = (0.4914, 0.4822, 0.4465)
_C.DATASET.IMG_STD = (0.2023, 0.1994, 0.2010)


###Train parameters###
_C.TRAIN = CN()
#Batch size
_C.TRAIN.BATCH_SIZE = 64
#Maximum clusters that X-Means can generate for each class
_C.TRAIN_MAX_ALLOWED_CLUSTERS = 5
#Margin value for triplet loss
_C.TRAIN.TRIPLET_LOSS_MARGIN = 0.7
#Number of training epochs
_C.TRAIN.NUM_EPOCHS = 200
#Number of epochs after which validation to be performed
_C.TRAIN.VAL_EPOCHS = 10
#Learning rate
_C.TRAIN.LEARNING_RATE = 0.0002
#False negative ratio
_C.TRAIN.FNR = 0.3
#Maximum number of allowed clusters
_C.TRAIN.MAX_CLUSTERS = 5

###Inference parameters###
_C.INF = CN()
#Path to ID test dataset
_C.INF.ID_TEST_DATASET = 'PATH TO IN-DISTRIBUTION TEST DATASET'
#Path to OOD dataset
_C.INF.OOD_TEST_DATASET = 'PATH TO OUT OF DISTRIBUTION TEST DATASET'
#OOD ID map
_C.INF.OOD_ID_MAP_PATH = 'PATH TO OUT OF DISTRIBUTION ID CSV FILE'
#Threshold on explained variance cumsum
_C.INF.EXP_VAR_THRESHOLD = 0.95
#Number of bins for calibration plot
_C.INF.NUM_BINS = 10



def get_cfg_defaults():

    return _C.clone()

