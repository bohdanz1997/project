import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

MAX_FEATURES = 3000
MAX_BIGRAMS = 500
MIN_DF = 2
MAX_DF = 0.8

SVM_C = 1.0
SVM_GAMMA = 0.1
SVM_KERNEL = 'rbf'

NN_HIDDEN_SIZE = 100
NN_EPOCHS = 70
NN_BATCH_SIZE = 32
NN_LEARNING_RATE = 0.001

STOP_WORDS_FILE = os.path.join(BASE_DIR, 'stop_words_uk.txt')

CLASSES = ['positive', 'negative', 'neutral']
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}
