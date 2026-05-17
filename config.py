"""
Configuration file - centralizes all hyperparameters and constants used in the project.
Aligns with the description in Chapter 3 and Chapter 4 of the paper.
"""
import os

# ============== Data Set Configuration ==============
# Moore dataset file names (10 subsets, see Section 3.1.1 of the paper)
MOORE_FILES = [
    'entry01.weka.allclass.arff',
    'entry02.weka.allclass.arff',
    'entry03.weka.allclass.arff',
    'entry04.weka.allclass.arff',
    'entry05.weka.allclass.arff',
    'entry06.weka.allclass.arff',
    'entry07.weka.allclass.arff',
    'entry08.weka.allclass.arff',
    'entry09.weka.allclass.arff',
    'entry10.weka.allclass.arff',
]

# Dataset path (place the Moore dataset under ./data/moore/)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'moore')

# 12 traffic classification labels (see Section 3.1.1 of the paper)
LIST_Y = [
    'WWW', 'MAIL', 'FTP-CONTROL_', 'FTP-PASV',
    'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA',
    'MULTIMEDIA', 'SERVICES', 'INTERACTIVE_',
    'GAMES'
]

NUM_CLASSES = len(LIST_Y)

# ARFF file header line count (Moore dataset arff first 253 lines are attribute declarations, data starts at line 254)
ARFF_HEADER_LINES = 253

# Original feature count (Moore dataset has 248 features per sample)
ORIGINAL_FEATURE_NUM = 248

# Number of zero padding (pad 248 features to 256 with 8 zeros, see Section 4.3 of the paper)
PADDING_NUM = 8

# Total feature count (256 = 16*16, enables reshaping to 16x16 for CNN)
TOTAL_FEATURES = ORIGINAL_FEATURE_NUM + PADDING_NUM  # 256

# CNN input window size
CNN_INPUT_SHAPE = (16, 16, 1)

# Number of question marks threshold: skip the sample if it has more than this many '?'
MAX_QUESTION_MARKS = 8

# ============== Training Configuration ==============
TEST_SIZE = 0.25
RANDOM_STATE = 0
EPOCHS = 25
BATCH_SIZE = 128

# ============== Output Configuration ==============
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== Class Balancing Configuration ==============
# Sample count after balancing for each class (see Section 3.1.2 Table 3.2 of the paper)
BALANCED_COUNTS = {
    'WWW': 65000,
    'MAIL': 65000,
    'BULK': 30000,         # FTP-CONTROL + FTP-PASV + FTP-DATA
    'DATABASE': 30000,
    'SERVICES': 30000,
    'P2P': 30000,
    'ATTACK': 20000,
    'MULTIMEDIA': 10000,
    'INTERACTIVE_': 10000,
}

# Classes that need to be augmented with Gaussian white noise (according to Section 3.1.3 of the paper)
NOISE_AUGMENT_CLASSES = ['MULTIMEDIA', 'P2P', 'DATABASE']
NOISE_MEAN = 0.0
NOISE_STD = 1.0
