from os.path import dirname, abspath

BASE_DIR = dirname(dirname(abspath(__file__))) + '/'

# path to configs
CONFIG_DIR = BASE_DIR + 'configs/'

# paths to data
DATA_DIR = BASE_DIR + 'data/'
RAW_DATA_DIR = DATA_DIR + 'raw/'
TEST_DATA_DIR = DATA_DIR + 'test/'
INT_DATA_DIR = DATA_DIR + 'intermediate/'
FINAL_DATA_DIR = DATA_DIR + 'final/'

# path to logs
LOGS_DIR = BASE_DIR + 'logs/'