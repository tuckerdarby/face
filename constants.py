_USERNAME = 'ubuntu'
_DATA_LOC = '/home/' + _USERNAME + '/Datasets/vgg_face_dataset/'
LINK_LOC = _DATA_LOC + 'files/'
RECORD_LOC = _DATA_LOC + 'records/'
TEST_RECORD_LOC = _DATA_LOC + 'test_records/'
PEOPLE_LOC = _DATA_LOC + 'test_people/'
TEST_PEOPLE_LOC = _DATA_LOC + 'test_people/'
CHECKPOINT_LOC = 'saves/'
FACE_SIZE = (32, 32)
IMG_SHAPE = FACE_SIZE
EMBEDDING_SIZE = 128

SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'

DM_DATA_LOC = '/home/' + _USERNAME + '/Datasets/dm/'
DM_PEOPLE_LOC = DM_DATA_LOC + 'people/'