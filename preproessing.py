import pickle
from scipy import signal


raw_data_dict = pickle.load(open('./dataset/all_32.dat','rb'),encoding='latin1')
fs = 128   # Sampling rate (128 Hz)
win = 0.5 * sf


participants = range(1,33)
videos = range(0,40)
channels = range(1,33)

# construct feature vectors
for person in participants:
	X = [] # make full dumb
	y = [] # put all ratings, so we can subset laters
	for vid in videos:
		channels_data = (((raw_data_dict[person])['data'])[vid])[:32]
		ratings = ((raw_data_dict[person]['labels'])[vid])
		y.append(ratings) # append video ratingS to labels
