from model.visil import ViSiL
from datasets import load_video
import sys

# Load the two videos from the video files
print('examples/' + sys.argv[1] + '.json')

query_video = load_video('examples/' + sys.argv[1] + '.gif')
target_video = load_video('examples/' + sys.argv[2] + '.gif')

# Initialize ViSiL model and load pre-trained weights
model = ViSiL('ckpt/resnet/')

# Extract features of the two videos
query_features = model.extract_features(query_video, batch_sz=32)
target_features = model.extract_features(target_video, batch_sz=32)

# Calculate similarity between the two videos
similarity = model.calculate_video_similarity(query_features, target_features)
print("SIMILARITY: " + str(similarity * 100) + "%")
