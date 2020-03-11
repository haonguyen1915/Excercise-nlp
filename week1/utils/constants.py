from nltk.corpus import stopwords
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

# DICT_SIZE = 5000
# VOCAB = words_counts.most_common(DICT_SIZE)
# WORDS_TO_INDEX = {item[0]:ii for ii, item in enumerate(sorted(VOCAB, key=lambda x: x[1], reverse=True))} ####### YOUR CODE HERE #######
# INDEX_TO_WORDS = {ii:word for word, ii in WORDS_TO_INDEX.items()} ####### YOUR CODE HERE #######
# ALL_WORDS = WORDS_TO_INDEX.keys()