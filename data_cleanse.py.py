import pickle
import pandas as pd
import csv
import re

pickle_file_name = 'data.pkl'
new_pickle = 'clean_data.pkl'

pickle_file = open(pickle_file_name, 'rb')
objects = []
min_length = 300
max_length = 50000
spam_words = ['ads', 'ad', 'adblock', 'adblocks', 'copyright', 'copyright', 'block', 'blocks']

while True:
    try:
        objects.append(pickle.load(pickle_file))
    except EOFError:
        break
pickle_file.close()

#objects[0].to_csv('dataset.csv')

stopwords = []
# removing English stop words
with open('stopwords.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        stopwords.append(row[0])


# cleans the strings
def clean_text(text_to_clean):
    text_to_clean = text_to_clean.rstrip()  # removing empty spaces at the end of the string
    text_to_clean = re.sub( '[^a-zA-Z0-9]', ' ', text_to_clean) # subs anything non alphanumerical such as Å
    text_to_clean = re.sub( '[0-9]', '', text_to_clean) # removes any numbers in the string
    text_to_clean = re.sub( '\s+', ' ', text_to_clean).strip() # subs tabs, newlines and "whitespace-like"
    words = text_to_clean.lower().split() # convert to lowercase split indv words
    stops = set(stopwords) # converting stop words to set
    spam = set(spam_words) # converting spam words to set
    meaningful_words = [w for w in words if not w in stops and not w in spam] # removing stop words
    return(" ".join(meaningful_words))  # rejoining the word

df = objects[0]

# engineering useful features
df['length'] = df.body_basic.str.len()


# the dataset contains about 270 rows
# the frequency of classes are similar, about a 33% split.
# there is no need to stratify or correct for imbalanced data.

# remove empty sets or fill in label with a basic search
df = df.dropna()

# fixing structural errors
df['body_basic'] = df['body_basic'].apply(lambda x: clean_text(x))

# removing duplicates
df = df.drop_duplicates(subset='body_basic', keep="first", inplace=False, ignore_index=True)

# filtering outliers
df = df.drop(df[df.length < min_length].index)
df = df.drop(df[df.length > max_length].index)

df = df.reset_index(drop=True)

print(df)
final_length = len(df.index)
fishing_length = (df.label == 'fly_fishing').sum()
hockey_length = (df.label == 'ice_hockey').sum()
ml_length = (df.label == 'machine_learning').sum()
print("the final dataset contains " + str(final_length) + " entries.")
print("fly fishing: " + str(fishing_length))
print("hockey: " + str(hockey_length))
print("machine learning: " + str(ml_length))

#df.to_csv('dataset_clean.csv')
#df.drop(columns=['length'])
df.to_pickle(new_pickle)