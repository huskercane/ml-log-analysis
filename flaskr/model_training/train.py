import pickle
import re

import nltk
import numpy as np
import pandas as pd
from datasketch.lsh import MinHashLSH
from datasketch.weighted_minhash import WeightedMinHashGenerator
from gensim.models import Word2Vec
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

# filepaths = [r'/home/rohits/Downloads/DebugLogFile.log.mod.noleak']

tsPattern = "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3}"

regex_for = {
    "\d{4}(?:-\d{2}){2}T\d{2}:\d{2}:\d{2}.\d{3}\+\d{4}": 'DATE',
    "\w{3,} \w{3,} \d{2} \d{2}:\d{2}:\d{2} \w{3,} \d{4}": 'DATE',
    "\d+": 'X',
    r'[^\w\s]': ' '
}


def download():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def convert_txt_to_dataframe(file_path):
    dfLog = pd.DataFrame(
        columns=['timestamp', 'user.id', 'session.id', 'client.ip', 'push.info', 'transaction.id', 'LogLevel',
                 'Activity', 'Module', 'Message'])
    num_row = -1
    with open(file_path, 'r') as f:
        contents = f.readlines()
        for content in contents:
            if re.search(tsPattern, content) is not None:
                msg = content.split('|')
                stripped = [s.strip() for s in msg]
                timestamp = stripped[0]
                loglevel = stripped[2]
                activity = stripped[3]
                module = stripped[4]

                message = stripped[5].strip('\n')

                client_data = stripped[1].split(' ')
                stripped_client_data = [s.strip('\"') for s in client_data]
                i = 0
                values = []
                for data in stripped_client_data:
                    after_split = data.split('=')
                    if len(after_split) > 1:
                        values.insert(i, after_split[1])
                num_row += 1
                dfLog.loc[num_row] = [timestamp, values[0], values[1], values[2], values[3], values[4], loglevel,
                                      activity, module, message]

            elif num_row > -1:
                try:
                    # message =content.strip('\n')
                    dfLog.loc[num_row, 'Message'] += message
                except:
                    print(content)
    return dfLog


def data_cleaning(content):
    message = content.split('at')[0]
    for rex, replace_value in regex_for.items():
        message = re.sub(rex, replace_value, message)
    sentences = nltk.sent_tokenize(message)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    filtered_sentence = []
    for sentence in sentences:
        output = [w for w in sentence if not w in stop_words]
        filtered_sentence.append(output)
    return filtered_sentence


def train_model(filepaths):
    dfLog = convert_txt_to_dataframe(filepaths)
    dflog_msg = dfLog[['timestamp', 'LogLevel', 'Message']]
    dflog_msg['Message'] = dflog_msg['Message'].apply(str)
    dflog_msg.reset_index()
    dflog_msg["uniquekey"] = dflog_msg["timestamp"] + " | " + dflog_msg["Message"]

    dflog_msg['processed'] = dflog_msg['Message'].apply(data_cleaning)

    list_dat = []
    index2word_set = set()
    for i in dflog_msg['processed']:
        for k in i:
            list_dat.append(k)
            for f in k:
                index2word_set.add(f)

    # Feeding data to model
    model = Word2Vec(list_dat, min_count=1)
    mg = WeightedMinHashGenerator(100, 128)
    lsh = MinHashLSH(threshold=0.5, num_perm=128)

    dflog_msg[:].apply(lambda row: add_to_lshhash(
        row['uniquekey'],
        row['processed'],
        model,
        index2word_set,
        mg,
        lsh),
                       axis=1)

    df = dflog_msg.set_index('uniquekey')

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(index2word_set, open("index2word_set.pkl", "wb"))
    pickle.dump(mg, open("mg.pkl", "wb"))
    pickle.dump(lsh, open("lsh.pkl", "wb"))
    # TODO log these
    # df.shape
    # df[df['Message'].str.contains('Get device')].shape


def avg_sentence_vector(sentences, model, num_features, index2word_set):
    # function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")

    for sentence in sentences:
        featureVecIn = np.zeros((num_features,), dtype="float32")
        nwords = 0
        for word in sentence:
            if word in index2word_set:
                nwords = nwords + 1
                featureVecIn = np.add(featureVecIn, model.wv[word])
        if nwords > 0:
            featureVecIn = np.divide(featureVecIn, nwords)
        featureVec = featureVec + featureVecIn
    return featureVec


def add_to_lshhash(key, prep, model, index2word_set, mg, lsh):
    try:
        vec1 = avg_sentence_vector(prep, model, 100, index2word_set)
        m1 = mg.minhash(vec1)
        if not lsh.__contains__(key):
            lsh.insert(key, m1)
    except:
        print('')


def evaluate_file(file_path, model, index2word_set, mg, lsh):
    # comparing files
    dfLog_test = convert_txt_to_dataframe(file_path)
    messages = set()

    import time

    start = time.time()

    for index, row in dfLog_test.iterrows():
        try:
            test_vec = row['Message']
            vec = avg_sentence_vector(data_cleaning(test_vec), model, 100, index2word_set)
            m = mg.minhash(vec)
            results = lsh.query(m)
            if len(results) == 0:
                messages.add(row['Message'])
        except:
            messages.add(row['Message'])

    end = time.time()

    # TODO: use logger
    print(str(int(end - start)) + ' Sec')

    print(messages)

    dfLog_test.shape

    return messages
