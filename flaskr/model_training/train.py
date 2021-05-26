import logging
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from datasketch.lsh import MinHashLSH
from datasketch.weighted_minhash import WeightedMinHashGenerator
from flask.logging import default_handler
from gensim.models import Word2Vec
from nltk.corpus import stopwords

root = logging.getLogger()
root.addHandler(default_handler)
stop_words = stopwords.words('english')


def download():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def convert_txt_to_dataframe(file_path):
    ts_pattern = "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3}"

    df_log = pd.DataFrame(
        columns=['timestamp', 'user.id', 'session.id', 'client.ip', 'push.info', 'transaction.id', 'LogLevel',
                 'Activity', 'Module', 'Message'])
    num_row = -1
    with open(file_path, 'r') as f:
        contents = f.readlines()
        for content in contents:
            if re.search(ts_pattern, content) is not None:
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
                df_log.loc[num_row] = [timestamp, values[0], values[1], values[2], values[3], values[4], loglevel,
                                       activity, module, message]

            elif num_row > -1:
                try:
                    # message =content.strip('\n')
                    df_log.loc[num_row, 'Message'] += message
                except Exception as e:
                    root.error(e)
                    root.debug(content)
    return df_log


def data_cleaning(content):
    regex_for = {
        "\d{4}(?:-\d{2}){2}T\d{2}:\d{2}:\d{2}.\d{3}\+\d{4}": 'DATE',
        "\w{3,} \w{3,} \d{2} \d{2}:\d{2}:\d{2} \w{3,} \d{4}": 'DATE',
        "\d+": 'X',
        r'[^\w\s]': ' '
    }

    message = content
    if ' at ' in content:
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


def train_model(file_path):
    df_log = convert_txt_to_dataframe(file_path)
    df_log_msg = df_log[['timestamp', 'LogLevel', 'Message']]
    df_log_msg['Message'] = df_log_msg['Message'].apply(str)
    df_log_msg.reset_index()
    df_log_msg["uniquekey"] = df_log_msg["timestamp"] + " | " + df_log_msg["Message"]

    df_log_msg['processed'] = df_log_msg['Message'].apply(data_cleaning)

    list_dat = []
    index2word_set = set()
    for i in df_log_msg['processed']:
        for k in i:
            list_dat.append(k)
            for f in k:
                index2word_set.add(f)

    # Feeding data to model
    model = Word2Vec(list_dat, min_count=1)
    mg = WeightedMinHashGenerator(100, 128)
    lsh = MinHashLSH(threshold=0.5, num_perm=128)

    df_log_msg[:].apply(lambda row: add_to_lsh_hash(
        row['uniquekey'],
        row['processed'],
        model,
        index2word_set,
        mg,
        lsh),
                        axis=1)

    df = df_log_msg.set_index('uniquekey')

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(index2word_set, open("index2word_set.pkl", "wb"))
    pickle.dump(mg, open("mg.pkl", "wb"))
    pickle.dump(lsh, open("lsh.pkl", "wb"))
    root.debug(df.shape)
    root.debug(df[df['Message'].str.contains('Get device')].shape)


def avg_sentence_vector(sentences, model, num_features, index2word_set):
    # function to average all words vectors in a given paragraph
    feature_vec = np.zeros((num_features,), dtype="float32")

    for sentence in sentences:
        feature_vec_in = np.zeros((num_features,), dtype="float32")
        number_of_words = 0
        for word in sentence:
            if word in index2word_set:
                number_of_words = number_of_words + 1
                feature_vec_in = np.add(feature_vec_in, model.wv[word])
        if number_of_words > 0:
            feature_vec_in = np.divide(feature_vec_in, number_of_words)
        feature_vec = feature_vec + feature_vec_in
    return feature_vec


def add_to_lsh_hash(key, prep, model, index2word_set, mg, lsh):
    try:
        vec1 = avg_sentence_vector(prep, model, 100, index2word_set)
        m1 = mg.minhash(vec1)
        if not lsh.__contains__(key):
            lsh.insert(key, m1)
    except Exception as e:
        root.error(e)


def evaluate_file(file_path, model, index2word_set, mg, lsh):
    # comparing files
    df_log_test = convert_txt_to_dataframe(file_path)
    messages = []

    import time

    start = time.time()

    for index, row in df_log_test.iterrows():
        test_vec = row['Message']
        try:
            vec = avg_sentence_vector(data_cleaning(test_vec), model, 100, index2word_set)
            m = mg.minhash(vec)
            results = lsh.query(m)
            if len(results) == 0:
                messages.append(test_vec)
        except Exception as e:
            root.error(e)
            messages.append("BAD:" + test_vec)

    end = time.time()

    root.debug(str(int(end - start)) + ' Sec')

    root.debug(messages)
    root.debug(df_log_test.shape)

    return messages
