
def spell_correction(text):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    # find those words that may be misspelled
    misspelled = spell.unknown(text)
    result = dict()
    for word in misspelled:
        # Get the one `most likely` answer
        print(word, spell.candidates(word))
        result[word] = spell.correction(word)
    return result

def filter_letters_only(text):
    import re

    # Use regular expressions to do a find-and-replace
    return re.sub("[^a-zA-Z]", " ", text).lower()


def filter_stop_words(text, lang="english"):
    from nltk.corpus import stopwords

    stops = set(stopwords.words(lang))
    return set(text) - set(stops)


def bag_of_words(text):
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=5000,
    )
    vectorizer.fit_transform(text)
    return vectorizer


def word_match_share(row, lang="english"):
    from nltk.corpus import stopwords

    stops = set(stopwords.words(lang))
    q1words = {}
    q2words = {}
    for word in str(row["question1"]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row["question2"]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (
        len(q1words) + len(q2words)
    )
    return R


def word_2_vector(sentences, size=200, **kwargs):
    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    from gensim.models import word2vec

    model = word2vec.Word2Vec(size=size, **kwargs)
    model.build_vocab(sentences)
    model.train(sentences)
    return model


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.0
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.0
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(text, model, num_features):
    # Given a set of text (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.0
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(text), num_features), dtype="float32")
    #
    # Loop through the text
    for review in text:
        #
        # Print a status message every 1000th review
        if counter % 1000.0 == 0.0:
            print("Review %d of %d" % (counter, len(text)))
        #
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        #
        # Increment the counter
        counter = counter + 1.0
    return reviewFeatureVecs


def word_cloud(train_qs):
    from wordcloud import WordCloud

    cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs))
    return cloud

def word_similarity(word1, word2):
    from nltk.corpus import wordnet as wn

    return wn.synset(word1).path_similarity(wn.synset(word2))

def sentiment(tagged_words, db='sentiwordnet'):
    if db=='sentiwordnet':
        pscore, nscore = 0.0, 0.0
        from nltk.corpus import sentiwordnet as swn
        for i in range(0,len(tagged_words)):
            if 'NN' in tagged_words[i][1] and len(list(swn.senti_synsets(tagged_words[i][0],'n')))>0:
                    pscore+=(list(swn.senti_synsets(tagged_words[i][0],'n'))[0]).pos_score() #positive score of a word
                    nscore+=(list(swn.senti_synsets(tagged_words[i][0],'n'))[0]).neg_score()  #negative score of a word
            elif 'VB' in tagged_words[i][1] and len(list(swn.senti_synsets(tagged_words[i][0],'v')))>0:
                   pscore+=(list(swn.senti_synsets(tagged_words[i][0],'v'))[0]).pos_score()
                   nscore+=(list(swn.senti_synsets(tagged_words[i][0],'v'))[0]).neg_score()
            elif 'JJ' in tagged_words[i][1] and len(list(swn.senti_synsets(tagged_words[i][0],'a')))>0:
                   pscore+=(list(swn.senti_synsets(tagged_words[i][0],'a'))[0]).pos_score()
                   nscore+=(list(swn.senti_synsets(tagged_words[i][0],'a'))[0]).neg_score()
            elif 'RB' in tagged_words[i][1] and len(list(swn.senti_synsets(tagged_words[i][0],'r')))>0:
                   pscore+=(list(swn.senti_synsets(tagged_words[i][0],'r'))[0]).pos_score()
                   nscore+=(list(swn.senti_synsets(tagged_words[i][0],'r'))[0]).neg_score()
            return (pscore, nscore)
    #elif db == 'sentiwords':

    else:
        raise NotImplementedError

def tfidf_model(corpus, gensim=False):
    if gensim:
        from gensim.models import tfidfmodel

        model = tfidfmodel.TfidfModel(corpus)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        model = TfidfVectorizer()
    return model
