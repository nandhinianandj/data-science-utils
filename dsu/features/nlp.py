def spell_correction( text, method='spellchecker',
                      source_words=None, measure='levenshtein'):
    if method=='spellchecker':
        from spellchecker import SpellChecker
        spell = SpellChecker()
        # find those words that may be misspelled
        misspelled = spell.unknown(text)
        result = dict()
        for word in misspelled:
            # Get the one `most likely` answer
            result[word] = spell.correction(word)
        return result
    else:
        assert source_words, measure
        import Levenshtein
        words = text.split()
        result = dict()
        for word in words:
            distances = dict(map(lambda x: (x, Levenshtein.distance(x,word)), source_words))
            closest_words = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])
                    if v > 1}
            result.update({word: closest_words})
        return result

    pass

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

def word_cloud(train_qs):
    from wordcloud import WordCloud
    cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs))
    return cloud

def wordnet_similarity(word1, word2):
    from nltk.corpus import wordnet as wn
    return wn.synset(word1).path_similarity(wn.synset(word2))

def sentiment(words, db='sentiwordnet'):

    if db=='sentiwordnet':
        from nltk.corpus import sentiwordnet as swn
        pscore, nscore = 0.0, 0.0
        for i in range(0,len(words)):
            top_synset = list(swn.senti_synsets(tagged_words[i]))[0]
            if top_synset:
                pscore += top_synset.pos_score()
                nscore += top_synset.neg_score()
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
