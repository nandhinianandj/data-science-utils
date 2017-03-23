def filter_letters_only(text):
    import re
    # Use regular expressions to do a find-and-replace
    return re.sub("[^a-zA-Z]", " ", text).lower()

def filter_stop_words(text, lang='english'):
    from nltk.corpus import stopwords
    stops = set(stopwords.words(lang))
    return set(text) -set(stops)

def word_match_share(row, lang='english'):
    from nltk.corpus import stopwords
    stops = set(stopwords.words(lang))
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def word_2_vector(sentences, size=200, **kwargs):
    from gensim.models import word2vec
    model = word2vec.Word2Vec(size=size, **kwargs)
    model.build_vocab(sentences)
    model.train(sentences)
    return model

def word_cloud(train_qs):
    from wordcloud import WordCloud
    cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
    plotter.show_image(cloud, figsize=(20,15))

def word_similarity(word1, word2):
    from nltk.corpus import wordnet as wn
    return wn.synset(word1).path_similarity(wn.synset(word2))

def tfidf_model(corpus, gensim=False):
    if gensim:
        from gensim.models import tfidfmodel
        model = tfidfmodel.TfidfModel(corpus)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        model = TfidfVectorizer()
    return model
