# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : autokeras_topics.py
#
#* Purpose :
#
#* Creation Date : 31-07-2019
#
#* Last Modified : Wednesday 31 July 2019 08:29:31 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

from autokeras import TopicClassifier
topic_classifier = TopicClassifier()

class_name = topic_classifier.predict("With some more practice, they will definitely make it to finals..")

