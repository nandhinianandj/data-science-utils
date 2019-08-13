# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : autokeras_voice.py
#
#* Purpose :
#
#* Creation Date : 31-07-2019
#
#* Last Modified : Wednesday 31 July 2019 08:31:30 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

from autokeras import VoiceGenerator
voice_generator = VoiceGenerator()
text = "The approximation of pi is 3.14"
# from deepvoice v3
voice_generator.predict(text, "test.wav")
