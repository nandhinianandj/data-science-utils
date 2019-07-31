# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : autokeras_obj_det.py
#
#* Purpose :
#
#* Creation Date : 31-07-2019
#
#* Last Modified : Wednesday 31 July 2019 08:28:20 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

from autokeras.pretrained.object_detector import ObjectDetector
detector = ObjectDetector()

results = detector.predict("/path/to/images/000001.jpg", output_file_path="/path/to/images/")
