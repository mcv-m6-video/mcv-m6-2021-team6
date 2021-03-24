# https://github.com/cheind/py-motmetrics

import numpy as np
import motmetrics as mm
from sklearn.metrics.pairwise import pairwise_distances


class MOTAcumulator:

    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, y_true, y_pred):
        # Calculating center points between X == TRUE and Y == PRED
        # y_true would be in format detectron and Bbox class
        # y_pred is passed in non-class form, for flexibility
        # Detection(frame=frame,
        #           id=None,
        #           label='car',
        #           xtl=float(det[1][0]),
        #           ytl=float(det[1][1]),
        #           xbr=float(det[1][2]),
        #           ybr=float(det[1][3]),
        #           score=det[2]

        X = np.array([[(d.xtl+d.xbr)/2, (d.ytl+d.ybr)/2] for d in y_true])
        '''
        for d in y_pred:
            print(d)

        for d in y_pred:
            print(d[0])
            '''
        # Y = np.array([[(d[0]+d[2])/2, (d[1]+d[3])/2] for d in y_pred])
        Y = np.array([[(d.xtl + d.xbr) / 2, (d.ytl + d.ybr) / 2] for d in y_pred])
        dists = pairwise_distances(X, Y, metric='euclidean')
        self.acc.update([i.id for i in y_true], [i.id for i in y_pred], dists)

    def get_idf1(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['idf1'], name='acc')
        return summary['idf1']['acc']