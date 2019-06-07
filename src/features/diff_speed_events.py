import sys
import os
sys.path.append(os.getcwd())
from src.features.feature_base import FeatureBase
from src import data
import pandas as pd
from src.utils.datetime_converter import convert_to_datetime
from src import utility
import src.utils.folder as folder
from src.utils import *
import src.data as data

# OK
class DiffSpeedEvents(FeatureBase):
    
    def __init__(self, mode):
        name = 'diff_speed_events'
        super(DiffSpeedEvents, self).__init__(
            name=name, mode=mode)

    def extract_feature(self):
        speeds = None

        if self.mode == 'local':
            tr = data.speeds_original('train')
            te = data.speed_test_masked()
            speeds = pd.concat([tr, te])
            del tr
            del te
        
        elif self.mode == 'full':
            tr = data.speeds(mode='full')
            te = data.speeds_original('test2')
            speeds = pd.concat([tr, te])
            del tr
            del te

        etr = data.events(self.mode, 'train')
        ete = data.events(self.mode, 'test')
        ef = pd.concat([etr, ete])
        del etr
        del ete

        t = ef[['START_DATETIME_UTC', 'END_DATETIME_UTC', 'KEY', 'KM_START', \
        'KM_END', 'DATETIME_UTC', 'EVENT_TYPE', 'EVENT_DETAIL']]
        t = t.loc[t.groupby(['START_DATETIME_UTC', 'END_DATETIME_UTC', 'KEY'], as_index=False).DATETIME_UTC.idxmin()]
        t['DATETIME_UTC-1'] = t.DATETIME_UTC - pd.Timedelta(minutes=15)
        t = t.drop(['START_DATETIME_UTC', 'END_DATETIME_UTC', 'DATETIME_UTC'], axis=1)
        speeds = speeds[['KEY', 'KM', 'DATETIME_UTC', 'SPEED_AVG']]

        final = pd.merge(t, speeds, left_on=['KEY', 'DATETIME_UTC-1'], right_on=['KEY', 'DATETIME_UTC'])
        final = final.rename(columns={'SPEED_AVG': 'speed_avg-1', 'KM': 'KM-1'})
        final = final.drop(['DATETIME_UTC'], axis=1)
        final = final[(final['KM-1'] >= final.KM_START) & (final['KM-1'] <= final.KM_END)]

        ds = []
        for ts in range(4):
            m_ = t.copy()
            print(len(m_))
            quarters_delta = ts+1
            m_['DATETIME_UTC_{}'.format(quarters_delta)] = m_['DATETIME_UTC-1'] + pd.Timedelta(minutes=15*quarters_delta)
            m_ = pd.merge(m_, speeds, \
                        left_on=['KEY', 'DATETIME_UTC_{}'.format(quarters_delta)], \
                        right_on=['KEY', 'DATETIME_UTC'], how='left')
            m_ = m_.rename(columns={'SPEED_AVG': 'speed_avg_{}'.format(quarters_delta), \
                                    'KM': 'KM_{}'.format(quarters_delta)})
            m_ = m_.drop(['DATETIME_UTC'], axis=1)
            m_ = m_[(m_['KM_{}'.format(quarters_delta)] >= m_.KM_START) & (m_['KM_{}'.format(quarters_delta)] <= m_.KM_END)]
            m_ = m_.rename(columns={'KM_{}'.format(quarters_delta): 'KM-1'})
            m_ = m_.drop(['DATETIME_UTC_{}'.format(quarters_delta)], axis=1)
            print(len(m_))
            ds.append(m_)

        final = final.drop(['KM-1'], axis=1)
        for i in range(len(ds)):
            df = ds[i]
            j = i+1
            print('shape before {}'.format(len(final)))
            final = pd.merge(final, df)
            print('shape after {}'.format(len(final)))

        final = final[['EVENT_TYPE', 'speed_avg-1', 'speed_avg_1', 'speed_avg_2', 'speed_avg_3', 'speed_avg_4']]
        final['diff-1-step'] = final['speed_avg_1'] - final['speed_avg-1']
        final['diff-2-step'] = final['speed_avg_2'] - final['speed_avg-1']
        final['diff-3-step'] = final['speed_avg_3'] - final['speed_avg-1']
        final['diff-4-step'] = final['speed_avg_4'] - final['speed_avg-1']
        final = final.drop(['speed_avg_1', 'speed_avg-1', 'speed_avg_2', 'speed_avg_3', 'speed_avg_4'], axis=1)
        return final.groupby(['EVENT_TYPE'], as_index=False).mean()

if __name__ == '__main__':
    from src.utils.menu import mode_selection
    mode = mode_selection()
    c = DiffSpeedEvents(mode)

    print('Creating {}'.format(c.name))
    c.save_feature()

    print(c.read_feature())

