from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn import preprocessing
import re
import math
import itertools as it
import functools

import pd_helpers as pdH

class Predictor(ABC):

    def __init__(self, expt, data):
        self._expt = expt
        self._data = data
        self._values = self._calculate()

    @abstractmethod
    def _calculate(self):
        pass

    @property
    def values(self):
        #TODO: store predictors as DF
        try:
            return self._values.to_frame()
        except:
            return self._values

    @values.setter
    def values(self, value):
        raise Exception('Predictor values cannot be set')

    def reindex(self, index):
        self._values = self._values.reindex(index)


class Lag(Predictor):
    def __init__(self, expt, data, cue, dt, n, offset):
        data_dt = np.round(
            data.index.unique(
                'trial_time').to_series().diff().value_counts().idxmax(),
            decimals=1)
        self._cue = cue
        self._dt = data_dt / dt
        self._n = n
        self._lags = np.arange(
            offset, self._n * self._dt + offset, self._dt).astype(int)
        super().__init__(expt, data)

    def _calculate(self):
        def lagger(x, lags):
            def alag(x, offset):
                if offset > 0:
                    return x[:-offset].set_index(
                        x[offset:].index).reindex(x.index).fillna(False)
                elif offset < 0:
                    return x[-offset:].set_index(
                        x[:offset].index).reindex(x.index).fillna(False)
                else:
                    return x
            return pd.concat(
                list(map(functools.partial(alag, x.droplevel(0)), lags)),
                axis=1)

        result = self._cue.values.groupby('trial').apply(
            lagger, lags=self._lags)
        result.columns = [
            '_'.join([x[0], str(x[1])]) for x in
            zip(result.columns, self._lags)]
        return result


class OdorOff(Predictor):

    def _calculate(self):
        feature = ((self._data['Odor'] > 0).astype(int).diff() < 0)
        feature.name = 'OdorOff'
        return feature

class LongOdorOff(Predictor):

    def _calculate(self):
        def find_long(x):
            res = pd.Series(0, x.index, name=x.name)
            if x.index.unique('type')[0] == 'LS':
                res[x[(x.diff() < 0).cumsum() == 1].head(1).index] = 1
            elif x.index.unique('type')[0] == 'SL':
                res[x[(x.diff() < 0).cumsum() == 2].head(1).index] = 1
            return res.droplevel(0)
        return (self._data['Odor'] > 0).astype(int).groupby('trial').apply(find_long)

class NotShort(Predictor):

    def _calculate(self):
        threshold = None
        odor_times = self._expt.odors_demixed
        ntrials = len(odor_times)

        odor_times = self._expt.odors_demixed
        ntrials = len(odor_times)
        odor_lengths = odor_times.applymap(lambda x: x[1] - x[0])
        masked_times = self._data.reset_index(
            'trial_time', drop=False)['trial_time'].diff().mask(
                ~self._data['Odor'].values)

        masked_times.index = self._data.index
        cumsum = masked_times.cumsum().fillna(method='pad')
        reset = -cumsum[masked_times.isnull()].diff().fillna(cumsum)
        result = masked_times.where(masked_times.notnull(), reset).cumsum()

        if threshold is None:
            splitter_trial = result.groupby('trial').max().sort_values().diff().shift(periods=-1).idxmax()
            threshold = result.groupby('trial').max()[splitter_trial]

        seen_nshort = ((result > threshold).groupby('trial').cumsum() > 0)
        seen_nshort.name = 'not_short'
        return seen_nshort


class Long(Predictor):

    def _calculate(self):
        threshold=None

        odor_times = self._expt.odors_demixed
        ntrials = len(odor_times)
        odor_lengths = odor_times.applymap(lambda x: x[1] - x[0])

        if threshold is None:
            splitter_trial = odor_lengths.stack().sort_values().diff().idxmax()
            threshold = odor_lengths[splitter_trial[1]][splitter_trial[0]]

        long_off = odor_times.mask(
            ~(odor_lengths >= threshold)
            ).dropna(how='all').stack().groupby('trial').head(1).apply(
                lambda x: x[1])

        trial_start_times = self._expt.behavior_data.reset_index(
            'Time')['Time'].groupby('trial').head(1)
        off_trial_times = (long_off - trial_start_times).apply(np.round, decimals=1)
        times = self._data.reset_index(level='trial_time')['trial_time'].droplevel([1, 2])
        trial_length = times.max()
        long_off_intervals = off_trial_times.apply(
            lambda x, max_length: (x, max_length), max_length=trial_length)
        long_off_intervals = long_off_intervals.reindex(self._data.index.unique('trial'), level='trial')
        long_off_intervals.index = \
            long_off_intervals.index.get_level_values('trial')
        seen_long = long_off_intervals.to_frame().apply(
            lambda x, times=times: pd.Series(
                True, index=pd.IntervalIndex.from_tuples(x.values)
                ).reindex(times[x.name]), axis=1)
        seen_long = seen_long.reindex(self._data.index.unique('trial')).fillna(False).stack()
        seen_long = seen_long.reindex(self._data.index.droplevel([2, 3]))
        seen_long.index = self._data.index
        seen_long.name = 'long'
        return seen_long


class CueChange(Predictor):
    def __init__(self, expt, data, cue, state, offset=0):
        self._cue = cue
        self._state = state
        self._offset = offset

        super().__init__(expt, data)

    def _calculate(self):
        if self._state not in ('off', 'on'):
            raise Exception('Unclear cue state transition')

        if self._cue == 'Odor1':
            cue_idx = 1
        elif self._cue == 'Odor2':
            cue_idx = 2

        if self._state == 'off':
            res = (
                self._data['Odor'].astype(int).diff() < 0
                ).groupby('trial').cumsum() == cue_idx

        elif self._state == 'on':
            res = (
                self._data['Odor'].astype(int).diff() > 0
                ).groupby('trial').cumsum() == cue_idx

        fixed_index = np.round(res.index.get_level_values('trial_time') + self._offset, 1)
        res.index = pdH.update_level(res.index, fixed_index, 'trial_time')
        res = res.reindex(self._data.index).fillna(method='ffill').fillna(method='bfill')

        if False:
            start_times = self._expt.trial_start_times
            odor2_endtimes = self._expt.odors_demixed[self._cue].apply(
                lambda x: x[self._state == 'off']) + self._offset
            odor2_trialendtimes = odor2_endtimes - start_times.values

            trial_times = self._data.index.get_level_values(
                'trial_time').to_series(index=self._data.index)
            cue_times = odor2_trialendtimes.apply(np.round, decimals=1).reindex(
                self._data.index.get_level_values('trial'))
            cue_times.index = self._data.index
            res = trial_times > cue_times

        res.name = f'{self._cue}_{self._state}'
        return res


class CueChangeDiff(CueChange):
    def _calculate(self):
        res = super(self.__class__, self)._calculate()
        return res.astype(int).diff() > 0

class CueChangeTimebins(CueChange):
    def __init__(self, expt, data, cue, state, winlength=None, offset=0):
        self._winlength = winlength
        self._offset = offset

        super().__init__(expt, data, cue, state, offset)

    def _calculate(self):
        res = super(self.__class__, self)._calculate()

        res_labeled = res.groupby(['trial']).apply(
                lambda x: (x.astype(int) *
                           x.index.get_level_values(
                                'trial_time').to_series().diff()
                          ).cumsum().droplevel(0))
        res_labeled = res_labeled.fillna(0).apply(np.round, decimals=1)
        if self._winlength is not None:
            res_labeled[res_labeled > self._winlength] = 0

        enc = preprocessing.OneHotEncoder()
        one_hot = enc.fit_transform(
            res_labeled.values.reshape(1, -1).T)
        columns = [f'{x}' for x in sorted(res_labeled.unique())]
        one_hot = pd.DataFrame(
            one_hot.toarray(), index=res_labeled.index, columns=enc.categories_)
        one_hot = one_hot.drop(0, axis=1)

        window = 10
        std = 2
        one_hot = one_hot.groupby('trial').apply(
                lambda x, window=window, std=std: x.rolling(
                    window, center=True, win_type='gaussian', min_periods=1
                    ).mean(std=std)).droplevel(0)

        one_hot.columns = [
            f'{self._cue}_{self._state}_' + str(a[0]) for a  in one_hot.columns]
        one_hot = one_hot.multiply(res.astype(int), axis=0)
        return one_hot/one_hot.max(axis=0)


class CueAnchor(CueChange):
    def __init__(self, expt, data, cue, state='off', offset=0):
        self._cue = cue
        super().__init__(expt, data, cue, state, 0)

    def _calculate(self):
        #TODO: dt and decimals should be interpreted from data
        dt = 0.1

        max_offset = np.round(
            self._expt.odors_demixed[self._cue].map(lambda x: x[1] - x[0]).max(), decimals=1)
        self._offset = -max_offset
        self._state = 'off'
        cue2_offtimes = super(self.__class__, self)._calculate()
        time_labels = cue2_offtimes.groupby(['trial']).cumsum()
        time_labels = time_labels.mask(time_labels == 0)
        time_labels = time_labels - int(np.round(max_offset/dt))

        self._state = 'on'
        self._offset = 0
        cue2_on = super(self.__class__, self)._calculate()
        time_labels = time_labels.mask(~cue2_on).dropna()

        enc = preprocessing.OneHotEncoder()
        one_hot = enc.fit_transform(
            time_labels.values.reshape(1, -1).T)
        one_hot = pd.DataFrame(
            one_hot.toarray(), index=time_labels.index, columns=enc.categories_)

        window = 10
        std = 2
        one_hot = one_hot.groupby('trial').apply(
                lambda x, window=window, std=std: x.rolling(
                    window, center=True, win_type='gaussian', min_periods=1
                    ).mean(std=std)).droplevel(0)

        one_hot.columns = [f'{self._cue}_anchor_' + str(a[0]) for a  in one_hot.columns]
        one_hot = one_hot.multiply(cue2_on.astype(int), axis=0)
        return (one_hot/one_hot.max(axis=0)).fillna(0)


class CueComparison(CueChange):
    def __init__(self, expt, data):
        super(self.__class__, self).__init__(expt, data, 'Odor2', 'off', 0)

    def _calculate(self):
        cue_time_diffs = self._expt.odors_demixed.applymap(
            lambda x: x[1] - x[0]).diff(axis=1).dropna(axis=1)
        splitter_trial = cue_time_diffs['Odor2'].abs().sort_values().diff().idxmax()
        nonmatch_trials = cue_time_diffs.abs() >= cue_time_diffs.xs(
            splitter_trial, axis=0).abs()
        odor2_off =  super(self.__class__, self)._calculate()
        return odor2_off.reindex(
            nonmatch_trials[nonmatch_trials['Odor2']].index, level='trial').reindex(
                odor2_off.index).fillna(False)


class LongTrial(CueChange):
    def __init__(self, expt, data, threshold=None):
        self._threshold = threshold
        super(LongTrial, self).__init__(expt, data, 'Odor2', 'off', 0)

    def _calculate(self):
        cue2_off = super(self.__class__, self)._calculate()

        if self._threshold is None:
            trial_lengths = (1 - cue2_off).groupby('trial').sum()
            splitter_trial = trial_lengths.sort_values().diff().idxmax()
            threshold = cue2_off.index.get_level_values('trial_time')[
                trial_lengths[splitter_trial]]
        else:
            threshold = self._threshold

        transition_times = cue2_off.mask(
            ~(cue2_off.astype(int).diff() == 1).values).dropna().reset_index(
                level='trial_time')['trial_time']
        long_trials = cue2_off.reindex(
            transition_times[transition_times >= threshold].index.get_level_values(
                'trial'),
            level='trial')
        length_cue = long_trials & \
                     (long_trials.reset_index('trial_time')['trial_time'] >
                      threshold).values
        length_cue = length_cue.reindex(cue2_off.index).fillna(False)
        length_cue.name = 'trial_length'
        return length_cue


class LongTrialDiff(Predictor):
    def __init__(self, expt, data, threshold=None):
        self._long_trial = LongTrial(expt, data, threshold)
        super(self.__class__, self).__init__(expt, data)

    def _calculate(self):
        return (self._long_trial.values.astype(int).diff() > 0)


class NotShortTrial(CueChange):
    def __init__(self, expt, data, threshold=None):
        if threshold is not None:
            threshold = float(threshold)
        self._threshold = threshold
        super(self.__class__, self).__init__(expt, data, 'Odor2', 'off', 0)

    def _calculate(self):
        cue2_off = super(self.__class__, self)._calculate()

        if self._threshold is None:
            if 'ISI_type' in self._data.index.names:
                trial_lengths = (1 - cue2_off.xs('N', level='ISI_type')).groupby('trial').sum()
            else:
                trial_lengths = (1 - cue2_off).groupby('trial').sum()
            splitter_trial = trial_lengths.sort_values(
                ascending=False).diff().idxmin()
            threshold = cue2_off.index.get_level_values('trial_time')[
                trial_lengths[splitter_trial]]
            print(threshold)
        else:
            threshold = self._threshold

        transition_times = cue2_off.mask(
            ~(cue2_off.astype(int).diff() == 1).values).dropna().reset_index(
                level='trial_time')['trial_time']
        not_short_times = (self._data.reset_index('trial_time')['trial_time'] > threshold)
        transition_times_idx = transition_times.droplevel(
            transition_times.index.names[1:]).reindex(
                self._data.index.get_level_values('trial'))
        not_short_trial = (not_short_times * transition_times > threshold)
        not_short_trial.index = self._data.reindex(not_short_trial.index.unique('trial'), level='trial').index
        not_short_trial.name = 'not_short_trial'
        return not_short_trial

class NotShortTrialDiff(Predictor):
    def __init__(self, expt, data, threshold=None):
        self._not_short_trial = NotShortTrial(expt, data, threshold)
        super(self.__class__, self).__init__(expt, data)

    def _calculate(self):
        return (self._not_short_trial.values.astype(int).diff() > 0)



class PredictorFactory:
    @classmethod
    def create(cls, expt, data, name):
        print(name)
        args = name.split('_')
        return globals()[args[0]](expt, data, *args[1:])

    @classmethod
    def interactions(cls, *features):
        list(map(lambda x, idx=features[0].values.index: x.reindex(idx),
                 features[1:]))

        interactions = np.array(list(
            map(math.prod, it.product(*[x.values.values.T for x in features])))).T
        interactions_columns = np.array(list(
            map(' x '.join, it.product(*[x.values.columns for x in features]))))
        interactions = pd.DataFrame(interactions, index=features[0].values.index)
        interactions.columns = interactions_columns
        return interactions

    @classmethod
    def parse(cls, string, expt, data):
        X = pd.DataFrame(None, index=data.index)
        features = string.split(' + ')
        for feature in features:
            interactions = feature.split(' x ')
            unmixed_features = []
            for x in interactions:
                if x == 'time_bins':
                    raise Exception('Not Implemented')
                    unmixed_features.append(time_bins(expt, data))
                    continue

                lag_test = re.search('^Lag\(([^\)(]+)\)', x)
                if lag_test is not None:
                    x = lag_test.groups()[0].split(',')[0]

                category_test = re.search('^C\(([^\)(]+)\)', x)
                if category_test is not None:
                    x = category_test.groups()[0]

                if x in data.columns:
                    feature_temp = data[x]
                else:
                    feature_temp = cls.create(expt, data, x)

                if category_test is not None:
                    enc = preprocessing.OneHotEncoder()
                    name = feature_temp.name
                    feature_temp = pd.DataFrame(
                        enc.fit_transform(
                            feature_temp.values.reshape(-1, 1)).toarray(),
                        index=data.index)
                    feature_temp.columns = [
                        f'{a}_{str(b)}' for a, b in
                        zip(it.repeat(name), feature_temp.columns)]

                if lag_test is not None:
                    lag_args = lag_test.groups()[0].split(',')
                    feature_temp = Lag(
                        expt, data, feature_temp, *map(float, lag_args[1:]))

                unmixed_features.append(feature_temp)
            if len(interactions) > 1:
                unmixed_features = [
                    x.to_frame() if type(x) == pd.Series else x for x in
                    unmixed_features]
                X = pd.concat((X, cls.interactions(*unmixed_features)), axis=1)
            else:
                X = pd.concat(
                    (X, *list(map(lambda x: x.values, unmixed_features))),
                    axis=1)
        X  = X[X.columns[~X.T.duplicated()]]
        print(X.shape)
        return X
