import os
import pandas as pd
import numpy as np
import functools
import hashlib
import json
import pandas as pd
import copy
import warnings
import sklearn.linear_model

import database
import pd_helpers as pdH
import itertools as it


class Experiment:

    @classmethod
    def expt_types(cls):
        return []

    @classmethod
    def register_types(cls):
        afunc = lambda x, func: [[a] + func(a, func) for a in
                                 x.__subclasses__()]
        lst = afunc(cls, func=afunc)

        def flatten_list(nested_list):
            flattened_list = []
            for item in nested_list:
                if isinstance(item, list):
                    flattened_list.extend(flatten_list(item))
                else:
                    flattened_list.append(item)
            return flattened_list

        res = flatten_list(lst)
        expt_classes = {}
        for a_cls in res:
            new_classes = {k:v for k, v in
                           zip(a_cls.expt_types(), it.repeat(a_cls))}
            if len(set(new_classes.keys()).intersection(
                    set(expt_classes.keys()))):
                collision = set(new_classes.keys()).intersection(
                    set(expt_classes.keys()))
                raise AttributeError(
                    f'experiment_type {collision} already associated with ' +
                     'Experiment Type')

            expt_classes.update(new_classes)
        cls._expt_classes = expt_classes


    @classmethod
    def create(cls, trial_id):
        expt_info = database.fetch_trial(int(trial_id))

        try:
            ExperimentType = cls._expt_classes.get(
                (expt_info['experiment_type']), cls)
        except AttributeError:
            cls.register_types()
            ExperimentType = cls._expt_classes.get(
                (expt_info['experiment_type']), cls)

        return ExperimentType(int(trial_id))


    def __init__(self, trial_id):
        self.attr = database.fetch_trial(trial_id)
        if self.attr is None:
            raise KeyError(f"Trial ID {trial_id} does not exist")

        self.attr['mouse_name'] = database.fetch_mouse(self.attr['mouse_id'])
        self.attr['project_name'] = database.fetch_project(
            self.attr['project_id'])

        self.attr.update(database.fetch_all_trial_attrs(self.trial_id,
                                                        parse=True))

        self._attr = copy.deepcopy(self.attr)

    def __repr__(self):
        return f"<{type(self).__name__}: trial_id={self.trial_id} " \
               + f"mouse={self.mouse_name} " \
               + f"project={self.project_name}>"

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, item):
        if item == 'attr' or item[0] == '_':
            return self.__dict__[item]

        return self.attr[item]

    def __setattr__(self, item, value):
        if item == 'attr' or item[0] == '_':
            self.__dict__[item] = value
        else:
            self.attr[item] = value

    def __delattr__(self, item):
        if '_' + item in self.__dict__.keys():
            raise Exception(f'{item} is not settable')

        if item in self._attr:
            self.attr[item] = None
        else:
            del self.attr[item]

    def get(self, item, default=None):
        return self.attr.get(item, default)

    def save(self, store=False):
        updates = {k: v for k, v in self.attr.items() if
                   (k not in self._attr.keys() and
                   (v is not None and k is not None)) or
                   v != self._attr.get(k)}

        print(f'changes to {self.trial_id}: {updates}')
        trial_args = database.ExperimentDatabase().table_columns('trials')
        trial_args.remove('mouse_id')
        trial_args.remove('project_id')
        trial_args.append('mouse_name')
        trial_args.append('project_name')

        if 'trial_id' in updates.keys():
            raise Exception('Changes to trial ID not permitted')

        if store:
            database.update_trial(**{k: self.get(k) for k in trial_args})

            for key, value in updates.items():
                if key in trial_args:
                    continue

                if value is None:
                    database.delete_trial_attr(self.trial_id, key)
                else:
                    if isinstance(value, dict) or isinstance(value, list):
                        value = json.dumps(value)
                    database.update_trial_attr(self.trial_id, key, value)

    def store_data(self, data, name, key):
        with pd.HDFStore(os.path.join(self.data_folder, name)) as f:
            f.put(key, data)

    def load_data(self, name, key):
        with pd.HDFStore(os.path.join(self.data_folder, name)) as f:
            data = f[key]
        return data


class tDNMS_Experiment(Experiment):
    @classmethod
    def expt_types(cls):
        return ['tDNMS', 'tDNMS_dreadd', 'tDNMS_dreadd_day14',
                'tDNMS_dreadd_day1']

    @classmethod
    def _odor_intervals(cls, data):
        odor_diff = data['Odor'].astype(int).diff()
        odor_start_index = data['Odor'][odor_diff > 0].index
        odor_stop_index = data['Odor'][odor_diff < 0].index

        odor_intervals = pd.Series(
            list(zip(odor_start_index.get_level_values('Time'),
                     odor_stop_index.get_level_values('Time'))))
        odor_intervals.index = odor_start_index
        return odor_intervals

    @classmethod
    def _determine_cue_lengths(cls, data):
        odor_intervals = cls._odor_intervals(data)
        odor_lengths = odor_intervals.apply(np.diff).apply(lambda x: x[0])
        cutoff = odor_lengths[odor_lengths.sort_values().diff().idxmax()]

        key = odor_lengths.apply(lambda x, d: 'S' if x < d else 'L', d=cutoff)
        key_index = pd.Series(
            key.values, index=pd.IntervalIndex.from_tuples(odor_intervals.values)
            ).reindex(data.index.get_level_values('Time'))
        key_index.index = data.index
        trial_types = key_index.dropna().groupby('trial').unique().map(''.join).reindex(
            data.index.get_level_values('trial'))
        trial_types[trial_types == 'S'] = 'SS'
        return trial_types

    @classmethod
    def _determine_results(cls, data, lick_only=False):
        trial_types = data.index.get_level_values('type').to_series(
            index=data.index.get_level_values('trial'))
        trials_index = trial_types.groupby('trial').head(1)
        licks = data['Lick_Start']

        if lick_only:
            water = data['Lick_Start']
        else:
            water = data['Water_Start']

        results = pd.Series(None, trials_index.index)
        results[(trials_index == 'SS') & (licks.groupby('trial').sum() == 0)] = 'CR'
        results[(trials_index == 'SS') & (licks.groupby('trial').sum() > 0)] = 'FA'
        results[((trials_index == 'LS') | (trials_index == 'SL') | (trials_index == 'LL')) &
                (water.groupby('trial').sum() > 0)] = 'H'
        results[((trials_index == 'LS') | (trials_index == 'SL') | (trials_index == 'LL')) &
                (water.groupby('trial').sum() == 0)] = 'M'
        return results

    @staticmethod
    def _determine_consumption_window(data):
        water_delivery_times = data['Water_Start'][
            data['Water_Start']].groupby('trial').apply(
                lambda x: x.index.get_level_values('Time')[0])
        trial_end_times = data['Water_Start'].groupby('trial').apply(
            lambda x: x.index.get_level_values('Time')[-1])
        reward_intervals = pd.concat(
            (water_delivery_times, trial_end_times), axis=1).dropna(axis=0).apply(
                tuple, axis=1)
        reward_times = pd.Series(
            True, index=pd.IntervalIndex.from_tuples(reward_intervals)).reindex(
                data.index.get_level_values('Time')).fillna(False)
        reward_times.index = data.index
        return reward_times


    @functools.cached_property
    def _behavior_data(self):
        data = self.load_data('behavior.h5', 'behavior')
        if len(pd.Index(['result', 'type']).difference(data.index.names)):
            trial_types = self.__class__._determine_cue_lengths(data)
            data.index = pdH.add_level(data.index, trial_types, 'type')

            results = self.__class__._determine_results(data)
            data.index = pdH.add_level(
                data.index,
                results.reindex(data.index.get_level_values('trial')),
                'result')
            self.store_data(data, 'behavior.h5', 'behavior')

        valid_trials = (
            ((data['Odor'].astype(int).diff() > 0).groupby('trial').sum() == 2) &
             (data['Odor'].groupby('trial').apply(lambda x: x.index.unique('trial_time').max()) >= 20)
            ).reset_index().set_index('Odor').xs(True)['trial']
        if len(valid_trials) != len(data.index.unique('trial')):
            warnings.warn(
                "\nInvalid Trials Removed:\n" +
                f"{data.index.unique('trial').difference(valid_trials)}")
        return data.reindex(valid_trials, level='trial')

    @property
    def behavior_data(self):
        return self._behavior_data.copy()

    def downsample_behavior(self, frame_rate):
        #TODO: Fix velocity data
        try:
            data = self.load_data('behavior.h5', '/' + str(frame_rate))
            if 'Odor' not in data.columns or data['Odor'].dtype != type(True):
                raise KeyError
            if self.experiment_type == 'tDNMS_longISI_probe' and 'S' in data.index.unique('ISI_type'):
                raise KeyError
            return data

        except (KeyError, AttributeError):
            pass
        data = self.behavior_data
        data.index = pdH.update_level(
            data.index,
            pd.TimedeltaIndex(data.index.get_level_values('trial_time'),
                              unit='s'), 'trial_time')

        resampled_data = data.astype(int).groupby(
            data.index.names.difference(['Time', 'trial_time'])).resample(
                f'{int(frame_rate*1e3)}ms', level='trial_time').sum()
        level_order = np.insert(
            np.arange(len(resampled_data.index.names) - 1), 1, -1)
        resampled_data = resampled_data.reorder_levels(level_order)
        resampled_data.index = pdH.update_level(
            resampled_data.index,
            resampled_data.index.get_level_values('trial_time').total_seconds(),
            'trial_time')

        max_counts = np.round(
            frame_rate / data.index.get_level_values('Time'
            ).to_series().diff().median(), 0)
        resampled_data['Odor'] = resampled_data['Odor'] > max_counts / 2

        self.store_data(resampled_data, 'behavior.h5', '/' + str(frame_rate))
        return resampled_data

    @functools.cached_property
    def trial_start_times(self):
        index = self.behavior_data.groupby('trial').head(1).index
        return index.get_level_values('Time').to_series(index=index)

    @functools.cached_property
    def consumption_window(self):
        return self.__class__._determine_consumption_window(self.behavior_data)

    @functools.cached_property
    def odors_demixed(self):
        odor_intervals = self.__class__._odor_intervals(self.behavior_data)
        return odor_intervals.groupby('trial').apply(list).apply(
            lambda x: x[:2] if len(x) >= 2 else x + [None]).apply(
                pd.Series, index=['Odor1', 'Odor2'])

    @functools.cached_property
    def odor_times(self):
        odors_demixed = self.odors_demixed
        odor_times = odors_demixed.apply(pd.IntervalIndex.from_tuples).apply(
            lambda x: pd.Series(True, index=x)).reindex(
                self._behavior_data.index.get_level_values('Time')).fillna(False)
        odor_times.index = self.behavior_data.index
        return odor_times

    @functools.cached_property
    def response_window(self):
        response_interval = self.odors_demixed['Odor2'].apply(lambda x: (x[1], x[1] + 3))
        response_times = pd.Series(
            True, index=pd.IntervalIndex.from_tuples(response_interval)).reindex(
                self._behavior_data.index.get_level_values('Time')).fillna(False)
        response_times.index = self._behavior_data.index
        return response_times


    @classmethod
    def _sort_model_parts(cls, model):
        return ' + '.join(
            pd.Series(model.split(' + ')).apply(
                lambda x: sorted(x.split(' x '))).apply(' x '.join).sort_values())


    @classmethod
    def _model_hash(cls, model):
        sorted_model = cls._sort_model_parts(model)
        return hashlib.md5(sorted_model.encode('utf-8')).hexdigest()[:12]


    def save_model_predictions(self, model, predictions):
        hash_key = self.__class__._model_hash(model)
        predictions_file = os.path.join(
            self.data_folder, 'model.{}.h5'.format(hash_key))
        predictions.attrs['model'] = model
        models = self.get('models', [])
        models.append(model)
        self.models = models
        self.save(store=True)
        with pd.HDFStore(predictions_file, 'w') as f:
            f['predictions'] = predictions
            f.get_storer('predictions').attrs.attrs = predictions.attrs


    def model_predictions(self, model):
        hash_key = self.__class__._model_hash(model)
        predictions_file = os.path.join(
            self.data_folder, 'model.{}.h5'.format(hash_key))
        with pd.HDFStore(predictions_file, 'r') as f:
            predictions = f['predictions']
            predictions.attrs = f.get_storer('predictions').attrs.attrs
        return predictions


    def save_model(self, model, classifier, index):
        hash_key = self.__class__._model_hash(model)
        model_file = os.path.join(
            self.data_folder, 'model.{}.h5'.format(hash_key))
        weights = pd.Series(classifier.coef_, index=index)
        weights.attrs['model'] = model
        weights.attrs['intercept'] = classifier.intercept_
        weights.attrs['alpha'] = classifier.alpha
        weights.attrs['index'] = index

        with pd.HDFStore(model_file, 'a') as f:
            f['weights'] = weights
            f.get_storer('weights').attrs.attrs = weights.attrs


    def model(self, model):
        hash_key = self.__class__._model_hash(model)
        model_file = os.path.join(
            self.data_folder, 'model.{}.h5'.format(hash_key))

        with pd.HDFStore(model_file, 'r') as f:
            weights = f['weights']
            weights.attrs = f.get_storer('weights').attrs.attrs

        alpha = weights.attrs['alpha']
        clf = sklearn.linear_model.PoissonRegressor(alpha=alpha, max_iter=1000)
        clf.coef_ = weights
        clf.intercept_ = weights.attrs['intercept']
        clf._base_loss = sklearn._loss.loss.HalfPoissonLoss()
        return clf


class tDNMS_ShortISIProbe(tDNMS_Experiment):
    @classmethod
    def expt_types(cls):
        return ['tDNMS_isi_probe']

    def isi_index(self, behavior_data):
        isi_times = self._odor_intervals(behavior_data).groupby('trial').apply(
            lambda x: x.values[1][0] - x.values[0][1]).apply(
                np.round, decimals=1)
        times = isi_times.value_counts().sort_values().iloc[-2:]
        short_isi = times.idxmin()
        normal_isi = times.idxmax()
        return isi_times.apply(
            lambda x, d={short_isi: 'S', normal_isi: 'N'}: d.get(x, None))

    @functools.cached_property
    def behavior_data(self):
        bd = super(self.__class__, self).behavior_data
        isi = self.isi_index(bd)

        fr = bd.index.to_frame()
        fr['ISI_type'] = isi.reindex(bd.index.get_level_values('trial')).values
        bd.index = pd.MultiIndex.from_frame(fr)
        return pdH.drop_null(bd, axis=0, level='ISI_type')


class tDNMS_LongISIProbe(tDNMS_Experiment):
    @classmethod
    def expt_types(cls):
        return ['tDNMS_longISI_probe']

    def isi_index(self, behavior_data):
        isi_times = self._odor_intervals(behavior_data).groupby('trial').apply(
            lambda x: x.values[1][0] - x.values[0][1]).apply(
                np.round, decimals=1)
        times = isi_times.value_counts().sort_values().iloc[-2:]
        #normal_isi = times.idxmax()
        #long_isi = times.idxmin()
        normal_isi = times.index.min()
        long_isi = times.index.max()
        return isi_times.apply(
            lambda x, d={normal_isi: 'N', long_isi: 'L'}: d.get(x, None))

    @functools.cached_property
    def behavior_data(self):
        bd = super(self.__class__, self).behavior_data
        isi = self.isi_index(bd)

        fr = bd.index.to_frame()
        fr['ISI_type'] = isi.reindex(bd.index.get_level_values('trial')).values
        bd.index = pd.MultiIndex.from_frame(fr)
        return pdH.drop_null(bd, axis=0, level='ISI_type')
