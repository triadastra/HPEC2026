"""Time series datasets, especially for medical time series."""


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.dataloaders.base import default_data_path, SequenceDataset, deprecated

import pandas as pd
from typing import List, Dict, Tuple
import pickle
import os
import glob

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
def create_blocks(X, y, N):
    """Convert time-series into blocks of N timesteps.
    Args:
        X: Input array of shape (Time, Nodes, Features)
        y: Target array of shape (Time, Nodes)
        N: Number of timesteps per block
    Returns:
        X_blocks: Shape (Time-N, N, Nodes, Features)
        y_aligned: Shape (Time-N, Nodes)
    """
    Time = X.shape[0]
    X_blocks = np.stack([X[i:i+N] for i in range(Time - N)], axis=0)
    y_aligned = y[N:]
    return np.transpose(X_blocks, (0, 2, 1, 3)), y_aligned    
    
    

class BIDMC(SequenceDataset):
    """BIDMC datasets for Respiratory Rate / Heart Rate / Oxygen Saturation regression"""

    _name_ = "bidmc"
    d_input = 2

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def l_output(self):
        return 4000 if self.prediction else 0

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": True,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / self.target / split / "trainx.npy")
        y_train = np.load(self.data_dir / self.target / split / "trainy.npy")
        X_val = np.load(self.data_dir / self.target / split / "validx.npy")
        y_val = np.load(self.data_dir / self.target / split / "validy.npy")
        X_test = np.load(self.data_dir / self.target / split / "testx.npy")
        y_test = np.load(self.data_dir / self.target / split / "testy.npy")

        if self.prediction:
            y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"BIDMC{self.target}_{split}"

class EegDataset(SequenceDataset):

    _name_ = "eegseizure"

    init_defaults = {
        "l_output": 0,
        "d_input": 19,
        "d_output": 2,
        "machine": "gemini",
        "hospital": "stanford",
        "clip_len": 60,
        "stride": 60,
        "offset": 0,
        "ss_clip_len": 0,
        "use_age": False,
        "gnn": False,
        "fft": False,
        "rerun_meerkatdp": False,
        "streaming_eval": False,
        "sz_label_sensitivity": 1,
    }

    def setup(self):
        import meerkat as mk
        from meerkat.contrib.eeg import (build_stanford_eeg_dp,
                                         build_streaming_stanford_eeg_dp,
                                         build_tuh_eeg_dp)
        from torch.utils.data import WeightedRandomSampler

        assert self.sz_label_sensitivity <= self.clip_len

        # from src.dataloaders.eegseizure import balance_dp, split_dp, merge_in_split
        if self.machine == "gemini":
            data_dir = "/media/4tb_hdd"
            data_dir_tuh = "/media/nvme_data/siyitang/TUH_eeg_seq_v1.5.2"
            raw_tuh_data_dir = "/media/nvme_data/TUH/v1.5.2"
        elif self.machine == "zaman":
            data_dir = "/data/ssd1crypt/datasets"
            data_dir_tuh = "/data/ssd1crypt/datasets/TUH_v1.5.2"
            raw_tuh_data_dir = data_dir_tuh

        if self.hospital == "tuh":
            dp = build_tuh_eeg_dp(
                f"{data_dir_tuh}/resampled_signal",
                f"{raw_tuh_data_dir}/edf",
                clip_len=self.clip_len,
                offset=self.offset,
                ss_clip_len=self.ss_clip_len,
                gnn=self.gnn,
                skip_terra_cache=self.rerun_meerkatdp,
            ).load()

        else:
            dp = build_stanford_eeg_dp(
                f"{data_dir}/eeg_data/stanford/stanford_mini",
                f"{data_dir}/eeg_data/lpch/lpch",
                "/home/ksaab/Documents/meerkat/meerkat/contrib/eeg/file_markers",
                clip_len=self.clip_len,
                offset=self.offset,
                skip_terra_cache=self.rerun_meerkatdp,
            ).load()

        if self.streaming_eval:
            streaming_dp = build_streaming_stanford_eeg_dp(
                f"{data_dir}/SEC-0.1/stanford",
                f"{data_dir}/SEC-0.1/lpch",
                "/data/crypt/eegdbs/SEC-0.1/SEC-0.1-sz-annotations-match-lvis",
                clip_len=self.clip_len,
                stride=self.stride,
                sz_label_sensitivity=self.sz_label_sensitivity,
                train_frac=0.0,
                valid_frac=0.5,
                test_frac=0.5,
                skip_terra_cache=self.rerun_meerkatdp,
            ).load()

            # remove patients in dp that are in streaming_dp
            streaming_patients = streaming_dp["patient_id"].unique()
            keep_patient_mask = np.array(
                [patient not in streaming_patients for patient in dp["patient_id"]]
            )
            dp = dp.lz[keep_patient_mask]

        # shuffle datapanel
        np.random.seed(0)
        ndxs = np.arange(len(dp))
        np.random.shuffle(ndxs)
        dp = dp.lz[ndxs]

        val_split = "valid"
        test_split = "test"

        input_key = "input"
        target_key = "target"

        train_mask = dp["split"] == "train"
        val_mask = dp["split"] == val_split
        test_mask = dp["split"] == test_split

        if self.fft:
            input_key = "fft_input"
            self.d_input = 1900
        if self.ss_clip_len > 0:
            target_key = "ss_output"
            self.d_output = 19*100 #int(19 * (200* self.ss_clip_len / 2))
            self.l_output = self.ss_clip_len

            # train_mask = np.logical_and(train_mask.data,(dp["target"]==1).data)
            # val_mask = np.logical_and(val_mask.data,(dp["target"]==1).data)
            # test_mask = np.logical_and(test_mask.data,(dp["target"]==1).data)

        self.dataset_train = dp.lz[train_mask][
            input_key, target_key, "age", "target"
        ]
        self.dataset_val = dp.lz[val_mask][
            input_key, target_key, "age", "target"
        ]
        self.dataset_test = dp.lz[test_mask][
            input_key, target_key, "age"
        ]


        # define whats returned by datasets
        if self.gnn:
            lambda_fnc = lambda x: (
                x[input_key][0],
                torch.tensor(x[target_key]).to(torch.long),
                x[input_key][1],  # graph supports
            )
            if self.ss_clip_len > 0:
                lambda_fnc = lambda x: (
                x[input_key][0],
                torch.tensor(x[target_key][0]).to(torch.long),
                torch.tensor(x[target_key][0]).to(torch.long), # decoder takes y as well
                x[input_key][1],  # graph supports
            )
            if self.use_age:
                lambda_fnc = lambda x: (
                    x[input_key][0],
                    torch.tensor(x[target_key]).to(torch.long),
                    x[input_key][1],  # graph supports
                    torch.tensor(x["age"]).to(torch.float),
                )
        else:
            lambda_fnc = lambda x: (
                x[input_key][0],
                torch.tensor(x[target_key]).to(torch.long)
                if self.ss_clip_len == 0
                else x[target_key],
            )
            if self.use_age:
                lambda_fnc = lambda x: (
                    x[input_key][0],
                    torch.tensor(x[target_key]).to(torch.long)
                    if self.ss_clip_len == 0
                    else x[target_key],
                    torch.tensor(x["age"]).to(torch.float),
                )

        self.dataset_train["examples"] = mk.LambdaColumn(self.dataset_train, lambda_fnc)

        if self.ss_clip_len == 0:
            # define train sampler
            train_target = self.dataset_train["target"].data.astype(np.int)
            class_sample_count = np.array(
                [len(np.where(train_target == t)[0]) for t in np.unique(train_target)]
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in train_target])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
        else:
            samples_weight = torch.ones(len(self.dataset_train))
        self.train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        self.dataset_val["examples"] = mk.LambdaColumn(self.dataset_val, lambda_fnc)

        self.dataset_test["examples"] = mk.LambdaColumn(self.dataset_test, lambda_fnc)
        print(
            f"Train:{len(self.dataset_train)} Validation:{len(self.dataset_val)} Test:{len(self.dataset_test)}"
        )

        if self.streaming_eval:
            self.stream_dataset_val = streaming_dp.lz[streaming_dp["split"] == "valid"][
                input_key, "target", "age", "clip_start"
            ]
            self.stream_dataset_test = streaming_dp.lz[streaming_dp["split"] == "test"][
                input_key, "target", "age", "clip_start"
            ]

            self.stream_dataset_val["examples"] = mk.LambdaColumn(
                self.stream_dataset_val,
                lambda x: (
                    x[input_key],
                    torch.tensor(x["target"]).to(torch.long),
                    torch.tensor(x["age"]).to(torch.float),
                    torch.tensor(x["clip_start"]).to(torch.float),
                ),
            )

            self.stream_dataset_test["examples"] = mk.LambdaColumn(
                self.stream_dataset_test,
                lambda x: (
                    x[input_key],
                    torch.tensor(x["target"]).to(torch.long),
                    torch.tensor(x["age"]).to(torch.float),
                    torch.tensor(x["clip_start"]).to(torch.float),
                ),
            )

    def train_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        # No collate_fn is passed in: the default one does the right thing

        return torch.utils.data.DataLoader(
            self.dataset_train["examples"],
            sampler=self.train_sampler,
            **kwargs,
        )

    def val_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        # No collate_fn is passed in: the default one does the right thing
        return torch.utils.data.DataLoader(
            self.dataset_val["examples"],
            **kwargs,
        )

    def test_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        # No collate_fn is passed in: the default one does the right thing
        return torch.utils.data.DataLoader(
            self.dataset_test["examples"],
            **kwargs,
        )

    def stream_val_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        if self.streaming_eval:
            # No collate_fn is passed in: the default one does the right thing
            return torch.utils.data.DataLoader(
                self.stream_dataset_val["examples"],
                **kwargs,
            )

    def stream_test_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        if self.streaming_eval:
            # No collate_fn is passed in: the default one does the right thing
            return torch.utils.data.DataLoader(
                self.stream_dataset_test["examples"],
                **kwargs,
            )

class PTBXL(SequenceDataset):

    _name_ = "ptbxl"

    init_defaults = {
        "sampling_rate": 100,
        "duration": 10,
        "nleads": 12,
        "ctype": "superdiagnostic",
        "min_samples": 0,
    }

    @property
    def d_input(self):
        return self.nleads

    def load_raw_data(self, df):
        import wfdb

        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(str(self.data_dir / f)) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(str(self.data_dir / f)) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.L = self.sampling_rate * self.duration
        self.l_output = 0  # TODO(Priya): This changes with every multilabel setting?

        # PTBXL imports
        import ast

        import pandas as pd
        from sklearn import preprocessing

        # load and convert annotation data
        Y = pd.read_csv(self.data_dir / "ptbxl_database.csv", index_col="ecg_id")
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(self.data_dir / "scp_statements.csv", index_col=0)

        if self.ctype in [
            "diagnostic",
            "subdiagnostic",
            "superdiagnostic",
            "superdiagnostic_multiclass",
        ]:
            agg_df = agg_df[agg_df.diagnostic == 1]

            def aggregate_superdiagnostic_multiclass(y_dic):
                lhmax = -1  # Superclass has the highest likelihood
                superclass = ""
                for key in y_dic.keys():
                    if key in agg_df.index and y_dic[key] > lhmax:
                        lhmax = y_dic[key]
                        superclass = agg_df.loc[key].diagnostic_class
                return superclass

            def aggregate_all_diagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        tmp.append(key)
                return list(set(tmp))

            def aggregate_subdiagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        c = agg_df.loc[key].diagnostic_subclass
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            def aggregate_superdiagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        c = agg_df.loc[key].diagnostic_class
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            # Apply aggregation
            if self.ctype == "superdiagnostic_multiclass":
                Y["target"] = Y.scp_codes.apply(aggregate_superdiagnostic_multiclass)
            elif self.ctype == "subdiagnostic":
                Y["target"] = Y.scp_codes.apply(aggregate_subdiagnostic)
            elif self.ctype == "superdiagnostic":
                Y["target"] = Y.scp_codes.apply(aggregate_superdiagnostic)
            elif self.ctype == "diagnostic":
                Y["target"] = Y.scp_codes.apply(aggregate_all_diagnostic)

        elif self.ctype in ["form", "rhythm"]:

            if self.ctype == "form":
                agg_df = agg_df[agg_df.form == 1]
            else:
                agg_df = agg_df[agg_df.rhythm == 1]

            def aggregate_form_rhythm(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        c = key
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            Y["target"] = Y.scp_codes.apply(aggregate_form_rhythm)

        elif self.ctype == "all":
            Y["target"] = Y.scp_codes.apply(lambda x: list(set(x.keys())))

        counts = pd.Series(np.concatenate(Y.target.values)).value_counts()
        counts = counts[counts > self.min_samples]
        Y.target = Y.target.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        Y["target_len"] = Y.target.apply(lambda x: len(x))
        Y = Y[Y.target_len > 0]
        # Load raw signal data
        X = self.load_raw_data(Y)

        # Split data into train, val and test
        val_fold = 9
        test_fold = 10

        # Convert labels to multiclass or multilabel targets
        if self.ctype == "superdiagnostic_multiclass":
            le = preprocessing.LabelEncoder()
        else:
            le = preprocessing.MultiLabelBinarizer()

        le.fit(Y.target)
        y = le.transform(Y.target)
        self.d_output = len(le.classes_)

        # Train
        X_train = X[np.where((Y.strat_fold != val_fold) & (Y.strat_fold != test_fold))]
        y_train = y[np.where((Y.strat_fold != val_fold) & (Y.strat_fold != test_fold))]
        # Val
        X_val = X[np.where(Y.strat_fold == val_fold)]
        y_val = y[np.where(Y.strat_fold == val_fold)]

        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = y[np.where(Y.strat_fold == test_fold)]

        def preprocess_signals(X_train, X_validation, X_test):
            # Standardize data such that mean 0 and variance 1
            ss = preprocessing.StandardScaler()
            ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

            return (
                apply_standardizer(X_train, ss),
                apply_standardizer(X_validation, ss),
                apply_standardizer(X_test, ss),
            )

        def apply_standardizer(X, ss):
            X_tmp = []
            for x in X:
                x_shape = x.shape
                X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
            X_tmp = np.array(X_tmp)
            return X_tmp

        X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.tensor(X_train).to(torch.float), torch.tensor(y_train)
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.tensor(X_val).to(torch.float), torch.tensor(y_val)
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.tensor(X_test).to(torch.float), torch.tensor(y_test)
        )

        print(
            f"Train:{len(X_train)} Validation:{len(X_val)} Test:{len(X_test)} Num_classes:{self.d_output}"
        )

        self.collate_fn = None

class IMU(SequenceDataset):
    """IMU (Inertial Measurement Units) dataset from an experimental study on Parkinson patients"""

    _name_ = "imu"
    d_input = 36  # len(imu_config)
    l_output = 0

    @property
    def d_output(self):
        return d_input if self.prediction else 2

    @property
    def init_defaults(self):
        return {
            #'target': 'RR', # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": True,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.collate_fn = None

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)

        # dictionary of config name to list of features
        # choose sensors06_chest_lumbar_ankles_feet by default
        # ignore this now as we're only using a fixed set of features
        with open(self.data_dir / "sensor_configs.pkl", "rb") as config_f:
            imu_config_map = pickle.load(config_f)
        imu_config = imu_config_map["sensors06_chest_lumbar_ankles_feet"]

        with open(self.data_dir / "0_train_matrices.pkl", "rb") as f_handle:
            tr = pickle.load(f_handle)
        with open(self.data_dir / "0_val_matrices.pkl", "rb") as f_handle:
            val = pickle.load(f_handle)
        with open(self.data_dir / "0_test_matrices.pkl", "rb") as f_handle:
            te = pickle.load(f_handle)

        X_train = tr[0]
        y_train = tr[1].astype(int)
        X_val = val[0]
        y_val = val[1].astype(int)
        X_test = te[0]
        y_test = te[1].astype(int)

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.tensor(y_train, dtype=torch.long)
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.tensor(y_val, dtype=torch.long)
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.tensor(y_test, dtype=torch.long)
        )

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"IMU_{split}"

class STGNN(SequenceDataset):
    """STGNN"""

    _name_ = "stgnn"
    d_input = 2

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def l_output(self):
        return 4000 if self.prediction else 0

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": False,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        N = 56
        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / "trainx.npy")
        y_train = np.load(self.data_dir / "trainy.npy")
        X_val = np.load(self.data_dir / "validx.npy")
        y_val = np.load(self.data_dir / "validy.npy")
        X_test = np.load(self.data_dir / "testx.npy")
        y_test = np.load(self.data_dir / "testy.npy")

        if self.prediction:
            print("YES I'M PREDICTION!")
            y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
        else:
            print("NO I'M NOT!")
        # Transform into blocks
        X_train_blocks, y_train_aligned = create_blocks(X_train, y_train, N)
        X_val_blocks, y_val_aligned = create_blocks(X_val, y_val, N)
        X_test_blocks, y_test_aligned = create_blocks(X_test, y_test, N)
        print(f"input tran shape is {X_train_blocks.shape} and y {y_train_aligned.shape}")
        print(f"input test shape is {X_test_blocks.shape} and y {y_test_aligned.shape}")

        # Convert to torch TensorDatasets
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_blocks), torch.FloatTensor(y_train_aligned)
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_blocks), torch.FloatTensor(y_val_aligned)
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_blocks), torch.FloatTensor(y_test_aligned)
        )            
        '''
        print(f"input tran shape is {X_train.shape} and y {y_train.shape}")
        print(f"input test shape is {X_test.shape} and y {y_test.shape}")
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train).contiguous(), torch.FloatTensor(y_train).contiguous()
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val).contiguous(), torch.FloatTensor(y_val).contiguous()
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test).contiguous(), torch.FloatTensor(y_test).contiguous()
        )
        '''
        
    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"STGNN"

class Barge3D(SequenceDataset):
    """Barge3D - Original numpy file loading"""
    i = 0
    _name_ = "Barge3D"
    d_input = 27

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def l_output(self):
        return 4000 if self.prediction else 0

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": False,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / "X1_train.npy")
        y_train = np.load(self.data_dir / "y1_train.npy")
        X_val = np.load(self.data_dir / "X1_val.npy")
        y_val = np.load(self.data_dir / "y1_val.npy")
        X_test = np.load(self.data_dir / "X1_test.npy")
        y_test = np.load(self.data_dir / "y1_test.npy")

        if self.prediction:
            print("YES I'M PREDICTION!")
            y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
        else:
            print("NO I'M NOT!")
        # Transform into blocks
        #X_train_blocks, y_train_aligned = create_blocks(X_train, y_train, N)
        #X_val_blocks, y_val_aligned = create_blocks(X_val, y_val, N)
        #X_test_blocks, y_test_aligned = create_blocks(X_test, y_test, N)
        print(f"input tran shape is {X_train.shape} and y {y_train.shape}")
        print(f"input test shape is {X_test.shape} and y {y_test.shape}")

        # Convert to torch TensorDatasets
        # Check if y tensors have extra dimensions and handle appropriately
        if len(y_train.shape) > len(X_train.shape):
            print("Warning: y_train has extra dimensions, using first slice")
            y_train_processed = y_train[..., 0] if y_train.shape[-1] == 1 else y_train[..., 0]
            y_val_processed = y_val[..., 0] if y_val.shape[-1] == 1 else y_val[..., 0]
            y_test_processed = y_test[..., 0] if y_test.shape[-1] == 1 else y_test[..., 0]
        else:
            y_train_processed = y_train
            y_val_processed = y_val
            y_test_processed = y_test

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train_processed)
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val_processed)
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test_processed)
        )

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"Barge3D"


class Barge3DComprehensive(SequenceDataset):
    """Barge3D - Comprehensive data loading following shared_dataloader.py pattern"""
    _name_ = "Barge3DComprehensive"
    d_input = 28  # 28 features: 2 current + 24 lags + 2 seasonal
    _collate_arg_names = ['group_mask', 'x_mask_t']

    @property
    def d_output(self):
        return 2  # value and weight

    @property
    def l_output(self):
        return 0

    @property
    def init_defaults(self):
        return {
            "target": "both",  # "value" | "weight" | "both"
            "input_len": 36,
            "horizon": 1,
        }

    def setup(self):
        # Import shared dataloader functions
        import sys
        sys.path.append('/root')
        from shared_dataloader import (
            read_folder, build_grid, temporal_split_calendar,
            compute_series_stats, apply_stats_with_backoff,
            add_lags, feature_view, build_combo_maps,
            ComboWindowDataset, SharedConfig
        )

        # Create config
        cfg = SharedConfig()
        # Allow explicit overrides via dataset.imports_dir / dataset.exports_dir
        explicit_imports = getattr(self, 'imports_dir', None)
        explicit_exports = getattr(self, 'exports_dir', None)
        if explicit_imports is not None:
            cfg.imports_dir = str(explicit_imports)
        else:
            cfg.imports_dir = str(self.data_dir / "import" if self.data_dir else "data/imports")
        if explicit_exports is not None:
            cfg.exports_dir = str(explicit_exports)
        else:
            cfg.exports_dir = str(self.data_dir / "export" if self.data_dir else "data/exports")
        cfg.input_len = getattr(self, 'input_len', 36)
        cfg.horizon = getattr(self, 'horizon', 1)
        cfg.target = getattr(self, 'target', 'both')

        print("Loading comprehensive Barge3D data...")

        # 1) Read & build combo maps (from full panel)
        imports = read_folder(cfg.imports_dir, "Import")
        exports = read_folder(cfg.exports_dir, "Export")
        panel = pd.concat([imports, exports], ignore_index=True)
        combo2id, id2combo, num_combos = build_combo_maps(panel)

        # 2) Grid + encodings
        grid = build_grid(panel, cfg.start, cfg.end)

        # 3) Temporal split: calendar Train/Val/Test
        tr_raw, va_raw, te_raw = temporal_split_calendar(grid, cfg)

        # 4) Train-only stats → normalize both splits with hierarchical backoff
        stats = compute_series_stats(tr_raw)
        tr = apply_stats_with_backoff(tr_raw, stats)
        va = apply_stats_with_backoff(va_raw, stats)
        te = apply_stats_with_backoff(te_raw, stats)

        # 5) Add 12 lags (value & weight) + drop 2008
        tr = add_lags(tr, 12)
        va = add_lags(va, 12)
        te = add_lags(te, 12)

        # 6) Feature assembly
        trf, feat_cols, core_feats = feature_view(tr)
        vaf, _, _ = feature_view(va)
        tef, _, _ = feature_view(te)

        print(f"Features per group: {len(feat_cols)}")
        print(f"Number of combinations: {num_combos}")

        # 7) Datasets
        tr_va_concat = pd.concat([trf, vaf], ignore_index=True)
        tr_va_te_concat = pd.concat([trf, vaf, tef], ignore_index=True)
        ds_tr = ComboWindowDataset(trf, feat_cols, cfg.input_len, cfg.horizon, cfg.target,
                                   combo2id, num_combos,
                                   target_time_start=None, target_time_end=pd.Timestamp(cfg.split_train_end))
        ds_va = ComboWindowDataset(tr_va_concat, feat_cols, cfg.input_len, cfg.horizon, cfg.target,
                                   combo2id, num_combos,
                                   target_time_start=pd.Timestamp(cfg.split_val_start),
                                   target_time_end=pd.Timestamp(cfg.split_val_end))
        # Use full history (train+val+test) for test windows, but restrict target time to test calendar
        ds_te = ComboWindowDataset(tr_va_te_concat, feat_cols, cfg.input_len, cfg.horizon, cfg.target,
                                   combo2id, num_combos,
                                   target_time_start=pd.Timestamp(cfg.split_test_start),
                                   target_time_end=pd.Timestamp(cfg.split_test_end))

        self.dataset_train = ds_tr
        self.dataset_val = ds_va
        self.dataset_test = ds_te

        print(f"Train samples: {len(ds_tr)}, Val samples: {len(ds_va)}, Test samples: {len(ds_te)}")

    def __str__(self):
        return f"Barge3DComprehensive"



class Barge4D(SequenceDataset):
    """Barge4D"""
    i = 0
    _name_ = "Barge4D"
    d_input = 27

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def l_output(self):
        return 4000 if self.prediction else 0

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": False,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        
        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / "X2_train.npy")
        y_train = np.load(self.data_dir / "y2_train.npy")
        X_val = np.load(self.data_dir / "X2_val.npy")
        y_val = np.load(self.data_dir / "y2_val.npy")
        X_test = np.load(self.data_dir / "X2_test.npy")
        y_test = np.load(self.data_dir / "y2_test.npy")

        if self.prediction:
            print("YES I'M PREDICTION!")
            y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
        else:
            print("NO I'M NOT!")
        # Transform into blocks
        #X_train_blocks, y_train_aligned = create_blocks(X_train, y_train, N)
        #X_val_blocks, y_val_aligned = create_blocks(X_val, y_val, N)
        #X_test_blocks, y_test_aligned = create_blocks(X_test, y_test, N)
        print(f"input tran shape is {X_train.shape} and y {y_train.shape}")
        print(f"input test shape is {X_test.shape} and y {y_test.shape}")

        # Convert to torch TensorDatasets
        # Check if y tensors have extra dimensions and handle appropriately
        if len(y_train.shape) > len(X_train.shape):
            print("Warning: y_train has extra dimensions, using first slice")
            y_train_processed = y_train[..., 0] if y_train.shape[-1] == 1 else y_train[..., 0]
            y_val_processed = y_val[..., 0] if y_val.shape[-1] == 1 else y_val[..., 0]
            y_test_processed = y_test[..., 0] if y_test.shape[-1] == 1 else y_test[..., 0]
        else:
            y_train_processed = y_train
            y_val_processed = y_val
            y_test_processed = y_test

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train_processed)
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val_processed)
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test_processed)
        )            
        '''
        print(f"input tran shape is {X_train.shape} and y {y_train.shape}")
        print(f"input test shape is {X_test.shape} and y {y_test.shape}")
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train).contiguous(), torch.FloatTensor(y_train).contiguous()
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val).contiguous(), torch.FloatTensor(y_val).contiguous()
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test).contiguous(), torch.FloatTensor(y_test).contiguous()
        )
        '''
        
    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"Barge4D"


class Barge4DComprehensive(SequenceDataset):
    """Barge4D - Comprehensive data loading following shared_dataloader.py pattern"""
    _name_ = "Barge4DComprehensive"
    d_input = 28  # 28 features: 2 current + 24 lags + 2 seasonal
    _collate_arg_names = ['group_mask', 'x_mask_t']

    @property
    def d_output(self):
        return 2  # value and weight

    @property
    def l_output(self):
        return 0

    @property
    def init_defaults(self):
        return {
            "target": "both",  # "value" | "weight" | "both"
            "input_len": 36,
            "horizon": 1,
        }

    def setup(self):
        # Import shared dataloader functions
        import sys
        sys.path.append('/root')
        from shared_dataloader import (
            read_folder, build_grid, temporal_split_calendar,
            compute_series_stats, apply_stats_with_backoff,
            add_lags, feature_view, build_combo_maps,
            ComboWindowDataset, SharedConfig
        )

        # Create config
        cfg = SharedConfig()
        # Allow explicit overrides via dataset.imports_dir / dataset.exports_dir
        explicit_imports = getattr(self, 'imports_dir', None)
        explicit_exports = getattr(self, 'exports_dir', None)
        if explicit_imports is not None:
            cfg.imports_dir = str(explicit_imports)
        else:
            cfg.imports_dir = str(self.data_dir / "import" if self.data_dir else "data/imports")
        if explicit_exports is not None:
            cfg.exports_dir = str(explicit_exports)
        else:
            cfg.exports_dir = str(self.data_dir / "export" if self.data_dir else "data/exports")
        cfg.input_len = getattr(self, 'input_len', 36)
        cfg.horizon = getattr(self, 'horizon', 1)
        cfg.target = getattr(self, 'target', 'both')

        print("Loading comprehensive Barge4D data...")

        # 1) Read & build combo maps (from full panel)
        imports = read_folder(cfg.imports_dir, "Import")
        exports = read_folder(cfg.exports_dir, "Export")
        panel = pd.concat([imports, exports], ignore_index=True)
        combo2id, id2combo, num_combos = build_combo_maps(panel)

        # 2) Grid + encodings
        grid = build_grid(panel, cfg.start, cfg.end)

        # 3) Temporal split: calendar Train/Val/Test
        tr_raw, va_raw, te_raw = temporal_split_calendar(grid, cfg)

        # 4) Train-only stats → normalize both splits with hierarchical backoff
        stats = compute_series_stats(tr_raw)
        tr = apply_stats_with_backoff(tr_raw, stats)
        va = apply_stats_with_backoff(va_raw, stats)
        te = apply_stats_with_backoff(te_raw, stats)

        # 5) Add 12 lags (value & weight) + drop 2008
        tr = add_lags(tr, 12)
        va = add_lags(va, 12)
        te = add_lags(te, 12)

        # 6) Feature assembly
        trf, feat_cols, core_feats = feature_view(tr)
        vaf, _, _ = feature_view(va)
        tef, _, _ = feature_view(te)

        print(f"Features per group: {len(feat_cols)}")
        print(f"Number of combinations: {num_combos}")

        # 7) Datasets
        tr_va_concat = pd.concat([trf, vaf], ignore_index=True)
        ds_tr = ComboWindowDataset(trf, feat_cols, cfg.input_len, cfg.horizon, cfg.target,
                                   combo2id, num_combos,
                                   target_time_start=None, target_time_end=pd.Timestamp(cfg.split_train_end))
        ds_va = ComboWindowDataset(tr_va_concat, feat_cols, cfg.input_len, cfg.horizon, cfg.target,
                                   combo2id, num_combos,
                                   target_time_start=pd.Timestamp(cfg.split_val_start),
                                   target_time_end=pd.Timestamp(cfg.split_val_end))
        ds_te = ComboWindowDataset(tef, feat_cols, cfg.input_len, cfg.horizon, cfg.target,
                                   combo2id, num_combos,
                                   target_time_start=pd.Timestamp(cfg.split_test_start),
                                   target_time_end=pd.Timestamp(cfg.split_test_end))

        self.dataset_train = ds_tr
        self.dataset_val = ds_va
        self.dataset_test = ds_te

        print(f"Train samples: {len(ds_tr)}, Val samples: {len(ds_va)}, Test samples: {len(ds_te)}")

    def __str__(self):
        return f"Barge4DComprehensive"



class Agg4DState(SequenceDataset):
    """
    Aggregate panel into 4D windows per State:
    - X: (batch, S, L, F) where S=#states, L=input_len, F=28 core features summed over (Commodity, Flow)
    - y: (batch, S, 2) next-step [Value, Weight] per state, summed over (Commodity, Flow)

    Train/Val/Test calendar windows mirror the flat baselines:
    - Train targets ≤ train_end (context uses train only)
    - Val targets within [val_start, val_end] (context uses train∪val)
    - Test targets within [test_start, test_end] (context uses train∪val∪test)
    """

    _name_ = "s4nd_agg4d"
    d_input = 28  # [Value, Weight] + 12 value lags + 12 weight lags + sin/cos
    _collate_arg_names = ['commodity_ids', 'flow_ids']

    @property
    def d_output(self):
        return 2

    @property
    def l_output(self):
        return 0

    @property
    def init_defaults(self):
        return {
            "imports_dir": "/root/data/imports",
            "exports_dir": "/root/data/exports",
            "start": "2008-01-01",
            "end":   "2025-05-01",
            "input_len": 36,
            "horizon": 1,
            "target": "both",
            # Calendar splits
            "split_train_end": "2023-12-01",
            "split_val_start": "2024-01-01",
            "split_val_end":   "2024-12-01",
            "split_test_start": "2025-01-01",
            "split_test_end":   "2025-05-01",
        }

    def _build_state_group_tensor(self, df_feat, feat_cols):
        """
        Build dense tensor with explicit group G=(Commodity×Flow) under each State:
          X_all: (T, S, G, F)
          group_list: sorted list of (Commodity, Import/Export)
        Missing entries are zeros.
        """
        states = sorted(df_feat['State'].dropna().unique().tolist())
        times = sorted([t for t in df_feat['Time'].unique().tolist() if pd.notna(t)])
        groups_df = df_feat[['Commodity','Import/Export']].drop_duplicates()
        groups = sorted([(r['Commodity'], r['Import/Export']) for _, r in groups_df.iterrows()])
        S = len(states); T = len(times); G = len(groups); F = len(feat_cols)
        X_all = np.zeros((T, S, G, F), dtype=np.float32)

        time_to_idx = {t:i for i,t in enumerate(times)}
        state_to_idx = {s:i for i,s in enumerate(states)}
        group_to_idx = {g:i for i,g in enumerate(groups)}

        for (s, c, f, t), gdf in df_feat.groupby(['State','Commodity','Import/Export','Time']):
            if pd.isna(t) or s not in state_to_idx or (c, f) not in group_to_idx:
                continue
            ti = time_to_idx.get(t, None)
            if ti is None: continue
            si = state_to_idx[s]; gi = group_to_idx[(c, f)]
            feats = gdf.sort_values('Time')[feat_cols].iloc[-1].to_numpy(dtype=np.float32)
            X_all[ti, si, gi, :] = feats
        return X_all, states, times, groups

    def _build_comm_flow_tensors(self, df_feat, states, times):
        """
        Build (T, S, C) commodity count and (T, S, 2) flow count tensors.
        Counts reflect number of records per (State, Time, Commodity|Flow).
        """
        # Commodity
        comms = sorted(df_feat['Commodity'].dropna().unique().tolist())
        C = len(comms)
        time_to_idx = {t: i for i, t in enumerate(times)}
        state_to_idx = {s: i for i, s in enumerate(states)}

        comm_counts = np.zeros((len(times), len(states), C), dtype=np.float32)
        for (s, t, c), g in df_feat.groupby(['State','Time','Commodity']):
            if pd.isna(t) or s not in state_to_idx or t not in time_to_idx or pd.isna(c):
                continue
            ti = time_to_idx[t]; si = state_to_idx[s]; ci = comms.index(c)
            comm_counts[ti, si, ci] += 1.0

        # Flow
        flows = ['Export','Import']
        flow_to_idx = {f:i for i,f in enumerate(flows)}
        flow_counts = np.zeros((len(times), len(states), 2), dtype=np.float32)
        for (s, t, f), g in df_feat.groupby(['State','Time','Import/Export']):
            if pd.isna(t) or s not in state_to_idx or t not in time_to_idx or f not in flow_to_idx:
                continue
            ti = time_to_idx[t]; si = state_to_idx[s]; fi = flow_to_idx[f]
            flow_counts[ti, si, fi] += 1.0

        return comm_counts, flow_counts, comms

    def _build_targets(self, df_feat):
        # Sum canonical targets across (Commodity, Flow) at each (State, Time)
        VC = 'Value'
        WC = 'Weight'
        keys = ['State', 'Time']
        agg = df_feat.groupby(keys)[[VC, WC]].mean().reset_index()

        states = sorted(df_feat['State'].dropna().unique().tolist())
        times = sorted([t for t in df_feat['Time'].unique().tolist() if pd.notna(t)])
        grid = (
            pd.MultiIndex.from_product([states, times], names=['State','Time']).to_frame(index=False)
            .merge(agg, on=['State','Time'], how='left')
            .sort_values(['State','Time'])
        )
        for c in [VC, WC]:
            grid[c] = grid[c].fillna(0.0)

        S = len(states)
        T = len(times)
        y_all = grid[[VC, WC]].to_numpy(dtype=np.float32).reshape(S, T, 2)
        y_all = np.transpose(y_all, (1, 0, 2)).astype(np.float32)  # (T, S, 2)
        return y_all, states, times

    def _windows(self, X_all, times, L, target_start=None, target_end=None):
        """Create samples: X: (N, S, G, L, F), y: (N, S, G, 2) selecting targets within [start,end]. Also returns target indices."""
        T = X_all.shape[0]  # X_all: (T, S, G, F)
        idxs = []
        for t in range(L, T):
            ts = times[t]
            if target_start is not None and ts < target_start: continue
            if target_end   is not None and ts > target_end:   continue
            idxs.append(t)
        if not idxs:
            return (
                np.zeros((0,) + (X_all.shape[1], X_all.shape[2], L, X_all.shape[3]), np.float32),
                np.zeros((0, X_all.shape[1], X_all.shape[2], 2), np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        X = np.stack([X_all[t-L:t, ...] for t in idxs], axis=0)          # (N, L, S, G, F)
        X = np.transpose(X, (0, 2, 3, 1, 4)).astype(np.float32)          # (N, S, G, L, F)
        y = np.stack([X_all[t, :, :, :2] for t in idxs], axis=0).astype(np.float32)  # (N, S, G, 2)
        return X, y, np.asarray(idxs, dtype=np.int64)

    def setup(self):
        # Import shared helpers
        import sys
        sys.path.append('/root')
        from shared_dataloader import (
            read_folder, build_grid, temporal_split_calendar,
            compute_series_stats, apply_stats_with_backoff,
            add_lags, feature_view
        )

        # IO
        imports = read_folder(self.imports_dir, "Import")
        exports = read_folder(self.exports_dir, "Export")
        panel  = pd.concat([imports, exports], ignore_index=True)
        grid   = build_grid(panel, self.start, self.end)

        # Splits
        tr_raw, va_raw, te_raw = temporal_split_calendar(grid, self)

        # Train-only stats
        stats = compute_series_stats(tr_raw)
        tr = apply_stats_with_backoff(tr_raw, stats)
        va = apply_stats_with_backoff(va_raw, stats)
        te = apply_stats_with_backoff(te_raw, stats)

        # Lags and features
        tr = add_lags(tr, 12)
        va = add_lags(va, 12)
        te = add_lags(te, 12)
        trf, feat_cols, _ = feature_view(tr)
        vaf, _, _ = feature_view(va)
        tef, _, _ = feature_view(te)

        # Build per-state dense tensors
        X_tr_all, states_tr, times_tr, groups = self._build_state_group_tensor(trf, feat_cols)
        y_tr_all, _,        _                = self._build_targets(trf)

        # Val context uses train∪val; targets restricted to val
        trva = pd.concat([trf, vaf], ignore_index=True)
        X_va_all, states_va, times_va, _ = self._build_state_group_tensor(trva, feat_cols)
        y_va_all, _,        _            = self._build_targets(trva)

        # Test context uses train∪val∪test; targets restricted to test
        trvate = pd.concat([trf, vaf, tef], ignore_index=True)
        X_te_all, states_te, times_te, _ = self._build_state_group_tensor(trvate, feat_cols)
        y_te_all, _,        _            = self._build_targets(trvate)

        # Sanity: align number of states across splits
        assert states_tr == states_va == states_te, "State sets differ across splits"

        L = int(self.input_len)
        t_val_start = pd.Timestamp(self.split_val_start)
        t_val_end   = pd.Timestamp(self.split_val_end)
        t_test_start= pd.Timestamp(self.split_test_start)
        t_test_end  = pd.Timestamp(self.split_test_end)

        X_tr, y_tr, idx_tr = self._windows(X_tr_all, times_tr, L, target_start=None, target_end=pd.Timestamp(self.split_train_end))
        X_va, y_va, idx_va = self._windows(X_va_all, times_va, L, target_start=t_val_start, target_end=t_val_end)
        X_te, y_te, idx_te = self._windows(X_te_all, times_te, L, target_start=t_test_start, target_end=t_test_end)

        # Save meta: number of commodities for embeddings
        # (computed below from groups)

        class _Agg4DStateDS(Dataset):
            def __init__(self, X, Y, commodity_ids, flow_ids):
                self.X = X; self.Y = Y; self.cid = commodity_ids; self.fid = flow_ids
            def __len__(self): return self.X.shape[0]
            def __getitem__(self, i):
                return (
                    torch.from_numpy(self.X[i]),
                    torch.from_numpy(self.Y[i]),
                    torch.from_numpy(self.cid[i]),
                    torch.from_numpy(self.fid[i]),
                )

        # Build per-(S,G) ids tensors
        comm_map = {g[0]: i for i, g in enumerate(groups)}  # same commodity index per pair
        # Better: extract commodity and flow ids grids of shape (S,G)
        commodities = sorted({c for c, _ in groups})
        commodity_to_id = {c:i for i,c in enumerate(commodities)}
        flow_to_id = {'Export':0,'Import':1}
        S, G = X_tr.shape[1], X_tr.shape[2]
        # Construct (S,G) arrays with (commodity_id, flow_id) using groups order (shared across states)
        cid_grid = np.zeros((S, G), dtype=np.int64)
        fid_grid = np.zeros((S, G), dtype=np.int64)
        for gi, (c, f) in enumerate(groups):
            cid = commodity_to_id.get(c, 0)
            fid = flow_to_id.get(f, 0)
            cid_grid[:, gi] = cid
            fid_grid[:, gi] = fid
        # Broadcast to batches N
        def tile_ids(n):
            return np.tile(cid_grid[None, ...], (n, 1, 1)), np.tile(fid_grid[None, ...], (n, 1, 1))

        cid_tr, fid_tr = tile_ids(X_tr.shape[0])
        cid_va, fid_va = tile_ids(X_va.shape[0])
        cid_te, fid_te = tile_ids(X_te.shape[0])

        self.n_comms = len(commodities)

        self.dataset_train = _Agg4DStateDS(X_tr, y_tr, cid_tr, fid_tr)
        self.dataset_val   = _Agg4DStateDS(X_va, y_va, cid_va, fid_va)
        self.dataset_test  = _Agg4DStateDS(X_te, y_te, cid_te, fid_te)

        # Update input feature dim to include CF embeddings (32 + 2)
        self.d_input = X_tr.shape[-1] + 32 + 2

    def __str__(self):
        return "Agg4DState"


class Agg6DAll(SequenceDataset):
    """
    Aggregate panel into 6D-style windows with explicit axes for State, Commodity, and Flow:
    - X: (batch, S, C, F2, L, Feat) where S=#states, C=#commodities, F2=2 flows, L=input_len, Feat=28 core features
    - y: (batch, S, C, F2, 2) next-step [Value, Weight] per (State, Commodity, Flow)

    Train/Val/Test calendar windows mirror the flat baselines and Agg4DState:
    - Train targets ≤ train_end (context uses train only)
    - Val targets within [val_start, val_end] (context uses train∪val)
    - Test targets within [test_start, test_end] (context uses train∪val∪test)
    """

    _name_ = "s4nd_agg6d"
    d_input = 28  # [Value, Weight] + 12 value lags + 12 weight lags + sin/cos
    _collate_arg_names = []

    @property
    def d_output(self):
        return 2

    @property
    def l_output(self):
        return 0

    @property
    def init_defaults(self):
        return {
            "imports_dir": "/root/data/imports",
            "exports_dir": "/root/data/exports",
            "start": "2008-01-01",
            "end":   "2025-05-01",
            "input_len": 36,
            "horizon": 1,
            "target": "both",
            # Calendar splits
            "split_train_end": "2023-12-01",
            "split_val_start": "2024-01-01",
            "split_val_end":   "2024-12-01",
            "split_test_start": "2025-01-01",
            "split_test_end":   "2025-05-01",
        }

    def _build_scflow_tensor(self, df_feat, feat_cols):
        """
        Build dense tensor with explicit axes State × Commodity × Flow under each timestamp:
          X_all: (T, S, C, 2, Feat)
          states, commodities, times are sorted lists; Flow is fixed order ['Export','Import']
        Missing entries are zeros.
        """
        states = sorted(df_feat['State'].dropna().unique().tolist())
        commodities = sorted(df_feat['Commodity'].dropna().unique().tolist())
        flows = ['Export', 'Import']
        times = sorted([t for t in df_feat['Time'].unique().tolist() if pd.notna(t)])

        S = len(states); C = len(commodities); F2 = 2; T = len(times); F = len(feat_cols)
        X_all = np.zeros((T, S, C, F2, F), dtype=np.float32)

        time_to_idx = {t:i for i,t in enumerate(times)}
        state_to_idx = {s:i for i,s in enumerate(states)}
        comm_to_idx = {c:i for i,c in enumerate(commodities)}
        flow_to_idx = {'Export':0, 'Import':1}

        for (s, c, f, t), gdf in df_feat.groupby(['State','Commodity','Import/Export','Time']):
            if pd.isna(t) or s not in state_to_idx or c not in comm_to_idx or f not in flow_to_idx:
                continue
            ti = time_to_idx.get(t, None)
            if ti is None: continue
            si = state_to_idx[s]; ci = comm_to_idx[c]; fi = flow_to_idx[f]
            feats = gdf.sort_values('Time')[feat_cols].iloc[-1].to_numpy(dtype=np.float32)
            X_all[ti, si, ci, fi, :] = feats
        return X_all, states, commodities, times

    def _windows(self, X_all, times, L, target_start=None, target_end=None):
        """
        Create samples:
          X: (N, S, C, 2, L, Feat)
          y: (N, S, C, 2)
        selecting targets within [start,end]. Also returns target indices.
        """
        T = X_all.shape[0]
        idxs = []
        for t in range(L, T):
            ts = times[t]
            if target_start is not None and ts < target_start: continue
            if target_end   is not None and ts > target_end:   continue
            idxs.append(t)
        if not idxs:
            return (
                np.zeros((0,) + (X_all.shape[1], X_all.shape[2], X_all.shape[3], L, X_all.shape[4]), np.float32),
                np.zeros((0, X_all.shape[1], X_all.shape[2], X_all.shape[3], 2), np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        X = np.stack([X_all[t-L:t, ...] for t in idxs], axis=0)          # (N, L, S, C, 2, F)
        X = np.transpose(X, (0, 2, 3, 4, 1, 5)).astype(np.float32)        # (N, S, C, 2, L, F)
        y = np.stack([X_all[t, :, :, :, :2] for t in idxs], axis=0).astype(np.float32)  # (N, S, C, 2)
        return X, y, np.asarray(idxs, dtype=np.int64)

    def setup(self):
        # Import shared helpers
        import sys
        sys.path.append('/root')
        from shared_dataloader import (
            read_folder, build_grid, temporal_split_calendar,
            compute_series_stats, apply_stats_with_backoff,
            add_lags, feature_view
        )

        # IO
        imports = read_folder(self.imports_dir, "Import")
        exports = read_folder(self.exports_dir, "Export")
        panel  = pd.concat([imports, exports], ignore_index=True)
        grid   = build_grid(panel, self.start, self.end)

        # Splits
        tr_raw, va_raw, te_raw = temporal_split_calendar(grid, self)

        # Train-only stats
        stats = compute_series_stats(tr_raw)
        tr = apply_stats_with_backoff(tr_raw, stats)
        va = apply_stats_with_backoff(va_raw, stats)
        te = apply_stats_with_backoff(te_raw, stats)

        # Lags and features
        tr = add_lags(tr, 12)
        va = add_lags(va, 12)
        te = add_lags(te, 12)
        trf, feat_cols, _ = feature_view(tr)
        vaf, _, _ = feature_view(va)
        tef, _, _ = feature_view(te)

        # Build dense tensors with explicit (S, C, F2) axes
        X_tr_all, states_tr, comms_tr, times_tr = self._build_scflow_tensor(trf, feat_cols)

        # Val context uses train∪val; targets restricted to val
        trva = pd.concat([trf, vaf], ignore_index=True)
        X_va_all, states_va, comms_va, times_va = self._build_scflow_tensor(trva, feat_cols)

        # Test context uses train∪val∪test; targets restricted to test
        trvate = pd.concat([trf, vaf, tef], ignore_index=True)
        X_te_all, states_te, comms_te, times_te = self._build_scflow_tensor(trvate, feat_cols)

        # Sanity: align sets across splits
        assert states_tr == states_va == states_te, "State sets differ across splits"
        assert comms_tr == comms_va == comms_te, "Commodity sets differ across splits"

        L = int(self.input_len)
        t_val_start = pd.Timestamp(self.split_val_start)
        t_val_end   = pd.Timestamp(self.split_val_end)
        t_test_start= pd.Timestamp(self.split_test_start)
        t_test_end  = pd.Timestamp(self.split_test_end)

        X_tr, y_tr, idx_tr = self._windows(X_tr_all, times_tr, L, target_start=None, target_end=pd.Timestamp(self.split_train_end))
        X_va, y_va, idx_va = self._windows(X_va_all, times_va, L, target_start=t_val_start, target_end=t_val_end)
        X_te, y_te, idx_te = self._windows(X_te_all, times_te, L, target_start=t_test_start, target_end=t_test_end)

        class _Agg6DAllDS(Dataset):
            def __init__(self, X, Y):
                self.X = X; self.Y = Y
            def __len__(self): return self.X.shape[0]
            def __getitem__(self, i):
                return (
                    torch.from_numpy(self.X[i]),  # (S, C, 2, L, F)
                    torch.from_numpy(self.Y[i]),  # (S, C, 2)
                )

        self.dataset_train = _Agg6DAllDS(X_tr, y_tr)
        self.dataset_val   = _Agg6DAllDS(X_va, y_va)
        self.dataset_test  = _Agg6DAllDS(X_te, y_te)

        # Update input feature dim
        self.d_input = X_tr.shape[-1]

    def __str__(self):
        return "Agg6DAll"


class Agg4DAllBag(SequenceDataset):
    """
    4D windows over State with CF bag features:
    - X: (batch, S, L, F) numeric features averaged across (Commodity, Flow)
    - Plus CF bag extras passed to encoder:
      comm_counts: (batch, S, L, C) and flow_counts: (batch, S, L, 2)
    - y: (batch, S, 2) next-step [Value, Weight] averaged across (Commodity, Flow)

    CF bag extras are consumed by CFBagEncoder which appends a 32-d projected
    commodity bag and a 2-d flow count one-hot to the last feature dim.
    """

    _name_ = "s4nd_agg4d_bag"
    d_input = 28  # [Value, Weight] + 12 value lags + 12 weight lags + sin/cos
    _collate_arg_names = ["comm_counts", "flow_counts"]

    @property
    def d_output(self):
        # After setup, produce per-(Commodity, Flow, Target) flattened: C * 2 flows * 2 targets
        n_comms = getattr(self, 'n_comms', None)
        if n_comms is None:
            return 2
        return int(n_comms) * 4

    @property
    def l_output(self):
        return 0

    @property
    def init_defaults(self):
        return {
            "imports_dir": "/root/data/imports",
            "exports_dir": "/root/data/exports",
            "start": "2008-01-01",
            "end":   "2025-05-01",
            "input_len": 36,
            "horizon": 1,
            "target": "both",
            # Calendar splits
            "split_train_end": "2023-12-01",
            "split_val_start": "2024-01-01",
            "split_val_end":   "2024-12-01",
            "split_test_start": "2025-01-01",
            "split_test_end":   "2025-05-01",
        }

    def _build_state_agg_features(self, df_feat, feat_cols):
        """
        Build dense tensor aggregated over (Commodity, Flow):
          X_all: (T, S, F)
        We average feature vectors across groups per (State, Time) to keep z-scale.
        """
        states = sorted(df_feat['State'].dropna().unique().tolist())
        times = sorted([t for t in df_feat['Time'].unique().tolist() if pd.notna(t)])
        S = len(states); T = len(times); F = len(feat_cols)
        X_all = np.zeros((T, S, F), dtype=np.float32)

        time_to_idx = {t:i for i,t in enumerate(times)}
        state_to_idx = {s:i for i,s in enumerate(states)}

        # Average across (Commodity, Flow) per (State, Time)
        keys = ['State','Time']
        agg = df_feat.groupby(keys)[feat_cols].mean().reset_index()
        for _, r in agg.iterrows():
            t = r['Time']; s = r['State']
            if pd.isna(t) or s not in state_to_idx: continue
            ti = time_to_idx.get(t, None)
            if ti is None: continue
            si = state_to_idx[s]
            X_all[ti, si, :] = r[feat_cols].to_numpy(dtype=np.float32)
        return X_all, states, times

    def _build_comm_flow_counts(self, df_feat, states, times):
        """Return (comm_counts (T,S,C), flow_counts (T,S,2), commodities list)."""
        comms = sorted(df_feat['Commodity'].dropna().unique().tolist())
        C = len(comms)
        time_to_idx = {t: i for i, t in enumerate(times)}
        state_to_idx = {s: i for i, s in enumerate(states)}

        comm_counts = np.zeros((len(times), len(states), C), dtype=np.float32)
        for (s, t, c), g in df_feat.groupby(['State','Time','Commodity']):
            if pd.isna(t) or s not in state_to_idx or t not in time_to_idx or pd.isna(c):
                continue
            ti = time_to_idx[t]; si = state_to_idx[s]; ci = comms.index(c)
            comm_counts[ti, si, ci] += 1.0

        flows = ['Export','Import']
        flow_to_idx = {f:i for i,f in enumerate(flows)}
        flow_counts = np.zeros((len(times), len(states), 2), dtype=np.float32)
        for (s, t, f), g in df_feat.groupby(['State','Time','Import/Export']):
            if pd.isna(t) or s not in state_to_idx or t not in time_to_idx or f not in flow_to_idx:
                continue
            ti = time_to_idx[t]; si = state_to_idx[s]; fi = flow_to_idx[f]
            flow_counts[ti, si, fi] += 1.0

        return comm_counts, flow_counts, comms

    def _build_targets_scflow(self, df_feat):
        """Build targets per (State, Commodity, Flow): y_all (T,S,C,2,2) where last dim is [Value, Weight]."""
        VC = 'Value'; WC = 'Weight'
        states = sorted(df_feat['State'].dropna().unique().tolist())
        comms  = sorted(df_feat['Commodity'].dropna().unique().tolist())
        flows  = ['Export','Import']
        times  = sorted([t for t in df_feat['Time'].unique().tolist() if pd.notna(t)])
        S = len(states); C = len(comms); F2 = 2; T = len(times)
        y_all = np.zeros((T, S, C, F2, 2), dtype=np.float32)

        time_to_idx = {t:i for i,t in enumerate(times)}
        state_to_idx = {s:i for i,s in enumerate(states)}
        comm_to_idx = {c:i for i,c in enumerate(comms)}
        flow_to_idx = {'Export':0,'Import':1}

        for (s, c, f, t), g in df_feat.groupby(['State','Commodity','Import/Export','Time']):
            if pd.isna(t) or s not in state_to_idx or c not in comm_to_idx or f not in flow_to_idx:
                continue
            ti = time_to_idx.get(t, None)
            if ti is None: continue
            si = state_to_idx[s]; ci = comm_to_idx[c]; fi = flow_to_idx[f]
            row = g.sort_values('Time').iloc[-1]
            y_all[ti, si, ci, fi, 0] = float(row[VC])
            y_all[ti, si, ci, fi, 1] = float(row[WC])
        return y_all, states, comms, times

    def _windows(self, X_all, y_all, comm_counts, flow_counts, times, L, target_start=None, target_end=None):
        """
        Create samples:
          X: (N, S, L, F)
          y: (N, S, 2)
          comm_counts_w: (N, S, L, C)
          flow_counts_w: (N, S, L, 2)
        selecting targets within [start,end]. Also returns target indices.
        """
        T = X_all.shape[0]
        idxs = []
        for t in range(L, T):
            ts = times[t]
            if target_start is not None and ts < target_start: continue
            if target_end   is not None and ts > target_end:   continue
            idxs.append(t)
        if not idxs:
            zN = np.zeros((0,), dtype=np.int64)
            return (
                np.zeros((0, X_all.shape[1], L, X_all.shape[2]), np.float32),
                np.zeros((0, X_all.shape[1], 2), np.float32),
                np.zeros((0, X_all.shape[1], L, comm_counts.shape[-1]), np.float32),
                np.zeros((0, X_all.shape[1], L, flow_counts.shape[-1]), np.float32),
                zN,
            )
        X = np.stack([X_all[t-L:t, ...] for t in idxs], axis=0)          # (N, L, S, F)
        X = np.transpose(X, (0, 2, 1, 3)).astype(np.float32)              # (N, S, L, F)
        y = np.stack([y_all[t, ...] for t in idxs], axis=0).astype(np.float32)   # (N, S, C, 2, 2)
        NS, NC = y.shape[1], y.shape[2]
        y = y.reshape(y.shape[0], NS, NC*2*2).astype(np.float32)                 # (N, S, C*4)
        CC = np.stack([comm_counts[t-L:t, ...] for t in idxs], axis=0)    # (N, L, S, C)
        CC = np.transpose(CC, (0, 2, 1, 3)).astype(np.float32)            # (N, S, L, C)
        FC = np.stack([flow_counts[t-L:t, ...] for t in idxs], axis=0)    # (N, L, S, 2)
        FC = np.transpose(FC, (0, 2, 1, 3)).astype(np.float32)            # (N, S, L, 2)
        return X, y, CC, FC, np.asarray(idxs, dtype=np.int64)

    def setup(self):
        # Import shared helpers
        import sys
        sys.path.append('/root')
        from shared_dataloader import (
            read_folder, build_grid, temporal_split_calendar,
            compute_series_stats, apply_stats_with_backoff,
            add_lags, feature_view
        )

        # IO
        imports = read_folder(self.imports_dir, "Import")
        exports = read_folder(self.exports_dir, "Export")
        panel  = pd.concat([imports, exports], ignore_index=True)
        grid   = build_grid(panel, self.start, self.end)

        # Splits
        tr_raw, va_raw, te_raw = temporal_split_calendar(grid, self)

        # Train-only stats
        stats = compute_series_stats(tr_raw)
        tr = apply_stats_with_backoff(tr_raw, stats)
        va = apply_stats_with_backoff(va_raw, stats)
        te = apply_stats_with_backoff(te_raw, stats)

        # Lags and features
        tr = add_lags(tr, 12)
        va = add_lags(va, 12)
        te = add_lags(te, 12)
        trf, feat_cols, _ = feature_view(tr)
        vaf, _, _ = feature_view(va)
        tef, _, _ = feature_view(te)

        # Build aggregated per-state dense tensors and per-(S,C,Flow) targets
        X_tr_all, states_tr, times_tr = self._build_state_agg_features(trf, feat_cols)
        y_tr_all, states_y_tr, comms_tr, _ = self._build_targets_scflow(trf)

        trva = pd.concat([trf, vaf], ignore_index=True)
        X_va_all, states_va, times_va = self._build_state_agg_features(trva, feat_cols)
        y_va_all, states_y_va, comms_va, _ = self._build_targets_scflow(trva)

        trvate = pd.concat([trf, vaf, tef], ignore_index=True)
        X_te_all, states_te, times_te = self._build_state_agg_features(trvate, feat_cols)
        y_te_all, states_y_te, comms_te, _ = self._build_targets_scflow(trvate)

        # Sanity: align states across splits
        assert states_tr == states_va == states_te == states_y_tr == states_y_va == states_y_te, "State sets differ across splits"
        assert comms_tr == comms_va == comms_te, "Commodity sets differ across splits"

        # Build CF counts using full contexts for respective splits
        CC_tr_all, FC_tr_all, comms_tr = self._build_comm_flow_counts(trf, states_tr, times_tr)
        CC_va_all, FC_va_all, comms_va = self._build_comm_flow_counts(trva, states_va, times_va)
        CC_te_all, FC_te_all, comms_te = self._build_comm_flow_counts(trvate, states_te, times_te)
        assert comms_tr == comms_va == comms_te, "Commodity sets differ across splits"

        L = int(self.input_len)
        t_val_start = pd.Timestamp(self.split_val_start)
        t_val_end   = pd.Timestamp(self.split_val_end)
        t_test_start= pd.Timestamp(self.split_test_start)
        t_test_end  = pd.Timestamp(self.split_test_end)

        X_tr, y_tr, CC_tr, FC_tr, _ = self._windows(X_tr_all, y_tr_all, CC_tr_all, FC_tr_all, times_tr, L, target_start=None, target_end=pd.Timestamp(self.split_train_end))
        X_va, y_va, CC_va, FC_va, _ = self._windows(X_va_all, y_va_all, CC_va_all, FC_va_all, times_va, L, target_start=t_val_start, target_end=t_val_end)
        X_te, y_te, CC_te, FC_te, _ = self._windows(X_te_all, y_te_all, CC_te_all, FC_te_all, times_te, L, target_start=t_test_start, target_end=t_test_end)

        class _Agg4DAllBagDS(Dataset):
            def __init__(self, X, Y, CC, FC):
                self.X = X; self.Y = Y; self.CC = CC; self.FC = FC
            def __len__(self): return self.X.shape[0]
            def __getitem__(self, i):
                return (
                    torch.from_numpy(self.X[i]),  # (S, L, F)
                    torch.from_numpy(self.Y[i]),  # (S, 2)
                    torch.from_numpy(self.CC[i]),  # (S, L, C)
                    torch.from_numpy(self.FC[i]),  # (S, L, 2)
                )

        self.dataset_train = _Agg4DAllBagDS(X_tr, y_tr, CC_tr, FC_tr)
        self.dataset_val   = _Agg4DAllBagDS(X_va, y_va, CC_va, FC_va)
        self.dataset_test  = _Agg4DAllBagDS(X_te, y_te, CC_te, FC_te)

        # Meta for encoders
        self.n_comms = CC_tr_all.shape[-1]
        # Account for extra encoder-concatenated dims (e.g., CFBag projections)
        extra = getattr(self, 'encoder_extra_dim', 0)
        try:
            extra_int = int(extra)
        except Exception:
            extra_int = 0
        self.d_input = X_tr.shape[-1] + extra_int

    def __str__(self):
        return "Agg4DAllBag"

class Agg5DAll(SequenceDataset):
    """
    5D with explicit (State, Commodity) axes and per-flow training via flow embedding:
    - X: (batch, S, C, L, F) where F = 28 core features (per-flow; no aggregation)
    - y: (batch, S, C, 2) next-step [Value, Weight] for the chosen flow
    - Extra collate arg: flow_ids (B, S, C) with values {0:Export,1:Import} to be consumed by FlowIDEncoder
    """

    _name_ = "s4nd_agg5d"
    d_input = 28  # core features only; flow embedding appended via encoder
    _collate_arg_names = ['state_ids', 'commodity_ids', 'flow_ids']

    @property
    def d_output(self):
        return 2

    @property
    def l_output(self):
        return 0

    @property
    def init_defaults(self):
        return {
            "imports_dir": "/root/data/imports",
            "exports_dir": "/root/data/exports",
            "start": "2008-01-01",
            "end":   "2025-05-01",
            "input_len": 36,
            "horizon": 1,
            "target": "both",
            # Calendar splits
            "split_train_end": "2023-12-01",
            "split_val_start": "2024-01-01",
            "split_val_end":   "2024-12-01",
            "split_test_start": "2025-01-01",
            "split_test_end":   "2025-05-01",
        }

    def _build_sc_tensor_per_flow(self, df_feat, feat_cols):
        """
        Build dense tensor with explicit axes State × Commodity × Flow under each timestamp (no aggregation):
          X_all: (T, S, C, 2, F) where F=len(feat_cols)
        Also construct id grids for state and commodity.
        Missing entries are zeros.
        """
        states = sorted(df_feat['State'].dropna().unique().tolist())
        commodities = sorted(df_feat['Commodity'].dropna().unique().tolist())
        flows = ['Export', 'Import']
        times = sorted([t for t in df_feat['Time'].unique().tolist() if pd.notna(t)])

        S = len(states); C = len(commodities); T = len(times); F = len(feat_cols)
        X_all = np.zeros((T, S, C, 2, F), dtype=np.float32)

        time_to_idx = {t:i for i,t in enumerate(times)}
        state_to_idx = {s:i for i,s in enumerate(states)}
        comm_to_idx = {c:i for i,c in enumerate(commodities)}
        flow_to_idx = {'Export':0, 'Import':1}

        # For each (S,C,Flow,Time), write features
        for (s, c, f, t), gdf in df_feat.groupby(['State','Commodity','Import/Export','Time']):
            if pd.isna(t) or s not in state_to_idx or c not in comm_to_idx or f not in flow_to_idx:
                continue
            ti = time_to_idx.get(t, None)
            if ti is None: continue
            si = state_to_idx[s]; ci = comm_to_idx[c]; fi = flow_to_idx[f]
            feats = gdf.sort_values('Time')[feat_cols].iloc[-1].to_numpy(dtype=np.float32)
            X_all[ti, si, ci, fi, :] = feats

        # Build id grids (S,C)
        state_ids = np.arange(S, dtype=np.int64)[:, None].repeat(C, axis=1)
        comm_ids = np.arange(C, dtype=np.int64)[None, :].repeat(S, axis=0)
        return X_all, states, commodities, times, state_ids, comm_ids

    def _build_targets_per_flow(self, df_feat):
        VC = 'Value'; WC = 'Weight'
        states = sorted(df_feat['State'].dropna().unique().tolist())
        commodities = sorted(df_feat['Commodity'].dropna().unique().tolist())
        flows = ['Export','Import']
        times = sorted([t for t in df_feat['Time'].unique().tolist() if pd.notna(t)])

        S = len(states); C = len(commodities); T = len(times)
        y_all = np.zeros((T, S, C, 2, 2), dtype=np.float32)  # (T,S,C,Flow,Targets)

        t2i = {t:i for i,t in enumerate(times)}
        s2i = {s:i for i,s in enumerate(states)}
        c2i = {c:i for i,c in enumerate(commodities)}
        f2i = {'Export':0, 'Import':1}

        for (s, c, f, t), gdf in df_feat.groupby(['State','Commodity','Import/Export','Time']):
            if pd.isna(t) or s not in s2i or c not in c2i or f not in f2i: continue
            ti = t2i.get(t, None)
            if ti is None: continue
            si = s2i[s]; ci = c2i[c]; fi = f2i[f]
            row = gdf.sort_values('Time').iloc[-1]
            y_all[ti, si, ci, fi, 0] = float(row[VC]) if pd.notna(row[VC]) else 0.0
            y_all[ti, si, ci, fi, 1] = float(row[WC]) if pd.notna(row[WC]) else 0.0

        return y_all, states, commodities, times

    def _windows(self, X_all, y_all, times, L, target_start=None, target_end=None):
        T = X_all.shape[0]
        idxs = []
        for t in range(L, T):
            ts = times[t]
            if target_start is not None and ts < target_start: continue
            if target_end   is not None and ts > target_end:   continue
            idxs.append(t)
        if not idxs:
            return (
                np.zeros((0, X_all.shape[1], X_all.shape[2], L, X_all.shape[3]), np.float32),
                np.zeros((0, X_all.shape[1], X_all.shape[2], 2), np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        X = np.stack([X_all[t-L:t, ...] for t in idxs], axis=0)          # (N, L, S, C, Fp)
        X = np.transpose(X, (0, 2, 3, 1, 4)).astype(np.float32)           # (N, S, C, L, Fp)
        y = np.stack([y_all[t, ...] for t in idxs], axis=0).astype(np.float32)  # (N, S, C, 2)
        return X, y, np.asarray(idxs, dtype=np.int64)

    def setup(self):
        import sys
        sys.path.append('/root')
        from shared_dataloader import (
            read_folder, build_grid, temporal_split_calendar,
            compute_series_stats, apply_stats_with_backoff,
            add_lags, feature_view
        )

        imports = read_folder(self.imports_dir, "Import")
        exports = read_folder(self.exports_dir, "Export")
        panel  = pd.concat([imports, exports], ignore_index=True)
        grid   = build_grid(panel, self.start, self.end)

        tr_raw, va_raw, te_raw = temporal_split_calendar(grid, self)

        stats = compute_series_stats(tr_raw)
        tr = apply_stats_with_backoff(tr_raw, stats)
        va = apply_stats_with_backoff(va_raw, stats)
        te = apply_stats_with_backoff(te_raw, stats)

        tr = add_lags(tr, 12)
        va = add_lags(va, 12)
        te = add_lags(te, 12)
        trf, feat_cols, _ = feature_view(tr)
        vaf, _, _ = feature_view(va)
        tef, _, _ = feature_view(te)

        X_tr_all, states_tr, comms_tr, times_tr, sid_grid, cid_grid = self._build_sc_tensor_per_flow(trf, feat_cols)
        trva = pd.concat([trf, vaf], ignore_index=True)
        X_va_all, states_va, comms_va, times_va, _, _ = self._build_sc_tensor_per_flow(trva, feat_cols)
        trvate = pd.concat([trf, vaf, tef], ignore_index=True)
        X_te_all, states_te, comms_te, times_te, _, _ = self._build_sc_tensor_per_flow(trvate, feat_cols)

        # Targets per flow
        y_tr_all, _, _, _ = self._build_targets_per_flow(trf)
        y_va_all, _, _, _ = self._build_targets_per_flow(trva)
        y_te_all, _, _, _ = self._build_targets_per_flow(trvate)

        assert states_tr == states_va == states_te, "State sets differ across splits"
        assert comms_tr == comms_va == comms_te, "Commodity sets differ across splits"

        L = int(self.input_len)
        t_val_start = pd.Timestamp(self.split_val_start)
        t_val_end   = pd.Timestamp(self.split_val_end)
        t_test_start= pd.Timestamp(self.split_test_start)
        t_test_end  = pd.Timestamp(self.split_test_end)

        # Build windows duplicated across flows
        def windows_per_flow(X_all, y_all, times, start, end):
            T = X_all.shape[0]
            idxs = []
            for t in range(L, T):
                ts = times[t]
                if start is not None and ts < start: continue
                if end   is not None and ts > end:   continue
                idxs.append(t)
            if not idxs:
                return (
                    np.zeros((0, X_all.shape[1], X_all.shape[2], L, X_all.shape[4]), np.float32),
                    np.zeros((0, X_all.shape[1], X_all.shape[2], 2), np.float32),
                    np.zeros((0, X_all.shape[1], X_all.shape[2]), np.int64),
                )
            X_list = []
            Y_list = []
            FID_list = []
            for t in idxs:
                for fid in (0,1):
                    x_seg = X_all[t-L:t, :, :, fid, :]  # (L,S,C,F)
                    x_seg = np.transpose(x_seg, (1,2,0,3)).astype(np.float32)  # (S,C,L,F)
                    y_seg = y_all[t, :, :, fid, :].astype(np.float32)          # (S,C,2)
                    X_list.append(x_seg)
                    Y_list.append(y_seg)
                    FID_list.append(np.full((X_all.shape[1], X_all.shape[2]), fid, dtype=np.int64))
            return np.stack(X_list, axis=0), np.stack(Y_list, axis=0), np.stack(FID_list, axis=0)

        X_tr, y_tr, fid_tr = windows_per_flow(X_tr_all, y_tr_all, times_tr, None, pd.Timestamp(self.split_train_end))
        X_va, y_va, fid_va = windows_per_flow(X_va_all, y_va_all, times_va, t_val_start, t_val_end)
        X_te, y_te, fid_te = windows_per_flow(X_te_all, y_te_all, times_te, t_test_start, t_test_end)

        # Build (B,S,C) id tensors
        S, C = sid_grid.shape
        def tile_ids(n):
            sid = np.tile(sid_grid[None, ...], (n, 1, 1))
            cid = np.tile(cid_grid[None, ...], (n, 1, 1))
            return sid, cid

        sid_tr, cid_tr = tile_ids(X_tr.shape[0])
        sid_va, cid_va = tile_ids(X_va.shape[0])
        sid_te, cid_te = tile_ids(X_te.shape[0])

        class _Agg5DAllDS(Dataset):
            def __init__(self, X, Y, SID, CID, FID):
                self.X = X; self.Y = Y; self.SID = SID; self.CID = CID; self.FID = FID
            def __len__(self): return self.X.shape[0]
            def __getitem__(self, i):
                return (
                    torch.from_numpy(self.X[i]),  # (S, C, L, F)
                    torch.from_numpy(self.Y[i]),  # (S, C, 2)
                    torch.from_numpy(self.SID[i]),
                    torch.from_numpy(self.CID[i]),
                    torch.from_numpy(self.FID[i]),
                )

        self.dataset_train = _Agg5DAllDS(X_tr, y_tr, np.tile(sid_grid[None,...], (X_tr.shape[0],1,1)), np.tile(cid_grid[None,...], (X_tr.shape[0],1,1)), fid_tr)
        self.dataset_val   = _Agg5DAllDS(X_va, y_va, np.tile(sid_grid[None,...], (X_va.shape[0],1,1)), np.tile(cid_grid[None,...], (X_va.shape[0],1,1)), fid_va)
        self.dataset_test  = _Agg5DAllDS(X_te, y_te, np.tile(sid_grid[None,...], (X_te.shape[0],1,1)), np.tile(cid_grid[None,...], (X_te.shape[0],1,1)), fid_te)

        self.n_states = S
        self.n_comms = C
        # Account for extra encoder-concatenated dims (e.g., FlowID/SCID embeddings)
        extra = getattr(self, 'encoder_extra_dim', 0)
        try:
            extra_int = int(extra)
        except Exception:
            extra_int = 0
        self.d_input = X_tr.shape[-1] + extra_int

    def __str__(self):
        return "Agg5DAll"

class Barge3D_old(SequenceDataset):
    """Barge3D"""

    _name_ = "Barge3D_old"
    d_input = 24

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def l_output(self):
        return 4000 if self.prediction else 0

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": False,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        combined_df, _ = load_and_preprocess_barrage_data(self.data_dir)
    
        # Initialize the barrage dataset
        window_size = 24
        barrage_dataset = BarrageSystemDataset(combined_df, window_size=window_size)
        
        # Use same dimensions as LSTM - no reshaping
        d_input_t = barrage_dataset.num_base_features  # Number of features per time step
        d_output_t = barrage_dataset.num_target_features  # Number of target features
        
        print(f"S4 Model dimensions (same as LSTM):")
        print(f"  d_input: {d_input_t} (num_base_features)")
        print(f"  d_output: {d_output_t} (num_target_features)")
        print(f"  num_combos: {barrage_dataset.num_combos}")
        print(f"  window_size: {window_size}")
        print(f"  Expected input shape: (batch_size, {barrage_dataset.num_combos}, {window_size}, {d_input_t})")
        print(f"  Expected output shape: (batch_size, {barrage_dataset.num_combos}, {d_output_t})")
        
        # Wrap with S4 dataset format (no reshaping)
        s4_dataset = S4BarrageDataset(barrage_dataset)
        
        N = len(s4_dataset)
        
        # Calculate split sizes
        train_size = int(N * 0.85)            # 85% train
        temp_size = N - train_size            # 15% combined val + test
        
        val_size = int(temp_size * (1/3))    # 5% val
        test_size = temp_size - val_size      # 10% test
        
        # Indices for each split
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, N))

        inputs = []
        labels = []
        for i in range(len(s4_dataset)):
            x, y = s4_dataset[i]
            inputs.append(x.numpy() if torch.is_tensor(x) else x)
            labels.append(y.numpy() if torch.is_tensor(y) else y)        
        X_array_b = np.stack(inputs)
        y_array_b = np.stack(labels)

        # Check tensor dimensions and handle appropriately
        print(f"X_array_b shape: {X_array_b.shape}")
        print(f"y_array_b shape: {y_array_b.shape}")

        # For S4, we need to preserve the tensor structure
        # X_array_b should be (N, num_combos, window_size, num_base_features)
        # y_array_b should be (N, num_combos, num_target_features)
        X_array = X_array_b
        y_array = y_array_b

        print(f"X_array is {X_array.shape}")
        print(f"y_array is {y_array.shape}")
        # Slice your arrays/tensors using those indices
        X_train = X_array[train_indices]
        y_train = y_array[train_indices]
        
        X_val = X_array[val_indices]
        y_val = y_array[val_indices]
        
        X_test = X_array[test_indices]
        y_test = y_array[test_indices]
        
        # Now create TensorDatasets from these splits
        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train).contiguous(),
            torch.FloatTensor(y_train).contiguous()
        )
        
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val).contiguous(),
            torch.FloatTensor(y_val).contiguous()
        )
        
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test).contiguous(),
            torch.FloatTensor(y_test).contiguous()
        )
    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"Barge3D_old"


def load_and_preprocess_barrage_data(data_dir):
    csv_files = []
    for subdir in ['import', 'export']:
        pattern = data_dir / subdir / "*.csv"
        csv_files.extend(glob.glob(str(pattern)))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    data_frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        #df = pd.read_csv(csv_file, skiprows=2, header=0)
        if 'Import/Export' not in df.columns:
            df['Import/Export'] = 1 if 'import' in csv_file.lower() else 0
        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df['time_parsed'] = combined_df['Time'].apply(parse_year_month)
    combined_df = combined_df.dropna(subset=['time_parsed']).sort_values('time_parsed')

    numeric_cols = ['Vessel Value ($US)', 'Vessel SWT (kg)']
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str).str.replace(',', '').astype(float)

    unique_states = sorted(combined_df['State'].unique())
    unique_commodities = sorted(combined_df['Commodity'].unique())
    unique_trade_dirs = sorted(combined_df['Import/Export'].unique())
    n_features = len(numeric_cols)

    d_input = len(unique_states) * len(unique_commodities) * len(unique_trade_dirs) * n_features
    return combined_df, d_input

def parse_year_month(time_str):
    try:
        parts = str(time_str).split('-')
        if len(parts) == 2:
            year_part, month_part = parts
            year = int(year_part)
            if year <= 25:
                year += 2000
            else:
                year += 2000 if year < 100 else 0
            month_num = pd.to_datetime(month_part, format='%b').month
            return pd.Timestamp(year=year, month=month_num, day=1)
    except:
        pass
    return pd.NaT

class BarrageSystemDataset(Dataset):
    """
    Dataset for barrage system time series forecasting.
    
    Creates tensors with shape:
    - Input Tensor (X): (batch, nodes, features) e.g., (48, 91, 24)
    - Target Tensor (y): (batch, nodes, 2) e.g., (48, 91, 2)
    
    Where:
    - batch: Number of valid prediction windows 
    - nodes: Unique combinations of (State, Import/Export, Commodity)
    - features: Lagged features flattened (window_size * 2)
    - target: [next_vessel_value, next_vessel_weight]
    """
    def __init__(self, 
                 df: pd.DataFrame,
                 window_size: int = 12,
                 target_cols: List[str] = None):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame with time series data
            window_size: Number of months to include in lag window
            target_cols: List of columns to predict [value_col, weight_col]
        """
        self.window_size = window_size
        
        # Default target columns
        if target_cols is None:
            target_cols = ['Containerized Vessel Total Exports Value ($US)', 'Containerized Vessel Total Exports SWT (kg)']
        
        self.target_cols = target_cols
        self.base_features = target_cols  # Use same columns for features and targets
        self.num_base_features = len(self.base_features)
        self.num_target_features = len(self.target_cols)
        
        # Total input features = base_features * window_size 
        self.num_total_features = self.num_base_features * window_size
        
        # Create a copy to avoid modifying original
        self.df = df.copy()
        
        print(f"Original data shape: {self.df.shape}")
        
        # Clean numeric columns - remove commas and convert to float
        numeric_cols = ['Vessel Value ($US)', 'Containerized Vessel Total Exports Value ($US)', 
                       'Vessel SWT (kg)', 'Containerized Vessel Total Exports SWT (kg)']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.replace(',', '').astype(float)
        
        # COLLAPSE COUNTRIES: Aggregate by State, Import/Export, Commodity, Time
        print("Collapsing countries - aggregating by State, Import/Export, Commodity, Time...")
        
        groupby_cols = ['State', 'Import/Export', 'Commodity', 'Time']
        agg_dict = {col: 'sum' for col in numeric_cols if col in self.df.columns}
        
        # Aggregate the data
        self.df = self.df.groupby(groupby_cols, as_index=False).agg(agg_dict)
        
        print(f"After country collapse - data shape: {self.df.shape}")
        print(f"Using features for lags: {self.base_features}")
        print(f"Target columns to predict: {self.target_cols}")
        print(f"Total input features with lags: {self.num_total_features}")
        
        # Process time column: expect format like '8-Jan' (year-month)
        print(f"Sample time values: {self.df['Time'].head().tolist()}")
        
        try:
            def parse_year_month(time_str):
                parts = str(time_str).split('-')
                if len(parts) == 2:
                    year_part, month_part = parts
                    # Convert 2-digit year to 4-digit (08->2008, 25->2025)
                    year = int(year_part)
                    if year <= 25:  # Assume 00-25 means 2000-2025
                        year += 2000
                    else:  # 8-99 -> 2008-2099
                        year += 2000
                    
                    # Get month number from abbreviation
                    month_num = pd.to_datetime(month_part, format='%b').month
                    
                    # Create timestamp for first day of month (but we'll treat as monthly)
                    return pd.Timestamp(year=year, month=month_num, day=1)
                return pd.NaT
            
            self.df['time_parsed'] = self.df['Time'].apply(parse_year_month)
            
            # Remove any NaT values
            self.df = self.df.dropna(subset=['time_parsed'])
            
            print(f"Time parsing successful. Sample parsed times: {sorted(self.df['time_parsed'].unique())[:5]} ... {sorted(self.df['time_parsed'].unique())[-5:]}")
            
        except Exception as e:
            print(f"Warning: Could not parse time format: {e}")
            print("Creating sequential time index...")
            unique_times = self.df['Time'].unique()
            time_mapping = {time: idx for idx, time in enumerate(sorted(unique_times))}
            self.df['time_parsed'] = self.df['Time'].map(time_mapping)
        
        # Create combo keys: (State, Import/Export, Commodity)
        self.df['combo_key'] = self.df.apply(
            lambda row: (row['State'], row['Import/Export'], row['Commodity']), 
            axis=1
        )
        
        # Get unique times and combos in correct order
        all_times = sorted(self.df['time_parsed'].unique())
        all_combos = sorted(self.df['combo_key'].unique())
        
        # Create mappings
        self.time_to_idx = {t: i for i, t in enumerate(all_times)}
        self.combo_to_idx = {c: i for i, c in enumerate(all_combos)}
        
        # Store meta info
        self.num_times = len(all_times)
        self.num_combos = len(all_combos)
        
        print(f"Dataset info: {self.num_times} time points, {self.num_combos} combos")
        
        # Create the full tensor for features
        # Shape: (num_times, num_combos, num_base_features)
        self.data_tensor = np.zeros((self.num_times, self.num_combos, self.num_base_features), dtype=np.float32)
        
        # Create the full tensor for targets  
        # Shape: (num_times, num_combos, num_target_features)
        self.target_tensor = np.zeros((self.num_times, self.num_combos, self.num_target_features), dtype=np.float32)
        
        # Fill tensors from dataframe
        for _, row in self.df.iterrows():
            if row['time_parsed'] in self.time_to_idx and row['combo_key'] in self.combo_to_idx:
                t_idx = self.time_to_idx[row['time_parsed']]
                c_idx = self.combo_to_idx[row['combo_key']]
                
                # Fill feature values
                for f_idx, feat in enumerate(self.base_features):
                    if feat in row and not pd.isna(row[feat]):
                        self.data_tensor[t_idx, c_idx, f_idx] = row[feat]
                
                # Fill target values
                for t_idx_target, target_col in enumerate(self.target_cols):
                    if target_col in row and not pd.isna(row[target_col]):
                        self.target_tensor[t_idx, c_idx, t_idx_target] = row[target_col]
        
        # Normalize features using z-score normalization
        print("Computing z-score normalization statistics...")
        self.feature_stats = {}
        
        for f_idx, feature_name in enumerate(self.base_features):
            feat_data = self.data_tensor[:, :, f_idx]
            non_zero_data = feat_data[feat_data != 0]
            
            if len(non_zero_data) > 0:
                mean_val = np.mean(non_zero_data)
                std_val = np.std(non_zero_data)
                if std_val == 0:
                    std_val = 1.0
                
                self.feature_stats[feature_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(np.min(non_zero_data)),
                    'max': float(np.max(non_zero_data)),
                    'count': int(len(non_zero_data))
                }
                
                print(f"  {feature_name}: mean={mean_val:.2f}, std={std_val:.2f}")
                
                # Apply z-score normalization only to non-zero values
                mask = self.data_tensor[:, :, f_idx] != 0
                self.data_tensor[:, :, f_idx][mask] = (
                    (self.data_tensor[:, :, f_idx][mask] - mean_val) / std_val
                )
            else:
                self.feature_stats[feature_name] = {
                    'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        # Normalize targets using z-score normalization
        self.target_stats = {}
        
        for t_idx, target_name in enumerate(self.target_cols):
            target_data = self.target_tensor[:, :, t_idx]
            non_zero_data = target_data[target_data != 0]
            
            if len(non_zero_data) > 0:
                mean_val = np.mean(non_zero_data)
                std_val = np.std(non_zero_data)
                if std_val == 0:
                    std_val = 1.0
                
                self.target_stats[target_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(np.min(non_zero_data)),
                    'max': float(np.max(non_zero_data)),
                    'count': int(len(non_zero_data))
                }
                
                print(f"  {target_name}: mean={mean_val:.2f}, std={std_val:.2f}")
                
                # Apply z-score normalization only to non-zero values
                mask = self.target_tensor[:, :, t_idx] != 0
                self.target_tensor[:, :, t_idx][mask] = (
                    (self.target_tensor[:, :, t_idx][mask] - mean_val) / std_val
                )
            else:
                self.target_stats[target_name] = {
                    'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        # For backward compatibility, create the old format arrays
        self.feature_mean = np.zeros((1, 1, self.num_base_features))
        self.feature_std = np.ones((1, 1, self.num_base_features))
        self.target_mean = np.zeros((1, 1, self.num_target_features))
        self.target_std = np.ones((1, 1, self.num_target_features))
        
        for f_idx, feature_name in enumerate(self.base_features):
            self.feature_mean[0, 0, f_idx] = self.feature_stats[feature_name]['mean']
            self.feature_std[0, 0, f_idx] = self.feature_stats[feature_name]['std']
            
        for t_idx, target_name in enumerate(self.target_cols):
            self.target_mean[0, 0, t_idx] = self.target_stats[target_name]['mean']
            self.target_std[0, 0, t_idx] = self.target_stats[target_name]['std']

        # Create valid window indices 
        self.samples = []
        for target_t in range(window_size, self.num_times):
            start_t = target_t - window_size
            
            # Create lagged features for all combos - Keep temporal structure!
            # Shape: (num_combos, window_size, num_base_features)
            lagged_features = np.zeros((self.num_combos, window_size, self.num_base_features), dtype=np.float32)
            
            for combo_idx in range(self.num_combos):
                # Extract lag window for this combo
                lag_data = self.data_tensor[start_t:target_t, combo_idx, :]  # Shape: (window_size, num_base_features)
                
                # Keep temporal structure - time as sequence dimension
                lagged_features[combo_idx, :, :] = lag_data
            
            # Target values for this time step (BOTH value and weight)
            target_values = self.target_tensor[target_t, :, :]  # Shape: (num_combos, num_target_features)
            
            self.samples.append((lagged_features, target_values))
        
        print(f"Created {len(self.samples)} valid prediction windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lagged_features, target_values = self.samples[idx]
        
        return {
            'input': torch.tensor(lagged_features, dtype=torch.float32),   # Shape: (num_combos, window_size, num_base_features)
            'target': torch.tensor(target_values, dtype=torch.float32),    # Shape: (num_combos, num_target_features)
            'sample_idx': idx
        }
    
    def get_combo_info(self):
        """Returns mapping of combo indices to their original values"""
        return {i: combo for combo, i in self.combo_to_idx.items()}


class S4BarrageDataset(torch.utils.data.Dataset):
    def __init__(self, barrage_dataset):
        self.barrage_dataset = barrage_dataset
        
    def __len__(self):
        return len(self.barrage_dataset)
    
    def __getitem__(self, idx):
        batch = self.barrage_dataset[idx]
        # Keep the same format as LSTM: (num_combos, window_size, num_base_features)
        input_tensor = batch['input']   # Shape: (num_combos, window_size, num_base_features)
        target_tensor = batch['target'] # Shape: (num_combos, num_target_features)
        
        # No reshaping - S4 should handle the same tensor format as LSTM
        return input_tensor, target_tensor


class AGGFlat(SequenceDataset):
    """
    Minimal dataset to mirror the notebook pipeline:
    - Read CSV with 'Time' column and 2 target columns
    - MinMax scale BOTH targets over entire series
    - Create sequences with window_size steps to predict next-step 2-D target
    - Sequential 80/20 split (no shuffle)
    """

    _name_ = "agg_flat"
    d_input = 2

    @property
    def d_output(self):
        return 2

    @property
    def l_output(self):
        return 0

    @property
    def init_defaults(self):
        return {
            "file_name": "custom_data_org_sum.csv",
            "window_size": 12,
            # Calendar-based splits (inclusive bounds on target time)
            "train_end": "2023-12-01",
            "val_start": "2024-01-01",
            "val_end": "2024-12-01",
            "test_start": "2025-01-01",
            "test_end": "2025-05-01",
            "scale": True,
            "targets": [
                "Total Vessel Value ($US)",
                "Total Vessel SWT (kg)",
            ],
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        file_path = os.path.join(str(self.data_dir), self.file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"AGGFlat: CSV not found at {file_path}")

        df = pd.read_csv(file_path, parse_dates=["Time"], index_col=None)
        # Ensure chronological order
        if "Time" in df.columns:
            df = df.sort_values("Time")

        data = df[self.targets].astype(float).values  # (T, 2)
        times = pd.to_datetime(df["Time"]).values

        if self.scale:
            # MinMax scaling over entire series (to match notebook)
            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            denom = (max_vals - min_vals)
            denom[denom == 0.0] = 1.0
            data = (data - min_vals) / denom

        # Create sequences (X windows and y next-step), keep target timestamps
        N = self.window_size
        X_list, y_list, t_list = [], [], []
        for i in range(len(data) - N):
            X_list.append(data[i:i+N])
            y_list.append(data[i+N])
            t_list.append(times[i+N])
        if len(X_list) == 0:
            raise ValueError("AGGFlat: Not enough data to create any sequences.")

        X = np.asarray(X_list, dtype=np.float32)  # (num_seq, N, 2)
        y = np.asarray(y_list, dtype=np.float32)  # (num_seq, 2)
        t = np.asarray(t_list)                    # (num_seq,)

        # Calendar-based split by target timestamp
        t_train_end = np.datetime64(pd.Timestamp(self.train_end))
        t_val_start = np.datetime64(pd.Timestamp(self.val_start))
        t_val_end   = np.datetime64(pd.Timestamp(self.val_end))
        t_test_start= np.datetime64(pd.Timestamp(self.test_start))
        t_test_end  = np.datetime64(pd.Timestamp(self.test_end))

        mask_train = t <= t_train_end
        mask_val   = (t >= t_val_start) & (t <= t_val_end)
        mask_test  = (t >= t_test_start) & (t <= t_test_end)

        X_train, y_train = X[mask_train], y[mask_train]
        X_val, y_val     = X[mask_val], y[mask_val]
        X_test, y_test   = X[mask_test], y[mask_test]

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )

    def __str__(self):
        return "AGGFlat"


########################################
# Flat trade dataset (Plan A style)    #
########################################

from torch.utils.data import Dataset as _FlatTorchDataset
from src.dataloaders.shared_dataloader import (
    read_folder as _flat_read_folder,
    build_grid as _flat_build_grid,
    temporal_split_calendar as _flat_temporal_split_calendar,
    compute_series_stats as _flat_compute_series_stats,
    apply_stats_with_backoff as _flat_apply_stats_with_backoff,
    add_lags as _flat_add_lags,
    feature_view as _flat_feature_view,
    build_id_maps as _flat_build_id_maps,
)

_FLAT_VALUE = "Value"
_FLAT_WEIGHT = "Weight"


class _FlatSeriesWindowDataset(_FlatTorchDataset):
    """
    Build sliding windows per series (State, Commodity, Import/Export).
    Returns tuples (x, y, state_id, comm_id, flow_id) where
      - x: (L, F_numeric)
      - y: (1, Ft) with Ft=2 if target=="both" else 1
    """

    def __init__(self,
                 df_feat: pd.DataFrame,
                 feat_cols: List[str],
                 input_len: int,
                 horizon: int,
                 target: str,
                 state2id: Dict[str, int],
                 comm2id: Dict[str, int],
                 flow2id: Dict[str, int],
                 target_time_start: pd.Timestamp | None = None,
                 target_time_end: pd.Timestamp | None = None):
        super().__init__()
        assert target in ("value", "weight", "both")
        assert int(horizon) == 1, "Only horizon=1 supported in flat loader"

        self.feat_cols = list(feat_cols)
        self.input_len = int(input_len)
        self.target = target

        groups = df_feat.groupby(["State", "Commodity", "Import/Export"], sort=False)
        samples: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]] = []

        for (state, comm, flow), g in groups:
            g = g.sort_values("Time").reset_index(drop=True)
            X = g[self.feat_cols].to_numpy(dtype=np.float32)
            times = g["Time"].to_numpy()
            Tn = X.shape[0]
            if Tn < self.input_len + 1:
                continue

            sid = state2id.get(state, 0)
            cid = comm2id.get(comm, 0)
            fid = flow2id.get(flow, 0)

            cols = ([
                _FLAT_VALUE,
            ] if self.target == "value" else [
                _FLAT_WEIGHT,
            ] if self.target == "weight" else [
                _FLAT_VALUE, _FLAT_WEIGHT
            ])
            Y = g[cols].to_numpy(dtype=np.float32)

            for t in range(self.input_len, Tn):
                t_pred_time = times[t]
                if target_time_start is not None and t_pred_time < target_time_start:
                    continue
                if target_time_end is not None and t_pred_time > target_time_end:
                    continue
                x = X[t - self.input_len : t, :]
                y = Y[t : t + 1, :]
                samples.append((x, y, (sid, cid, fid)))

        if not samples:
            self.X = np.zeros((0, self.input_len, len(self.feat_cols)), dtype=np.float32)
            self.Y = np.zeros((0, 1, 2 if self.target == "both" else 1), dtype=np.float32)
            self.S = np.zeros((0,), dtype=np.int64)
            self.C = np.zeros((0,), dtype=np.int64)
            self.F = np.zeros((0,), dtype=np.int64)
        else:
            self.X = np.stack([s[0] for s in samples], axis=0)
            self.Y = np.stack([s[1] for s in samples], axis=0)
            ids = np.array([s[2] for s in samples], dtype=np.int64)
            self.S, self.C, self.F = ids[:, 0], ids[:, 1], ids[:, 2]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.Y[i]),
            torch.tensor(self.S[i], dtype=torch.long),
            torch.tensor(self.C[i], dtype=torch.long),
            torch.tensor(self.F[i], dtype=torch.long),
        )


class Flat(SequenceDataset):
    """
    Flat (B, L, F) windows with Plan A features (z-scored Value/Weight, 12 lags, sin/cos month).
    Provides IDs for encoders: SCID (state, commodity) and FlowID.

    Collate returns extra dict with keys ['state_ids','commodity_ids','flow_ids'].
    """

    _name_ = "flat"
    _collate_arg_names = ['state_ids', 'commodity_ids', 'flow_ids']

    @property
    def init_defaults(self):
        return {
            "imports_dir": "/root/data/imports",
            "exports_dir": "/root/data/exports",
            "start": "2008-01-01",
            "end":   "2025-05-01",
            "input_len": 36,
            "horizon": 1,
            "target": "both",  # value | weight | both
            # splits
            "split_train_end": "2023-12-01",
            "split_val_start": "2024-01-01",
            "split_val_end":   "2024-12-01",
            "split_test_start": "2025-01-01",
            "split_test_end":   "2025-05-01",
            # embedding dims (for computing d_input once encoders append)
            "state_emb_dim": 8,
            "commodity_emb_dim": 32,
            "flow_emb_dim": 2,
        }

    def setup(self):
        # 1) Read & build ID maps from full panel
        imports = _flat_read_folder(self.imports_dir, "Import")
        exports = _flat_read_folder(self.exports_dir, "Export")
        panel  = pd.concat([imports, exports], ignore_index=True)
        state2id, comm2id, flow2id = _flat_build_id_maps(panel)
        self.n_states, self.n_comms = len(state2id), len(comm2id)

        # 2) Grid + encodings
        grid = _flat_build_grid(panel, self.start, self.end)

        # 3) Calendar split
        tr_raw, va_raw, te_raw = _flat_temporal_split_calendar(grid, self)

        # 4) Train-only stats → normalize all splits
        stats = _flat_compute_series_stats(tr_raw)
        tr = _flat_apply_stats_with_backoff(tr_raw, stats)
        va = _flat_apply_stats_with_backoff(va_raw, stats)
        te = _flat_apply_stats_with_backoff(te_raw, stats)

        # 5) Add lags (12)
        tr = _flat_add_lags(tr, 12)
        va = _flat_add_lags(va, 12)
        te = _flat_add_lags(te, 12)

        # 6) Feature assembly
        trf, feat_cols, _ = _flat_feature_view(tr)
        vaf, _, _ = _flat_feature_view(va)
        tef, _, _ = _flat_feature_view(te)
        self._feat_cols = feat_cols

        # 7) Build flat windows with IDs (val/test see pre-split context via concat)
        def build_ds(df_feat, t_start, t_end):
            inner = _FlatSeriesWindowDataset(
                df_feat=df_feat,
                feat_cols=feat_cols,
                input_len=int(self.input_len),
                horizon=1,
                target=str(self.target),
                state2id=state2id,
                comm2id=comm2id,
                flow2id=flow2id,
                target_time_start=t_start,
                target_time_end=t_end,
            )
            class _Wrap(_FlatTorchDataset):
                def __init__(self, parent): self.p = parent
                def __len__(self): return len(self.p)
                def __getitem__(self, i):
                    x, y, sid, cid, fid = self.p[i]
                    return x, y, sid, cid, fid
            return _Wrap(inner)

        t_train_end = pd.Timestamp(self.split_train_end)
        t_val_start = pd.Timestamp(self.split_val_start)
        t_val_end   = pd.Timestamp(self.split_val_end)
        t_test_start= pd.Timestamp(self.split_test_start)
        t_test_end  = pd.Timestamp(self.split_test_end)

        tr_ds = build_ds(trf, None, t_train_end)
        trva  = pd.concat([trf, vaf], ignore_index=True)
        va_ds = build_ds(trva, t_val_start, t_val_end)
        trvate= pd.concat([trf, vaf, tef], ignore_index=True)
        te_ds = build_ds(trvate, t_test_start, t_test_end)

        self.dataset_train = tr_ds
        self.dataset_val   = va_ds
        self.dataset_test  = te_ds

        # 8) d_input = numeric features + embedding dims appended by encoders
        F_numeric = len(feat_cols)
        self.d_input = (
            F_numeric
            + int(self.state_emb_dim)
            + int(self.commodity_emb_dim)
            + int(self.flow_emb_dim)
        )

    def __str__(self):
        return "flat"

class UsImport(SequenceDataset):
    _name_ = "us_import"
    d_input = 1
    d_output = 2

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        data = pd.read_csv(self.data_dir / "data.csv")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test)
        )

class UsImportOneHot(SequenceDataset):
    _name_ = "us_import_onehot"
    d_input = 1
    d_output = 2

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        data = pd.read_csv(self.data_dir / "data.csv")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # One-hot encode X
        X_onehot = pd.get_dummies(X.flatten()).values
        
        X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test)
        )