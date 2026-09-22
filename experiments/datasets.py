import os
from typing import Optional, Tuple, Union

import gin
import json
import numpy as np
import pandas as pd
import netCDF4 as nc
from torch.utils.data import Dataset


@gin.configurable()
class ForecastDataset(Dataset):
    def __init__(self,
                 flag: str,
                 input_len: int,
                 output_len: int,
                 data_dir: str,
                 scale: bool,
                 y_start: int = 1,
                 train_size: Union[int, float] = 0.7,
                 val_size: Union[int, float] = 0.1,
                 test_size: Union[int, float] = 0.2,
                 variable_name: Optional[str] = None):
        assert flag in ('train', 'val', 'test')
        self.flag = flag
        self.data_dir = data_dir
        self.scale = scale
        self.input_len = input_len
        self.output_len = output_len
        self.y_start = y_start
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.variable_name = variable_name

        self.data = None
        self.time_feats = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        self.mean = None
        self.std = None
        self.load_data()

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'datasets', path)

    def _read_csv(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        # water
        df = pd.read_csv(path)
        df_stamp = df.iloc[:, :1].copy()
        data = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # (T, N)

        df_stamp.columns = ['time']
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        df_stamp['month'] = df_stamp['time'].dt.month
        df_stamp['day'] = df_stamp['time'].dt.day
        df_stamp['weekday'] = df_stamp['time'].dt.weekday
        df_stamp['hour'] = df_stamp['time'].dt.hour
        time_feats = df_stamp.drop(['time'], axis=1).to_numpy(dtype=np.float32)  # (T, Ct)
        return data, time_feats

    def _read_dat(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        # PEMS
        desc_path = os.path.join(os.path.dirname(path), 'desc.json')
        with open(desc_path, 'r') as f:
            desc = json.load(f)
        shape = tuple(desc.get('shape', []))

        mm = np.memmap(path, dtype='float32', mode='r', shape=shape)
        ori = mm.copy()  # (T, N, 1 + Ct)
        data = ori[..., 0]  # (T, N)
        time_feats = ori[:, 0, 1:]  # (T, Ct)
        return data, time_feats

    def _read_nc(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        # air
        ds = nc.Dataset(path)
        if self.variable_name is None:
            raise ValueError('variable_name must be provided for .nc files')
        data = np.array(ds.variables[self.variable_name])  # (T, N)
        ds.close()

        length_idx = data.shape[0]
        base = pd.Timestamp('2016-01-01 00:00')
        idx = pd.date_range(start=base, periods=length_idx, freq='h')
        time_feats = np.column_stack([idx.month, idx.day, idx.weekday, idx.hour])  # (T, Ct)
        return data, time_feats

    def _split_indices(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def to_len(v):
            if isinstance(v, float):
                return int(round(num_samples * v))
            return int(v)
        n_train = to_len(self.train_size)
        n_val = to_len(self.val_size)
        n_test = to_len(self.test_size)
        if n_train + n_val + n_test != num_samples:
            n_train = min(n_train, num_samples)
            remaining = num_samples - n_train
            n_val = min(n_val, remaining)
            n_test = num_samples - n_train - n_val
        train_idx = np.arange(0, n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        test_idx = np.arange(n_train + n_val, num_samples)
        return train_idx, val_idx, test_idx

    def load_data(self):
        path = self._resolve_path(self.data_dir)
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            data, time_feats = self._read_csv(path)
        elif ext == '.dat':
            data, time_feats = self._read_dat(path)
        elif ext == '.nc':
            data, time_feats = self._read_nc(path)
        else:
            raise ValueError('Unsupported data format')

        self.x_offsets = np.arange(-(self.input_len - 1), 1, 1)
        self.y_offsets = np.arange(self.y_start, (self.output_len + 1), 1)

        length = data.shape[0]
        self.min_t = int(abs(min(self.x_offsets)))
        max_t = int(abs(length - abs(max(self.y_offsets))))

        num_samples = max_t - self.min_t
        train_idx, val_idx, test_idx = self._split_indices(num_samples)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        if self.scale:
            x_train = data[train_idx]  # (train_size, N)
            m = np.mean(x_train, axis=0, keepdims=True)  # (1, N)
            s = np.std(x_train, axis=0, keepdims=True, ddof=1)
            s[s == 0] = 1.0
            self.mean = m
            self.std = s
            data = self.transform(data)

        self.data = data
        self.time_feats = time_feats

        if self.flag == 'train':
            self.indices = self.train_idx
        elif self.flag == 'val':
            self.indices = self.val_idx
        else:
            self.indices = self.test_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Map logical index to time index
        t = self.min_t + self.indices[index]

        x = self.data[t + self.x_offsets]
        y = self.data[t + self.y_offsets]
        x_time = self.time_feats[t + self.x_offsets]
        y_time = self.time_feats[t + self.y_offsets]
        return x, y, x_time, y_time

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, round: bool = False):
        inv = data * self.std + self.mean
        if round and 'PEMS' in self.data_dir:
            inv = np.round(inv)
        return inv

    @classmethod
    def from_existing(cls, base: "ForecastDataset", flag: str) -> "ForecastDataset":
        assert flag in ('train', 'val', 'test')
        assert base.data is not None

        new = cls.__new__(cls)
        new.flag = flag
        new.data_dir = base.data_dir
        new.scale = base.scale
        new.input_len = base.input_len
        new.output_len = base.output_len
        new.y_start = base.y_start
        new.train_size = base.train_size
        new.val_size = base.val_size
        new.test_size = base.test_size
        new.variable_name = base.variable_name

        # Share the data reference
        new.data = base.data
        new.time_feats = base.time_feats
        new.x_offsets = base.x_offsets
        new.y_offsets = base.y_offsets
        new.min_t = base.min_t

        new.train_idx = base.train_idx
        new.val_idx = base.val_idx
        new.test_idx = base.test_idx

        new.mean = base.mean
        new.std = base.std

        if new.flag == 'train':
            new.indices = new.train_idx
        elif new.flag == 'val':
            new.indices = new.val_idx
        else:
            new.indices = new.test_idx

        return new


if __name__ == '__main__':
    print('load water')
    ds = ForecastDataset(flag='train',
                         output_len=72,
                         data_dir='water_mn/data.csv',
                         scale=True,
                         y_start=1,
                         train_size=0.7,
                         val_size=0.1,
                         test_size=0.2)
    ds.load_data()
    print(ds.data.shape, ds.time_feats.shape, len(ds))

    ds_val = ForecastDataset.from_existing(ds, 'val')
    print(ds_val.data.shape, ds_val.time_feats.shape, len(ds_val))

    ds_test = ForecastDataset.from_existing(ds, 'test')
    print(ds_test.data.shape, ds_test.time_feats.shape, len(ds_test))

    print('load traffic')
    ds = ForecastDataset(flag='train',
                         data_dir='PEMS04/data.dat',
                         scale=True,
                         output_len=72,
                         y_start=1,
                         train_size=0.6,
                         val_size=0.2,
                         test_size=0.2)
    ds.load_data()
    print(ds.data.shape, ds.time_feats.shape, len(ds))

    ds_val = ForecastDataset.from_existing(ds, 'val')
    print(ds_val.data.shape, ds_val.time_feats.shape, len(ds_val))

    ds_test = ForecastDataset.from_existing(ds, 'test')
    print(ds_test.data.shape, ds_test.time_feats.shape, len(ds_test))

    print('load air')
    ds = ForecastDataset(flag='train',
                         data_dir='air_bthsa/data.nc',
                         scale=True,
                         output_len=72,
                         y_start=1,
                         train_size=0.5,
                         val_size=0.25,
                         test_size=0.25,
                         variable_name='PM2.5')
    ds.load_data()
    print(ds.data.shape, ds.time_feats.shape, len(ds))

    ds_val = ForecastDataset.from_existing(ds, 'val')
    print(ds_val.data.shape, ds_val.time_feats.shape, len(ds_val))

    ds_test = ForecastDataset.from_existing(ds, 'test')
    print(ds_test.data.shape, ds_test.time_feats.shape, len(ds_test))
