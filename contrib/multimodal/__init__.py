import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import collections
import src.data
import src.models
import src.utils

MultiModalSSTTrainingItem = collections.namedtuple(
    "MultiModalSSTTrainingItem", ["input", "tgt", "sst"]
)


def load_data_with_courant(obs_var='five_nadirs'):
    print('load_data_with_sst (multimodal)')
    inp = (xr.open_dataset( "/DATASET/turbidity_kd/Obs_patch50_FrMedCoast_log10_2019_2020_2021_.nc").kd490.sel(time=slice('2019-01-01','2021-06-30')))
    gt = (xr.open_dataset("/DATASET/turbidity_kd/GT_FrMedCoast_log10_2019_2020_2021_.nc").kd490.sel(time=slice('2019-01-01','2021-06-30')))
    sst = (xr.open_dataset("/DATASET/turbidity_kd/forcages/2019_to_2021_courant_FrMedCoast_log10.nc").uo.sel(time=slice('2019-01-01','2021-06-30')))
    ds =  (
	xr.Dataset(dict(
            input=inp, tgt=(gt.dims, gt.values), sst=(sst.dims, sst.values)
        ), inp.coords).load()
        .transpose('time', 'lat', 'lon')
    )

    return ds.to_array()
 
def load_data_with_sst(obs_var='five_nadirs'):
    print('load_data_with_sst (multimodal)')
    inp = (xr.open_dataset( "/DATASET/turbidity_kd/Obs_patch50_FrMedCoast_log10_2019_2020_2021_.nc").kd490.sel(time=slice('2019-01-01','2021-06-30')))
    gt = (xr.open_dataset("/DATASET/turbidity_kd/GT_FrMedCoast_log10_2019_2020_2021_.nc").kd490.sel(time=slice('2019-01-01','2021-06-30')))
    sst = (xr.open_dataset("/DATASET/turbidity_kd/forcages/2019_to_2021_sst_FrMedCoast_log10.nc").sst.sel(time=slice('2019-01-01','2021-06-30')))
    ds =  (
	xr.Dataset(dict(
            input=inp, tgt=(gt.dims, gt.values), sst=(sst.dims, sst.values)
        ), inp.coords).load()
        .transpose('time', 'lat', 'lon')
    )

    return ds.to_array()

def load_data_with_wave(obs_var='five_nadirs'):
    print('load_data_with_sst (multimodal)')
    inp = (xr.open_dataset( "../data/Obs_patch50_FrMedCoast_log10_2019_2020_2021_.nc").kd490.sel(time=slice('2019-01-01','2021-06-30')))
    gt = (xr.open_dataset("../data/GT_FrMedCoast_log10_2019_2020_2021_.nc").kd490.sel(time=slice('2019-01-01','2021-06-30')))
    sst = (xr.open_dataset("../data/2019_to_2021_courant_FrMedCoast_log10.nc").uo.sel(time=slice('2019-01-01','2021-06-30')))
    ds =  (
	xr.Dataset(dict(
            input=inp, tgt=(gt.dims, gt.values), sst=(sst.dims, sst.values)
        ), inp.coords).load()
        .transpose('time', 'lat', 'lon')
    )

    return ds.to_array()

def load_data_with_wind(obs_var='five_nadirs'):
    print('load_data_with_sst (multimodal)')
    inp = (xr.open_dataset( "../data/Obs_patch50_FrMedCoast_log10_2019_2020_2021_.nc").kd490.sel(time=slice('2019-01-01','2021-06-30')))
    gt = (xr.open_dataset("../data/GT_FrMedCoast_log10_2019_2020_2021_.nc").kd490.sel(time=slice('2019-01-01','2021-06-30')))
    sst = (xr.open_dataset("../data/2019_to_2021_courant_FrMedCoast_log10.nc").uo.sel(time=slice('2019-01-01','2021-06-30')))
    ds =  (
	xr.Dataset(dict(
            input=inp, tgt=(gt.dims, gt.values), sst=(sst.dims, sst.values)
        ), inp.coords).load()
        .transpose('time', 'lat', 'lon')
    )

    return ds.to_array()

class MultiModalDataModule(src.data.BaseDataModule):
    def post_fn(self):
        print('post_fn (MultiModalModule)')

        normalize_ssh = lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1]
        m_sst, s_sst = self.train_mean_std('sst')
        normalize_sst = lambda item: (item - m_sst) / s_sst
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                MultiModalSSTTrainingItem._make,
                lambda item: item._replace(tgt=normalize_ssh(item.tgt)),
                lambda item: item._replace(input=normalize_ssh(item.input)),
                lambda item: item._replace(sst=normalize_sst(item.sst)),
            ],
        )

class MultiModalObsCost(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        print('init (MultiModalObsCost)')
        super().__init__()
        self.base_cost = src.models.BaseObsCost()

        self.conv_ssh =  torch.nn.Conv2d(dim_in, dim_hidden, (3, 3), padding=1, bias=False)
        self.conv_sst =  torch.nn.Conv2d(dim_in, dim_hidden, (3, 3), padding=1, bias=False)

    def forward(self, state, batch):
        print('forward (MultiModalObsCost)')
        ssh_cost =  self.base_cost(state, batch)
        sst_cost =  torch.nn.functional.mse_loss(
            self.conv_ssh(state),
            self.conv_sst(batch.sst.nan_to_num()),
        )
        return ssh_cost + sst_cost
