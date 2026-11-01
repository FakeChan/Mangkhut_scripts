"""Seven mean-TSK curves: three experiments x two filters, plus NR.

Default: 0--6 h every 30 min, ocean points within 150 km. Two region modes:
  nr_each_time: circle follows the NR minimum-PSFC center at each output;
  nr_initial:   circle fixed at the initial NR center (cooling comparison).

All seven curves use the same member-grid points and ocean mask. Each
experiment/filter curve is the equal-weight mean of member spatial means.
NR is interpolated once per time to that grid, not averaged across filters.

For a moving region, distinguish:
  regional_mean_change = mean_R(t)[TSK(t)] - mean_R(0)[TSK(0)]
  physical_change      = mean_R(t)[TSK(t) - TSK(0)]
  sampling_change      = mean_R(t)[TSK(0)] - mean_R(0)[TSK(0)]
The first equals the last two added together. "physical" here means a
change at fixed geographic grid points, not a Lagrangian water parcel.

Edit CONFIG, then run this file without command-line arguments. Imports do
not read server data or create the output directory.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

# =====================================================================
# User configuration
# =====================================================================
CONFIG = {
    "member_base": Path('/scratch/lililei1/kcfu/tc_mangkhut/cycle_test'),
    "nr_base": Path('/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain'),
    "output_dir": Path(__file__).resolve().parent/'figs'/'mean_tsk_timeseries',
    "experiments": ('6mem_oceanAssim0Run0','6mem_oceanAssim0Run1','6mem_oceanAssim1Run1'),
    "filters": ('EAKF','QCF_RHF'),
    "members": ('006','015','029','037','043','044'),
    "member_domain": 'd02',
    "nr_domain": 'd02',
    "start_time": '2018-09-10_00:00:00',
    "end_time": '2018-09-10_06:00:00',
    "step_minutes": 30,
    "radius_km": 150.0,
    "center_pressure_var": 'PSFC',
    "region_modes": ('nr_each_time','nr_initial'),
    "bbox_pad": 3,
    "nr_margin_deg": 0.06,
    "grid_tolerance_deg": 1e-6,
    "fill_threshold": 1e30,
    "temperature_unit": 'K',  # 'K' or 'degC'; temperature changes always K
    "figure_formats": ('png','pdf'),
    "figure_dpi": 300,
    "figure_size": (11.0,5.1),
    "font_size": 9,
    "experiment_labels": {
        '6mem_oceanAssim0Run0': 'Assim0 / Run0',
        '6mem_oceanAssim0Run1': 'Assim0 / Run1',
        '6mem_oceanAssim1Run1': 'Assim1 / Run1',
    },
    "experiment_colors": {
        '6mem_oceanAssim0Run0': '#0072B2',
        '6mem_oceanAssim0Run1': '#D55E00',
        '6mem_oceanAssim1Run1': '#009E73',
    },
    "filter_styles": {'EAKF': ('-','o'), 'QCF_RHF': ('--','s')},
}

TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'
CHANGE_COLUMNS = ('mean_tsk_k','regional_mean_change_k','physical_change_k','sampling_change_k')


def build_times(config: dict) -> list[datetime]:
    start = datetime.strptime(config['start_time'],TIME_FORMAT)
    end = datetime.strptime(config['end_time'],TIME_FORMAT)
    step = config['step_minutes']
    if step <= 0 or end < start or (end-start).total_seconds() % (step*60):
        raise ValueError('Time range must be nonnegative and divisible by step_minutes > 0')
    return [start+timedelta(minutes=i*step)
            for i in range(int((end-start).total_seconds()/(step*60))+1)]


def _array(value) -> np.ndarray:
    return np.ma.asarray(value,dtype=float).filled(np.nan)


def _valid(value, config) -> np.ndarray:
    return np.isfinite(value) & (np.abs(value)<config['fill_threshold'])


def _distance(lat,lon,center) -> np.ndarray:
    a,b = np.deg2rad(lat),np.deg2rad(center[0])
    dl = np.deg2rad(lon-center[1])
    h = np.sin((a-b)/2)**2+np.cos(a)*np.cos(b)*np.sin(dl/2)**2
    return 12742.0*np.arcsin(np.sqrt(np.clip(h,0,1)))


def _bbox(mask,pad):
    j,i = np.where(mask)
    if not len(j):
        raise ValueError('No model grid points within the requested NR-centered region')
    return (slice(max(0,int(j.min())-pad),min(mask.shape[0],int(j.max())+pad+1)),
            slice(max(0,int(i.min())-pad),min(mask.shape[1],int(i.max())+pad+1)))


def _file(root: Path,domain: str,time: datetime) -> Path:
    name = f'wrfout_{domain}_{time.strftime(TIME_FORMAT)}'
    direct = Path(root)/name
    if direct.is_file():
        return direct
    matches = list(Path(root).rglob(name)) if Path(root).is_dir() else []
    if len(matches)!=1:
        raise FileNotFoundError(f'Expected one {name} under {root}, found {len(matches)}')
    return matches[0]


def _check_time(d,path,time):
    if 'Times' not in d.variables:
        raise ValueError(f'Missing WRF Times variable: {path}')
    first = np.asarray(d['Times'][0])
    text = (first.tobytes().decode().rstrip('\x00 ') if first.dtype.kind=='S'
            else ''.join(first.ravel().astype(str)).strip())
    if text!=time.strftime(TIME_FORMAT):
        raise ValueError(f'WRF time mismatch: {path}: {text}')


def _center(path,time,config):
    with Dataset(path,'r') as d:
        _check_time(d,path,time)
        ps = _array(d[config['center_pressure_var']][0])
        ps = np.where(_valid(ps,config),ps,np.nan)
        j,i = np.unravel_index(np.nanargmin(ps),ps.shape)
        return (float(d['XLAT'][0,j,i]),float(d['XLONG'][0,j,i]))


class _NrReader:
    """Reuse one local Delaunay geometry; reject moving/changing NR grids."""
    def __init__(self,path,lat,lon,config):
        self.lat,self.lon,self.config = lat,lon,config
        with Dataset(path,'r') as d:
            source_lat,source_lon = _array(d['XLAT'][0]),_array(d['XLONG'][0])
        margin = config['nr_margin_deg']
        select = ((source_lat>=lat.min()-margin)&(source_lat<=lat.max()+margin)
                  &(source_lon>=lon.min()-margin)&(source_lon<=lon.max()+margin))
        self.ys,self.xs = _bbox(select,config['bbox_pad'])
        self.source_lat = source_lat[self.ys,self.xs]
        self.source_lon = source_lon[self.ys,self.xs]
        self.points = np.column_stack([self.source_lon.ravel(),self.source_lat.ravel()])
        if not np.isfinite(self.points).all():
            raise ValueError('NR coordinates contain missing values')
        self.tri = Delaunay(self.points)
        self.queries = np.column_stack([lon.ravel(),lat.ravel()])

    def read(self,path,time):
        with Dataset(path,'r') as d:
            _check_time(d,path,time)
            for name,expected in [('XLAT',self.source_lat),('XLONG',self.source_lon)]:
                actual = _array(d[name][0,self.ys,self.xs])
                if actual.shape!=expected.shape or not np.allclose(actual,expected,atol=self.config['grid_tolerance_deg'],rtol=0):
                    raise ValueError(f'NR grid changed: {path}')
            ts = _array(d['TSK'][0,self.ys,self.xs])
            land = _array(d['LANDMASK'][0,self.ys,self.xs])
        ts[~_valid(ts,self.config)] = np.nan
        land[~_valid(land,self.config)] = np.nan
        temperature = LinearNDInterpolator(self.tri,ts.ravel(),fill_value=np.nan)(self.queries)
        # Land is categorical. Outside-hull temperatures stay NaN, never extrapolated.
        ocean_type = NearestNDInterpolator(self.points,land.ravel())(self.queries)
        return temperature.reshape(self.lat.shape),ocean_type.reshape(self.lat.shape)


def compute_timeseries(config: dict) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Strict complete ensemble; common initial support frozen across all times.

    Reads only latitude/longitude, TSK, LANDMASK and NR center pressure.
    Missing members or lost common support raise, rather than changing the sample.
    """
    times = build_times(config)
    if not config['members'] or not config['experiments'] or not config['filters']:
        raise ValueError('Experiments, filters and members must be nonempty')
    if config['radius_km']<=0 or not config['region_modes']:
        raise ValueError('Positive radius and at least one region mode are required')
    if set(config['region_modes'])-{'nr_each_time','nr_initial'}:
        raise ValueError('Unknown region mode')
    specs = [(e,f,m) for e in config['experiments'] for f in config['filters'] for m in config['members']]
    nr_files = {t:_file(Path(config['nr_base']),config['nr_domain'],t) for t in times}
    member_files = {(spec,t):_file(Path(config['member_base'])/spec[0]/spec[1]/spec[2],config['member_domain'],t)
                    for spec in specs for t in times}
    centers = {t:_center(nr_files[t],t,config) for t in times}
    with Dataset(member_files[specs[0],times[0]],'r') as d:
        full_lat,full_lon = _array(d['XLAT'][0]),_array(d['XLONG'][0])
    union = np.zeros_like(full_lat,dtype=bool)
    clipped = {}
    for t in times:
        disk = _distance(full_lat,full_lon,centers[t])<=config['radius_km']
        clipped[t] = bool(disk[0,:].any() or disk[-1,:].any() or disk[:,0].any() or disk[:,-1].any())
        union |= disk
    ys,xs = _bbox(union,config['bbox_pad'])
    lat,lon = full_lat[ys,xs],full_lon[ys,xs]
    union = union[ys,xs]
    nr_reader = _NrReader(nr_files[times[0]],lat,lon,config)

    def load(t):
        surfaces = []
        support = union.copy()
        for spec in specs:
            path = member_files[spec,t]
            with Dataset(path,'r') as d:
                _check_time(d,path,t)
                for name,expected in [('XLAT',lat),('XLONG',lon)]:
                    actual = _array(d[name][0,ys,xs])
                    if actual.shape!=expected.shape or not np.allclose(actual,expected,atol=config['grid_tolerance_deg'],rtol=0):
                        raise ValueError(f'Member grid differs from reference grid: {path}')
                ts = _array(d['TSK'][0,ys,xs])
                land = _array(d['LANDMASK'][0,ys,xs])
            support &= _valid(ts,config)&_valid(land,config)&(land<.5)
            surfaces.append(ts)
        nr_ts,nr_land = nr_reader.read(nr_files[t],t)
        support &= _valid(nr_ts,config)&_valid(nr_land,config)&(nr_land<.5)
        surfaces.append(nr_ts)
        return surfaces,support

    baseline,support0 = load(times[0])
    disk0 = _distance(lat,lon,centers[times[0]])<=config['radius_km']
    initial_mask = support0&disk0
    if not initial_mask.any():
        raise ValueError('No common ocean support at initial time')
    initial_means = [float(a[initial_mask].mean()) for a in baseline]
    keys = specs+[('NR','','NR')]
    member_rows,center_rows = [],[]
    for t in times:
        print(f'[mean TSK] {t.strftime(TIME_FORMAT)}',flush=True)
        surfaces,support = (baseline,support0) if t==times[0] else load(t)
        lost = support0&~support
        if lost.any():
            raise ValueError(f'Common spatial support lost at {t.strftime(TIME_FORMAT)}: {lost.sum()} points; '
                             'inspect missing/fill values or changing land masks')
        hour = (t-times[0]).total_seconds()/3600
        center_rows.append(dict(valid_time=t.strftime(TIME_FORMAT),lead_hour=hour,
                                center_lat=centers[t][0],center_lon=centers[t][1],
                                moving_circle_clipped=clipped[t],initial_circle_clipped=clipped[times[0]]))
        for mode in config['region_modes']:
            center = centers[t] if mode=='nr_each_time' else centers[times[0]]
            disk = _distance(lat,lon,center)<=config['radius_km']
            mask = support0&disk
            if not mask.any():
                raise ValueError(f'Empty ocean support: {mode} at {t}')
            for (exp,filt,member),current,initial,mean0 in zip(keys,surfaces,baseline,initial_means):
                mean = float(current[mask].mean())
                member_rows.append(dict(
                    region_mode=mode,valid_time=t.strftime(TIME_FORMAT),lead_hour=hour,
                    experiment=exp,filter=filt,member=member,
                    curve_id='NR' if exp=='NR' else f'{exp}/{filt}',
                    mean_tsk_k=mean,regional_mean_change_k=mean-mean0,
                    physical_change_k=float((current[mask]-initial[mask]).mean()),
                    sampling_change_k=float(initial[mask].mean())-mean0,
                    n_points=int(mask.sum()),available_disk_points=int(disk.sum()),
                    center_lat=center[0],center_lon=center[1],
                    circle_clipped=clipped[t] if mode=='nr_each_time' else clipped[times[0]],
                ))
    members = pd.DataFrame(member_rows)
    group = ['region_mode','valid_time','lead_hour','experiment','filter','curve_id']
    curves = members.groupby(group,sort=False,dropna=False)[list(CHANGE_COLUMNS)].mean().reset_index()
    extra = members.groupby(group,sort=False,dropna=False).agg(
        n_members=('member','size'),n_points=('n_points','first'),
        available_disk_points=('available_disk_points','first'),
        center_lat=('center_lat','first'),center_lon=('center_lon','first'),circle_clipped=('circle_clipped','first'),
    ).reset_index()
    curves = curves.merge(extra,on=group,validate='one_to_one')
    return members,curves,pd.DataFrame(center_rows)


def plot_timeseries(curves: pd.DataFrame,config: dict) -> list[Path]:
    """Two panels per region mode; exactly six experiment/filter curves + NR."""
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':config['font_size'],
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False})
    output = Path(config['output_dir']); output.mkdir(parents=True,exist_ok=True)
    unit = config['temperature_unit']
    if unit not in ('K','degC'):
        raise ValueError("temperature_unit must be 'K' or 'degC'")
    offset = 273.15 if unit=='degC' else 0.0
    series = [(f'{e}/{f}',f"{config['experiment_labels'][e]} | {f}",
               config['experiment_colors'][e],*config['filter_styles'][f])
              for e in config['experiments'] for f in config['filters']]
    series.append(('NR','NR','#202020','-','D'))
    saved = []
    for mode in config['region_modes']:
        sub = curves[curves.region_mode==mode]
        if set(sub.curve_id)!=set(x[0] for x in series):
            raise ValueError(f'Missing/extra curves for {mode}')
        fig,axes = plt.subplots(1,2,figsize=config['figure_size'])
        for cid,label,color,style,marker in series:
            data = sub[sub.curve_id==cid].sort_values('lead_hour')
            if len(data)!=len(build_times(config)) or data.lead_hour.duplicated().any():
                raise ValueError(f'Incomplete time series: {cid} / {mode}')
            for ax,column in zip(axes,('mean_tsk_k','physical_change_k')):
                values = data[column]-(offset if column=='mean_tsk_k' else 0)
                ax.plot(data.lead_hour,values,label=label,color=color,linestyle=style,
                        marker=marker,markersize=3.5,markerfacecolor='white' if style=='--' else color,
                        linewidth=2.0 if cid=='NR' else 1.35,zorder=5 if cid=='NR' else 3)
        for ax in axes:
            ax.set_xlabel('Forecast lead time (h)')
            ax.set_xlim(0,float(curves.lead_hour.max()))
            ax.grid(alpha=.3,linestyle=':',linewidth=.6)
            ax.ticklabel_format(axis='y',useOffset=False)
        axes[0].set_ylabel(f'Mean TSK ({unit})')
        axes[0].set_title('(a) Regional mean temperature')
        axes[1].set_ylabel('Mean [TSK(t) − TSK(0)] (K)')
        axes[1].set_title('(b) Change at the same geographic points')
        axes[1].axhline(0,color='.5',linewidth=.7,zorder=1)
        mode_label = ('Circle follows NR pressure center' if mode=='nr_each_time'
                      else 'Circle fixed at initial NR pressure center')
        fig.suptitle(f'Mean sea-surface skin temperature: r ≤ {config["radius_km"]:g} km\n{mode_label}',fontsize=11)
        handles,labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8)
        fig.subplots_adjust(left=.085,right=.98,top=.80,bottom=.27,wspace=.28)
        for fmt in config['figure_formats']:
            path = output/f'mean_tsk_{mode}.{fmt}'
            fig.savefig(path,dpi=config['figure_dpi'],facecolor='white')
            saved.append(path)
        plt.close(fig)
    return saved


def main() -> None:
    members,curves,centers = compute_timeseries(CONFIG)
    output = Path(CONFIG['output_dir']); output.mkdir(parents=True,exist_ok=True)
    members.to_csv(output/'mean_tsk_by_member.csv',index=False)
    curves.to_csv(output/'mean_tsk_seven_curves.csv',index=False)
    centers.to_csv(output/'nr_centers.csv',index=False)
    (output/'config_snapshot.json').write_text(json.dumps(CONFIG,default=str,ensure_ascii=False,indent=2))
    (output/'columns_readme.txt').write_text(
        '六条预报曲线=逐成员空间平均后成员等权平均；NR 为一条独立曲线。\n'
        '所有曲线共用参考成员网格，NR 经纬度线性插值，凸包外不外推。\n'
        '使用所有成员/试验/filter/NR 的初始共同有效海洋格点，后续有效覆盖减少则报错。\n'
        'nr_each_time：每个时刻使用 NR 最低 PSFC 中心的圆；nr_initial：固定起报 NR 中心。\n'
        'mean_tsk_k：当前圆域平均 TSK；regional_mean_change_k：当前区域平均减初始区域平均。\n'
        'physical_change_k：在当前圆域同一地理格点上先算 TSK(t)-TSK(0)，再取平均。\n'
        'sampling_change_k：初始温度场在当前圆域与起报圆域的平均之差。\n'
        'regional_mean_change_k = physical_change_k + sampling_change_k。\n'
        '固定圆域下 sampling_change_k=0；physical_change_k 是欧拉变化，不是水团随体变化。\n'
        '0 h TSK 可能尚未反映海洋分析增量；首次耦合赋值也包含在相对 0 h 的变化中。\n'
        '讨论起报调整后的冷却速率时，可另外比较 0.5–2 h 各时次，而非将首次变化全归为混合冷却。\n'
        '各圆域与可用模式域、共同海洋掩膜取交集；circle_clipped=True 表示圆触及模式域边界。\n'
        '此脚本的共同掩膜与 RMSE 脚本逐变量掩膜有区别，不能直接将平均偏差代入原 RMSE。\n'
        '完全相同的曲线允许重合，不人为偏移。图中的负温度变化表示冷却。\n',encoding='utf-8')
    for path in plot_timeseries(curves,CONFIG):
        print(f'Saved {path}',flush=True)
    print(f'Saved seven curves and per-member source data under {output}',flush=True)


if __name__=='__main__':
    main()
