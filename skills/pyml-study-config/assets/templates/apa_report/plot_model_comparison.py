"""Generate task-specific, exportable model comparison figures from saved outputs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
def make_figures(run,root,m,summary,ranks,states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator,LogFormatterSciNotation
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    figures=[]
    def example(ax,method,profile,table):
        key=f'model:{method}:{profile}:table{table}'
        file=run/'results'/key.replace(':','_')/'01.json'
        if states.get(key+'/01') is not True or not file.exists():
            ax.text(.5,.5,'固定第1次未通过/尚未提交\n参见阶段与错误清单',transform=ax.transAxes,ha='center');return None
        return json.loads(file.read_text(encoding='utf-8'))
    def save(fig,name,title,note):
        for ext in ['png','pdf']:fig.savefig(root/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig);figures.append({'name':name+'.png','title':title,'note':note})
    for profile,label,accuracy_reference in [('binary','二分类',.5),('multiclass','三分类',1/3)]:
        for table in [1,2]:
            methods=[a['base_key'] for a in m['methods'] if a['table']==table and a['profile']==profile and a['kind']=='model']
            data=summary[(summary.profile==profile)&(summary.table==table)].set_index('method').reindex(methods)
            fig,axs=plt.subplots(1,2,figsize=(8,9),sharey=True)
            for ax,c,title,reference in zip(axs,['accuracy','roc_auc'],['准确率','ROC-AUC'],[accuracy_reference,.5]):
                ax.errorbar(data[c+'_mean'],range(len(methods)),xerr=data[c+'_sd'],fmt='o',markersize=4,capsize=2,color='#347b74');ax.axvline(reference,color='gray',linestyle='--');ax.set_xlim(-.2,1.2);ax.set_xticks(np.linspace(0,1,6));ax.set_xlabel(title+'：均值 ± 种子间SD')
                for i,v in enumerate(data[c+'_mean']):
                    if pd.isna(v):ax.text(.02,i,'不可估计/失败',fontsize=7,va='center')
            axs[0].set_yticks(range(len(methods)),[x.replace('model:','') for x in methods]);axs[0].invert_yaxis();fig.tight_layout()
            save(fig,f'classification_{profile}_table{table}',f'表{table}：{label}逐模型准确率与AUC','真实696列＋人工目标；按ID排列。AUC为正类1/三分类OvR宏平均。虚线为accuracy均匀猜测（.5或1/3）及AUC=.5参考。误差条为种子间SD，可超0–1，非独立样本置信区间。')
    for table in [1,2]:
        data=summary[(summary.profile=='continuous')&(summary.table==table)].sort_values('method')
        fig,axs=plt.subplots(1,3,figsize=(8.4,9.5),sharey=True)
        for ax,c,title in zip(axs,['mae','rmse','r2'],['验证MAE','验证RMSE','验证R²']):
            values=data[c+'_mean'];error=data[c+'_sd'];ax.errorbar(values,range(len(data)),xerr=error,fmt='o',markersize=3,capsize=2,color='#347b74');ax.set_xscale('symlog',linthresh=1);ax.set_xlabel(title+'\n对称对数刻度');ax.axvline(0,color='gray',linewidth=.7)
            lo,hi=ax.get_xlim();ticks=[0.]
            if lo<-1:ticks.append(-10.**np.floor(np.log10(-lo)))
            if hi>1:ticks.append(10.**np.floor(np.log10(hi)))
            ax.set_xticks(sorted(ticks));ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10));ax.xaxis.set_minor_locator(NullLocator())
        axs[0].set_yticks(range(len(data)),data.method.str.replace('model:','',regex=False));axs[0].invert_yaxis();fig.tight_layout()
        save(fig,f'continuous_table{table}',f'表{table}：连续回归模型的验证误差与R²','同一表内使用固定正连续人工目标；按ID排列。误差条为种子间SD，可越过指标取值界限；对称对数刻度保留极端值、零和负R²，|x|≤1为线性区。不可解释的R²留空。')
    fig,axs=plt.subplots(1,2,figsize=(8,3.5))
    for ax,table in zip(axs,[1,2]):
        d=example(ax,'R01','continuous',table)
        if d is None:continue
        rows=[r for r in d['rows'] if r['split']=='validation'];y=[r['y_true'] for r in rows];p=[r['y_pred'] for r in rows];ax.scatter(y,p,color='#347b74');lo=min(y+p);hi=max(y+p);ax.plot([lo,hi],[lo,hi],'--',color='gray');ax.set(xlabel='人工连续目标',ylabel='R01预测值',title=f'表{table}，固定第1次')
    fig.tight_layout();save(fig,'continuous_predictions','连续回归固定模型的预测—目标图','预先固定R01第1次，不按分数选例；对应完整验证记录。小样本和人工目标下不作生理解释。')
    for profile,label in [('sequence','人工顺序窗口回归'),('forecast','独立人工目标序列预测')]:
        fig,axs=plt.subplots(2,2,figsize=(8,5))
        for row,table in enumerate([1,2]):
            data=summary[(summary.profile==profile)&(summary.table==table)].sort_values('method')
            for ax,c in zip(axs[row],['mae','rmse']):
                ax.errorbar(data[c+'_mean'],range(len(data)),xerr=data[c+'_sd'],fmt='o',capsize=2,color='#347b74');ax.set_yticks(range(len(data)),data.method.str.replace('model:','',regex=False));ax.invert_yaxis();ax.set_xlabel(c.upper());ax.set_title(f'表/夹具{table}')
        fig.tight_layout();save(fig,profile+'_comparison',label+'的误差对比','仅比较同一来源与评价单位的方法；窗口和人工顺序不是真实实验时间，预测适配器仅消费人工目标历史。')
    fig,axs=plt.subplots(1,2,figsize=(8,3.5))
    for ax,table in zip(axs,[1,2]):
        d=example(ax,'R21','count',table)
        if d is None:continue
        rows=[r for r in d['rows'] if r['split']=='validation'];y=[r['y_true'] for r in rows];p=[r['y_pred'] for r in rows];ax.scatter(y,p,color='#347b74');lo=min(y+p);hi=max(y+p);ax.plot([lo,hi],[lo,hi],'--',color='gray');ax.set(xlabel='人工计数真实值',ylabel='Poisson预测值',title=f'表{table}，固定第1次')
    fig.tight_layout();save(fig,'count_predictions','人工正计数目标的固定示例预测','计数为1+Poisson(3)的工程目标，非实际事件数；偏差指标及完整结果另附。')
    fig,axs=plt.subplots(2,1,figsize=(8,6))
    for ax,table in zip(axs,[1,2]):
        d=example(ax,'T04','forecast',table)
        if d is None:continue
        rows=[r for r in d['rows'] if r['split']=='validation'];x=[r['row_position'] for r in rows];ax.plot(x,[r['y_true'] for r in rows],label='人工目标',color='#34495e');ax.plot(x,[r['y_pred'] for r in rows],label='固定起点预测',color='#ae7f3e');ax.set(xlabel='人工序号',ylabel='目标数值',title=f'独立夹具{table}：T04，第1次');ax.legend()
    fig.tight_layout();save(fig,'forecast_examples','ARIMA固定示例的人工目标序列预测','不使用验证目标更新历史；顺序与目标为独立夹具，不是真实EEG时间。')
    fig,axs=plt.subplots(2,1,figsize=(8,9))
    for ax,profile in zip(axs,['binary','multiclass']):
        data=ranks[(ranks.method=='model:C01')&(ranks.table==1)&(ranks.profile==profile)].sort_values('rank').head(20);ax.barh(data.feature,data['mean'],xerr=data.sd,capsize=2,color='#347b74');ax.invert_yaxis();ax.axvline(0,color='gray',linewidth=.7);ax.set_title(profile+'：整段C01前20列');ax.set_xlabel('验证balanced accuracy下降量')
    fig.tight_layout();save(fig,'feature_importance_examples','二分类与三分类固定模型的真实输入特征敏感性','完整696列排序及原始特征映射另附；目标为人工，负值保留，不作脑区或因果解释。')
    fig,axs=plt.subplots(1,2,figsize=(8,3.5))
    for ax,table in zip(axs,[1,2]):
        d=example(ax,'D01','unsupervised',table)
        if d is None:continue
        v=np.asarray(d['output']);ax.scatter(v[:,0],v[:,1],s=22,color='#347b74');ax.set(title=f'真实表{table}：696→2维',xlabel='主成分1',ylabel='主成分2')
    fig.tight_layout();save(fig,'real_pca','真实全部特征的独立PCA描述图','未使用人工目标；两表分别全量拟合，非独立验证，不把视觉分离解释为实验条件效应。')
    profiles=list(dict.fromkeys(a['profile'] for a in m['methods']));fig,ax=plt.subplots(figsize=(7,4.5));passed=[];failed=[];missing=[]
    for profile in profiles:
        keys=[f"{a['key']}/{i:02d}" for a in m['methods'] if a['profile']==profile for i in range(1,11)];passed.append(sum(states.get(k) is True for k in keys));failed.append(sum(states.get(k) is False for k in keys));missing.append(sum(k not in states for k in keys))
    ax.barh(profiles,passed,label='PASS',color='#347b74');ax.barh(profiles,failed,left=passed,label='未通过',color='#bd624e');ax.barh(profiles,missing,left=np.asarray(passed)+failed,label='未提交',color='gray');ax.invert_yaxis();ax.set_xlabel('外层循环槽位数');ax.legend();fig.tight_layout();save(fig,'coverage','全任务实际执行覆盖','方法×目标类型×表×循环计数，保留失败和未提交，不以图中覆盖量代替效度。')
    return figures

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='从匿名示例图表数据重绘；不拟合模型，不代表新研究结果。')
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'figure_data')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    methods=json.loads((args.data/'methods.json').read_text(encoding='utf-8'));states=json.loads((args.data/'stages.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(args.data/'metric-summary.csv');ranks=pd.read_csv(args.data/'example-importance.csv')
    figures=make_figures(args.data,args.output,methods,summary,ranks,states)
    (args.output/'figures.json').write_text(json.dumps({'scope':'anonymized historical interface example; no model fitting','figures':figures},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Recreated',len(figures),'example figures; no model fitting')
