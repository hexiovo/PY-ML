"""Configure a copy outside the skill; never save personal settings in templates."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
SKILL_ROOT=next((p for p in HERE.parents if (p/'SKILL.md').is_file()),HERE)

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def resolve_python(value):
    candidate=shutil.which(value) or value
    path=Path(candidate).expanduser().resolve()
    if not path.is_file():raise ValueError('请选择可用的Python解释器。')
    return path

def validate(config,mode):
    cfg=dict(config);cfg['python']=str(resolve_python(cfg.get('python') or sys.executable))
    for key in ['work_root','output_root']:
        if not cfg.get(key):raise ValueError('工作目录和输出目录不能为空。')
        path=Path(cfg[key]).expanduser().resolve()
        if path==SKILL_ROOT or SKILL_ROOT in path.parents or any((p/'SKILL.md').is_file() for p in [path,*path.parents]):raise ValueError('工作和输出目录须位于SKILL之外，防止个人配置进入Git。')
        cfg[key]=str(path)
    if mode=='convert':
        for key in ['input','plugin','memory_helper_root']:
            if not cfg.get(key) or not Path(cfg[key]).is_dir():raise ValueError(f'缺少有效目录：{key}')
            cfg[key]=str(Path(cfg[key]).resolve())
        if not cfg.get('matlab') or not Path(cfg['matlab']).is_file():raise ValueError('请选择本机MATLAB程序。')
        cfg['matlab']=str(Path(cfg['matlab']).resolve())
        plugin=load(Path(cfg['plugin'])/'pyml-plugin.json')
        if plugin.get('id')!='data-conversion' or plugin.get('source_project','HyperEEG')!='HyperEEG':raise ValueError('所选插件不是HyperEEG转换插件。')
        if not (Path(cfg['plugin'])/'hypereeg_converter/core.py').is_file():raise ValueError('此模板需要HyperEEG插件Python源码和依赖，不能直接使用只有EXE的目录。')
        source=Path(cfg['input']);manifest=load(source/'结果清单.json')
        if manifest.get('storage_mode')!='linked_native_results':raise ValueError('此转换模板仅支持已核实的linked_native_results引用包。')
        if not (source/'特征结果索引.mat').is_file() or not Path(manifest.get('retained_run','')).is_dir():raise ValueError('引用索引或原生保留目录缺失。')
        helper=Path(cfg['memory_helper_root'])/'+HyperEEG/+MultiCH/+core/FeatureArtifactMemoryEstimate.m'
        if not helper.is_file():raise ValueError('HyperEEG工程缺少内存准入辅助实现；不能绕过预算检查。')
        native=Path(manifest['retained_run']).resolve()
        for key in ['work_root','output_root']:
            path=Path(cfg[key])
            if any(path==p or p in path.parents for p in [source,native]):raise ValueError('工作和输出目录不能写入输入结果包或原生保留目录。')
        for key,default in [('max_artifact_load_gib',1.5),('reserve_free_gib',1.),('max_output_cells',1000000)]:
            cfg[key]=cfg.get(key,default)
            if isinstance(cfg[key],bool) or not isinstance(cfg[key],(int,float)) or cfg[key]<=0:raise ValueError(f'预算参数必须为正数：{key}')
    return cfg

def prepare(config,mode):
    cfg=validate(config,mode);work=Path(cfg['work_root']);work.mkdir(parents=True,exist_ok=True)
    app=work/'apps'/('hypereeg_converter' if mode=='convert' else 'apa_report')
    shutil.copytree(HERE/app.name,app,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if mode=='convert':
        settings={k:cfg[k] for k in ['plugin','input','matlab','output_root','memory_helper_root','max_artifact_load_gib','reserve_free_gib','max_output_cells']}
        (app/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
        options=load(app/'hypereeg-options.json');options['matlab']=cfg['matlab'];(app/'hypereeg-options.json').write_text(json.dumps(options,ensure_ascii=False,indent=2),encoding='utf-8')
    (work/f'{mode}.local.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    return cfg,app

def command(cfg,app,mode):
    if mode=='convert':return [cfg['python'],'-X','utf8',str(app/'gui.py')]
    return [cfg['python'],'-X','utf8',str(app/'build_apa_template.py'),'--output',cfg['output_root']]

def show(mode,initial=None,smoke=False):
    import tkinter as tk
    from tkinter import ttk,filedialog,messagebox
    root=tk.Tk();root.title('PYML便携模板 · '+('HyperEEG转换' if mode=='convert' else 'APA报告模板'));root.geometry('820x620' if mode=='convert' else '820x370')
    frame=ttk.Frame(root,padding=24);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='选择本次目录；模板不保存固定机器路径',font=('Microsoft YaHei UI',15)).grid(row=0,column=0,columnspan=3,sticky='w',pady=(0,18))
    fields=[('python','Python解释器','file'),('work_root','本次工作目录','dir'),('output_root','输出目录','dir')]
    if mode=='convert':fields += [('plugin','HyperEEG插件源码目录','dir'),('input','HyperEEG引用式结果包','dir'),('matlab','MATLAB程序','file'),('memory_helper_root','HyperEEG工程目录','dir')]
    values={};initial=initial or {}
    def browse(key,kind):
        value=filedialog.askopenfilename(parent=root) if kind=='file' else filedialog.askdirectory(parent=root)
        if value:values[key].set(value)
    for row,(key,label,kind) in enumerate(fields,1):
        ttk.Label(frame,text=label).grid(row=row,column=0,sticky='w',padx=(0,12),pady=7)
        value=tk.StringVar(value=initial.get(key,sys.executable if key=='python' else ''));values[key]=value
        ttk.Entry(frame,textvariable=value,width=66).grid(row=row,column=1,sticky='ew',pady=7)
        ttk.Button(frame,text='选择',command=lambda k=key,t=kind:browse(k,t)).grid(row=row,column=2,padx=(10,0),pady=7)
    row=len(fields)+1;stage=tk.StringVar(value='转换仅搬运已支持的原生特征，不推断标签。' if mode=='convert' else '生成可编辑APA模板；示例报告和图表数据随模板提供。')
    elapsed=tk.StringVar(value='已用 00:00:00 · 剩余时间待估计')
    ttk.Label(frame,textvariable=stage,wraplength=740).grid(row=row,column=0,columnspan=3,sticky='w',pady=(18,8))
    ttk.Label(frame,textvariable=elapsed).grid(row=row+1,column=0,columnspan=3,sticky='w')
    process=None;started=0.;log=None;active_cfg=None
    def poll():
        nonlocal process,log
        n=int(time.monotonic()-started);elapsed.set(f'已用 {n//3600:02d}:{n//60%60:02d}:{n%60:02d} · 剩余时间待估计')
        if process.poll() is None:root.after(300,poll);return
        log.close();start.configure(state='normal')
        if mode=='convert':root.deiconify()
        if process.returncode:stage.set('执行失败；记录已保存在工作目录。');messagebox.showerror('执行失败','请查看工作目录的launcher.log；未标记为完成。',parent=root)
        else:stage.set('转换窗口已正常结束。' if mode=='convert' else 'APA报告模板已生成，可打开输出目录。')
        process=None
    def run():
        nonlocal process,started,log,active_cfg
        try:
            cfg,app=prepare(dict(initial,**{k:v.get().strip() for k,v in values.items()}),mode);active_cfg=cfg
            log=(Path(cfg['work_root'])/'launcher.log').open('ab');env=os.environ.copy();env['PYTHONUTF8']='1'
            process=subprocess.Popen(command(cfg,app,mode),stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            started=time.monotonic();start.configure(state='disabled');stage.set('已启动；阶段详情在转换窗口显示。' if mode=='convert' else '正在生成APA模板；剩余时间待估计。')
            if mode=='convert':root.withdraw()
            root.after(300,poll)
        except Exception as exc:messagebox.showerror('配置检查失败',str(exc),parent=root)
    start=ttk.Button(frame,text='开始转换' if mode=='convert' else '生成报告模板',command=run);start.grid(row=row+2,column=1,sticky='e',pady=18)
    def close():
        if process is not None and process.poll() is None:messagebox.showinfo('仍在运行','请先在转换窗口停止，或等待报告生成结束。',parent=root)
        else:root.destroy()
    root.protocol('WM_DELETE_WINDOW',close);frame.columnconfigure(1,weight=1)
    if smoke:root.after(700,root.destroy)
    root.mainloop()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['convert','report']);parser.add_argument('--config',type=Path);parser.add_argument('--validate-only',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    cfg=load(args.config) if args.config else None
    if args.validate_only:
        if cfg is None:raise ValueError('--validate-only需要--config。')
        validate(cfg,args.mode);print('Configuration valid; conversion/model fitting NOT RUN');return
    show(args.mode,cfg,args.smoke)

if __name__=='__main__':main()
