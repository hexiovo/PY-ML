"""Bounded synthetic checks for grouped stacking and ordinary-call compatibility."""
from pathlib import Path
import argparse,json,sys,warnings

def verify(project):
    import numpy as np,pandas as pd
    sys.path.insert(0,str(Path(project).resolve()/'src'))
    from pyml_workbench import build_estimator
    rng=np.random.default_rng(12873);X=pd.DataFrame(rng.normal(size=(60,6)),columns=[f'x{i}' for i in range(6)]);X.iloc[::9,0]=np.nan
    groups=np.repeat(np.arange(6),10);targets=[('C30',np.tile([0,1],30),['C01','C14']),('C30',np.tile([0,1,2],20),['C01','C14']),('R29',2+np.nan_to_num(X.x0.to_numpy())+rng.normal(0,.2,60),['R01','R12'])]
    checked=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        for model,y,members in targets:
            params={'members':members,'cv':3,'n_jobs':1};plain=build_estimator(model,params,seed=41).fit(X,y)
            assert plain.inner_cv_audit_==[] and np.isfinite(plain.predict(X[:4]).astype(float)).all()
            grouped=build_estimator(model,params,seed=41).fit(X,y,groups=groups);seen=[]
            for fold in grouped.inner_cv_audit_:
                assert not set(fold['train_groups'])&set(fold['validation_groups'])
                assert not set(fold['train_indices'])&set(fold['validation_indices']);seen.extend(fold['validation_indices'])
            assert sorted(seen)==list(range(len(X)))
            assert np.isfinite(grouped.predict(X[:4]).astype(float)).all()
            for invalid in [groups[:-1],np.zeros(len(X))]:
                try:build_estimator(model,params,seed=41).fit(X,y,groups=invalid)
                except ValueError:pass
                else:raise AssertionError('Invalid group metadata accepted')
            checked.append({'model':model,'target_classes':len(np.unique(y)) if model=='C30' else None,'ordinary_compatible':True,'inner_groups_disjoint':True,'each_row_in_oof_once':True,'invalid_groups_rejected':True})
    return {'status':'PASS','data':'independent synthetic fixture, 60 rows x 6 features','checks':checked,'scientific_final_test_calls':0}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();print(json.dumps(verify(args.project),ensure_ascii=False,indent=2))
