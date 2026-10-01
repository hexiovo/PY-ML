from pathlib import Path
import json
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

if sys.version_info[:3] == (3, 12, 0):
    raise SystemExit('Do not build with Python 3.12.0: code.replace corrupts inlined-comprehension locals. Use Python 3.12.14.')

root = Path(SPECPATH).parent
catalog = json.loads((root / 'src/pyml_workbench/model_catalog.json').read_text(encoding='utf-8'))
models = catalog['active_models'] + catalog['deferred_models']
hidden = sorted({m['implementation'].rsplit('.', 1)[0] for m in models if m.get('implementation')})
hidden += ['pyml_workbench._worker', 'hmmlearn.hmm', 'skorch.classifier', 'skorch.regressor',
           'optuna', 'pymoo.core.mixed', 'pymoo.core.variable', 'pymoo.optimize',
           'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_svg',
           'openpyxl', 'xlrd']
hidden += collect_submodules('sklearn', filter=lambda name: '.tests' not in name)
hidden += collect_submodules('pymoo', filter=lambda name: '.tests' not in name)
hidden += collect_submodules('optuna', filter=lambda name: '.testing' not in name)
datas = collect_data_files('pyml_workbench')
# One source document is used by both the public docs and the offline viewer.
datas += [(str(root / 'docs/overall-guide.md'), 'pyml_workbench/resources')]
for package in ['pyml-workbench', 'scikit-learn', 'hmmlearn', 'skorch', 'torch', 'optuna', 'pymoo', 'moocore']:
    datas += copy_metadata(package)
a = Analysis([str(root / 'packaging/desktop_entry.py'), str(root / 'packaging/worker_entry.py')],
    pathex=[str(root / 'src')], binaries=[], datas=datas, hiddenimports=hidden,
    hookspath=[], hooksconfig={'matplotlib': {'backends': ['QtAgg', 'Agg', 'svg']}},
    runtime_hooks=[str(root / 'packaging/runtime_hook.py')],
    excludes=['tkinter', 'IPython', 'notebook', 'pytest', 'sphinx'],
    noarchive=False)
pyz = PYZ(a.pure)
gui_scripts = [entry for entry in a.scripts if entry[0] != 'worker_entry']
worker_scripts = [entry for entry in a.scripts if entry[0] not in {'desktop_entry', 'pyi_rth_pyside6'}]
gui = EXE(pyz, gui_scripts, [], exclude_binaries=True, name='PYML-Workbench',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False)
worker = EXE(pyz, worker_scripts, [], exclude_binaries=True, name='PYML-Worker',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
# Qt uses the unversioned Windows system ICU API. A same-named Poppler/Conda
# library found on the build host PATH exports versioned ICU symbols instead.
binaries = [entry for entry in a.binaries if Path(entry[0]).name.casefold() != 'icuuc.dll']
coll = COLLECT(gui, worker, binaries, a.datas, strip=False, upx=False, name='PYML-Workbench')
