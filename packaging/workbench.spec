from pathlib import Path
import json
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

if sys.version_info[:3] == (3, 12, 0):
    raise SystemExit('Do not build with Python 3.12.0: code.replace corrupts inlined-comprehension locals. Use Python 3.12.14.')

root = Path(SPECPATH).parent
profile_config = json.loads((root / 'packaging/profiles.json').read_text(encoding='utf-8'))
profile_name = os.environ.get('PYML_PACKAGE_PROFILE', profile_config['default_profile']).casefold()
if profile_name not in profile_config['profiles']:
    raise SystemExit(f'Unknown PYML_PACKAGE_PROFILE: {profile_name}')
profile = profile_config['profiles'][profile_name]
catalog = json.loads((root / 'src/pyml_workbench/model_catalog.json').read_text(encoding='utf-8'))
models = [model for section in profile['model_sections'] for model in catalog[section]]
model_modules = set()
profile_extras = set(profile['extras'])
for model in models:
    implementation = model.get('implementation')
    if not implementation:
        continue
    module = implementation.rsplit('.', 1)[0]
    extra = model.get('optional_extra')
    if module.startswith('pyml_workbench.') or not extra or extra in profile_extras:
        model_modules.add(module)
hidden = sorted(model_modules)
hidden += ['pyml_workbench._worker', 'matplotlib.backends.backend_qtagg',
           'matplotlib.backends.backend_svg', 'openpyxl', 'xlrd', 'pyi_splash']
for package in profile['collect_submodules']:
    if package == 'optuna':
        hidden += collect_submodules(package, filter=lambda name: '.testing' not in name)
    else:
        hidden += collect_submodules(package, filter=lambda name: '.tests' not in name)
datas = collect_data_files('pyml_workbench')
# One source document is used by both the public docs and the offline viewer.
datas += [(str(root / 'docs/overall-guide.md'), 'pyml_workbench/resources')]
datas += [(str(root / 'docs/assets/guide-ui'), 'pyml_workbench/resources/assets/guide-ui')]
for package in ['pyml-workbench', *profile['runtime_distributions']]:
    datas += copy_metadata(package)
a = Analysis([str(root / 'packaging/desktop_entry.py'), str(root / 'packaging/worker_entry.py')],
    pathex=[str(root / 'src')], binaries=[], datas=datas, hiddenimports=hidden,
    hookspath=[], hooksconfig={'matplotlib': {'backends': ['QtAgg', 'Agg', 'svg']}},
    runtime_hooks=[str(root / 'packaging/runtime_hook.py')],
    excludes=['tkinter', 'IPython', 'notebook', 'pytest', 'sphinx', *profile['exclude_modules']],
    noarchive=False)
pyz = PYZ(a.pure)
gui_scripts = [entry for entry in a.scripts if entry[0] != 'worker_entry']
worker_scripts = [entry for entry in a.scripts if entry[0] not in {'desktop_entry', 'pyi_rth_pyside6'}]
import sys
sys.path.insert(0, str(root / 'packaging'))
from crisp_splash import CrispSplash

splash = CrispSplash(str(root / 'assets/startup/opening.png'), binaries=a.binaries, datas=a.datas,
    # The generated Tcl command requires braces around a font with spaces.
    text_pos=(32, 214), text_size=11, text_font='{Microsoft YaHei UI}',
    text_color='#475569', text_default='正在加载运行组件，请稍候…',
    always_on_top=True, center='active')
gui = EXE(pyz, gui_scripts, splash, [], exclude_binaries=True, name='PYML-Workbench',
    manifest=str(root / 'packaging/gui.manifest'),
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False)
worker = EXE(pyz, worker_scripts, [], exclude_binaries=True, name='PYML-Worker',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
# Qt uses the unversioned Windows system ICU API. A same-named Poppler/Conda
# library found on the build host PATH exports versioned ICU symbols instead.
binaries = [entry for entry in a.binaries if Path(entry[0]).name.casefold() != 'icuuc.dll']
coll = COLLECT(gui, worker, binaries, a.datas, splash.binaries, strip=False, upx=False,
               name=f'PYML-Workbench-{profile_name}')
