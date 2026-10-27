"""XGBoost loads its native library and VERSION file relative to the package."""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

binaries = collect_dynamic_libs('xgboost')
datas = collect_data_files('xgboost', includes=['VERSION'])
