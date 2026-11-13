# PYEDA 特征转换与质控示例

示例复用 [PYEDA 转换插件](https://github.com/hexiovo/PY-ML_support_trans/tree/master/pyeda)。使用 Python 3.12，命令行只需标准库。Windows 图形界面使用随 Python 安装的 Tk，双击 `Start.pyw` 由 `pythonw.exe` 打开进度窗口。选择插件仓库的 `pyeda` 源码目录和输出根目录，结果包留空即可运行合成数据。

也可以在本目录运行以下命令，替换插件源码目录。输出路径须是新的目录。

```powershell
python -X utf8 convert_features.py --plugin ../PY-ML_support_trans/pyeda --output ./pyeda-demo
```

提供 `--input <PYEDA结果目录>` 可处理自有结果包。CLI 通过 `--source` 选择参与者任务角色表、参与者暴露表或记录级特征。记录级输入默认筛选 `status=valid`。

- `conversion/` 保存转换器原始输出、有限值掩码、特征字典、来源哈希和进度日志。
- `features.csv` 核对数据与掩码的表头及样本顺序后，把掩码为 0 的值统一写为空白，保留全部登记特征及必要分组键。
- `quality.json` 记录各特征缺失数、全缺失列和总行数。合成示例预期得到 12 行、3 项特征，其中一列全缺失、一列有一个非有限值。

示例覆盖特征转换与质控，产物可供后续独立配置建模。自有结果包的输出目录含原始身份键和来源路径，应保存在本地。原始记录与已有输出可保留，每次运行使用新目录。
