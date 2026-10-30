param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'PYML-Workbench'),
    [string]$ShortcutDirectory = [Environment]::GetFolderPath('Desktop')
)

$ErrorActionPreference = 'Stop'
$source = $PSScriptRoot
$info = Get-Content -LiteralPath (Join-Path $source 'build-info.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$version = [string]$info.version
if ($version -notmatch '^\d+\.\d+\.\d+$') { throw '无法识别程序版本。请重新下载完整安装包。' }
$base = [IO.Path]::GetFullPath($InstallRoot)
$destination = Join-Path $base $version
if ([IO.Path]::GetFullPath($source).TrimEnd('\') -eq $destination.TrimEnd('\')) {
    throw '程序已位于安装目录，直接运行 PYML-Workbench.exe 即可。'
}
if ($destination.StartsWith($source.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw '安装位置不能设在解压目录内部。请选择其他位置。'
}
foreach ($name in @('PYML-Workbench.exe', 'PYML-Worker.exe', '_internal', 'SHA256SUMS.txt')) {
    if (-not (Test-Path -LiteralPath (Join-Path $source $name))) { throw "程序包缺少 $name，请完整解压后重试。" }
}
if (Test-Path -LiteralPath $destination) {
    throw "目标版本目录已存在：$destination。为保留已有文件，本次安装已停止。可直接启动该目录中的程序，或选择新的安装位置。"
}
New-Item -ItemType Directory -Path $destination -Force | Out-Null
try {
    Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $destination -Recurse
    $shell = New-Object -ComObject WScript.Shell
    $desktop = [IO.Path]::GetFullPath($ShortcutDirectory)
    $shortcut = $shell.CreateShortcut((Join-Path $desktop "PY-ML Workbench $version.lnk"))
    $shortcut.TargetPath = Join-Path $destination 'PYML-Workbench.exe'
    $shortcut.WorkingDirectory = $destination
    $shortcut.Description = 'PY-ML 机器学习工作台'
    $shortcut.Save()
    Write-Host "安装完成：$destination"
    Write-Host "桌面快捷方式：PY-ML Workbench $version"
    Write-Host '无需管理员权限。旧版本、研究数据和个人设置均保留。'
} catch {
    Write-Host "安装未完成。已复制的文件保留在：$destination"
    throw
}
