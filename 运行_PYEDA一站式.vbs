Option Explicit

Dim shell, fso, root, script, python, command, config, code
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
script = fso.BuildPath(root, "skills\pyml-study-config\assets\templates\pyeda_ml\pyeda_ml_pipeline.py")
python = shell.Environment("PROCESS")("PYML_PYTHON")
If Len(python) = 0 Then
    python = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
    If Not fso.FileExists(python) Then python = "pythonw.exe"
End If
command = Quote(python) & " -X utf8 " & Quote(script) & " --gui"
If WScript.Arguments.Count > 0 Then
    config = fso.GetAbsolutePathName(WScript.Arguments(0))
    command = command & " --config " & Quote(config)
End If
If shell.Environment("PROCESS")("PYEDA_SMOKE") = "1" Then command = command & " --smoke"
code = shell.Run(command, 0, True)
If code <> 0 Then
    MsgBox "PYEDA startup failed (exit " & code & "). Run the CMD launcher to see details.", 16, "PYEDA"
End If
WScript.Quit code

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
