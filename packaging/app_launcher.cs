// Single visible APP entry; the frozen runtime stays in the project release.
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("PY-ML 机器学习工作台")]
[assembly: AssemblyDescription("PY-ML 工作台启动入口")]
[assembly: AssemblyVersion("0.4.4.0")]

internal static class AppLauncher
{
    [STAThread]
    private static void Main()
    {
        try
        {
            string directory = Path.GetFullPath(Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                @"..\PY-ML\dist\ui-polish-20261008\PYML-Workbench-standard"));
            string executable = Path.Combine(directory, "PYML-Workbench.exe");
            if (!File.Exists(executable) || !File.Exists(Path.Combine(directory, "PYML-Worker.exe"))
                || !Directory.Exists(Path.Combine(directory, "_internal")))
                throw new FileNotFoundException("程序运行目录不完整，请保留：\n" + directory);
            using (Process application = Process.Start(new ProcessStartInfo
            {
                FileName = executable,
                WorkingDirectory = directory,
                UseShellExecute = false,
                CreateNoWindow = true
            }))
            {
                application.WaitForExit();
                Environment.ExitCode = application.ExitCode;
            }
        }
        catch (Exception error)
        {
            MessageBox.Show("无法启动 PY-ML 工作台。\n\n" + error.Message,
                "启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            Environment.ExitCode = 1;
        }
    }
}
