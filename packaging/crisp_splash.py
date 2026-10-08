"""Draw the bootloader splash at the monitor's DPI instead of scaling a bitmap."""
from pathlib import Path

from PyInstaller.building.splash import Splash


class CrispSplash(Splash):
    def generate_script(self):
        script = super().generate_script()
        canvas_script = (Path(__file__).with_name('startup_canvas.tcl')).read_text(encoding='utf-8')
        script = script.replace('pack .root', canvas_script + '\npack .root', 1)
        Path(self.script_name).write_text(script, encoding='utf-8')
        return script
