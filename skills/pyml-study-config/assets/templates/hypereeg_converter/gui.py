"""A fixed PySide6 window for the one-click HyperEEG converter."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QVBoxLayout, QWidget

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def clock(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'

class Window(QMainWindow):
    def __init__(self, validate=False, smoke=False):
        super().__init__()
        self.validate, self.smoke = validate, smoke
        self.config = read(HERE/'settings.json')
        self.output = Path(self.config['output_root'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.run = None
        self.snapshots = []
        self.validation_dir = self.output/'ui-validation'
        self.validation_dir.mkdir(exist_ok=True)
        self.setWindowTitle('HyperEEG → PYML · 一键转换')
        self.resize(760, 460)
        self.setMinimumSize(660, 420)
        self.setFont(QFont('Microsoft YaHei UI', 10))
        body=QWidget();self.setCentralWidget(body)
        layout=QVBoxLayout(body);layout.setContentsMargins(32,28,32,26);layout.setSpacing(18)
        title=QLabel('HyperEEG 数据转换');title.setObjectName('title');layout.addWidget(title)
        subtitle=QLabel('原生特征表转换 · 原始数据保留');subtitle.setObjectName('subtitle');layout.addWidget(subtitle)
        card=QFrame();card.setObjectName('card');box=QVBoxLayout(card);box.setContentsMargins(24,20,24,22);box.setSpacing(16)
        self.stage=QLabel('准备启动');self.stage.setObjectName('stage');box.addWidget(self.stage)
        self.bar=QProgressBar();self.bar.setRange(0,0);self.bar.setTextVisible(False);self.bar.setFixedHeight(8);box.addWidget(self.bar)
        row=QHBoxLayout()
        for label, name in [('已用时间','elapsed'),('预计剩余（当前步骤）','remaining')]:
            col=QVBoxLayout();caption=QLabel(label);caption.setObjectName('caption');value=QLabel('00:00:00' if name=='elapsed' else '正在估计…');value.setObjectName('time');setattr(self,name,value);col.addWidget(caption);col.addWidget(value);row.addLayout(col)
        box.addLayout(row);layout.addWidget(card)
        self.status=QLabel('当前转换支持的标量特征，范围外内容将单独记录。');self.status.setWordWrap(True);layout.addWidget(self.status)
        path=QLabel('结果保存至  '+str(self.output));path.setObjectName('path');path.setWordWrap(True);layout.addWidget(path)
        layout.addStretch()
        buttons=QHBoxLayout();buttons.addStretch()
        self.stop_button=QPushButton('停止');self.stop_button.clicked.connect(self.cancel)
        self.open_button=QPushButton('打开结果');self.open_button.setEnabled(False);self.open_button.clicked.connect(self.open_result)
        buttons.addWidget(self.stop_button);buttons.addWidget(self.open_button);layout.addLayout(buttons)
        self.setStyleSheet("""
QMainWindow { background: #f3f6fb; color: #172640; }
QLabel#title { font-size: 25px; font-weight: 650; }
QLabel#subtitle,QLabel#caption,QLabel#path { color: #64748b; }
QFrame#card { background: white; border: 1px solid #dbe4ef; border-radius: 14px; }
QLabel#stage { font-size: 17px; font-weight: 600; }
QLabel#time { font-size: 28px; color: #1e40af; font-weight: 600; }
QProgressBar { border: none; background: #e8eef7; border-radius: 4px; }
QProgressBar::chunk { background: #3b6fe8; border-radius: 4px; }
QPushButton { background: white; border: 1px solid #cbd5e1; border-radius: 7px; padding: 8px 20px; }
QPushButton:disabled { color: #94a3b8; }
""")
        self.process=QProcess(self)
        env=QProcessEnvironment.systemEnvironment();env.insert('PYTHONUTF8','1');env.insert('PYML_PROGRESS_SILENT','1')
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.drain)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.timer=QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(500)
        self.launched_at=time.time();self.started=time.monotonic()
        self.log_file=self.output/('launcher_'+time.strftime('%Y%m%d_%H%M%S')+'.log')
        QTimer.singleShot(200,lambda:self.capture('initial') if self.validate or self.smoke else None)
        if smoke:
            self.stop_button.setEnabled(False);QTimer.singleShot(900,self.close)
        else:
            QTimer.singleShot(0,self.start)

    def start(self):
        self.launched_at=time.time();self.started=time.monotonic()
        self.process.setWorkingDirectory(str(HERE))
        self.process.start(str(Path(sys.executable).with_name('python.exe')),['-X','utf8',str(HERE/'convert.py'),'--no-open'])

    def drain(self):
        with self.log_file.open('ab') as stream:
            stream.write(bytes(self.process.readAllStandardOutput()))

    def refresh(self):
        if self.smoke:return
        if self.process.state()!=QProcess.ProcessState.NotRunning:
            self.elapsed.setText(clock(time.monotonic()-self.started))
        try:
            pointer=self.output/'latest-run.json'
            if pointer.stat().st_mtime < self.launched_at-0.2:return
            latest=read(pointer)
            if not latest.get('run'):return
            self.run=Path(latest['run'])
            state=read(self.run/'progress.json')
        except (FileNotFoundError,json.JSONDecodeError,PermissionError):return
        self.stage.setText(f"阶段 {state['stage']}/6 · {state['label']}")
        self.elapsed.setText(clock(state['elapsed_seconds']))
        eta=state.get('eta_seconds');self.remaining.setText('正在估计…' if eta is None else '约 '+clock(eta))
        if state.get('total'):
            self.bar.setRange(0,1000);self.bar.setValue(round(1000*state['completed']/state['total']))
        else:self.bar.setRange(0,0)
        if self.validate and state['elapsed_seconds']>=15 and 'running' not in self.snapshots:self.capture('running')
        if state['status']=='failed':
            self.stage.setText('转换失败');self.status.setText('已停止。请打开结果目录查看错误日志；原始数据保留。');self.remaining.setText('—')
        elif state['status']=='cancelled':
            self.stage.setText('已停止');self.status.setText('本轮部分输出已保留，未标记为转换完成。');self.remaining.setText('—')
        elif state['status']=='completed_supported_tables':
            self.stage.setText('支持范围内转换完成');self.remaining.setText('00:00:00')
            self.status.setText('数据已写入并通过核验。范围外及未映射项见结果说明。')

    def cancel(self):
        if self.run:
            (self.run/'cancel.request').write_text('user requested stop',encoding='utf-8')
            self.stop_button.setEnabled(False);self.status.setText('正在停止并保留本轮记录…')
        else:self.status.setText('正在启动；获得运行目录后可停止。')

    def finished(self, code, status):
        self.drain();self.refresh();self.stop_button.setEnabled(False);self.open_button.setEnabled(True)
        if code and self.stage.text() not in ['转换失败','已停止']:
            self.stage.setText('转换失败');self.remaining.setText('—');self.status.setText('启动或转换失败，请查看输出目录中的 launcher 日志。')
        if self.validate:
            self.capture('finished')
            (self.validation_dir/'ui-validation.json').write_text(json.dumps({'exit_code':code,'stage':self.stage.text(),'elapsed':self.elapsed.text(),'remaining':self.remaining.text(),'snapshots':self.snapshots,'run':str(self.run),'fixed_window':True},ensure_ascii=False,indent=2),encoding='utf-8')
            QTimer.singleShot(500,self.close)

    def process_error(self, error):
        if error==QProcess.ProcessError.FailedToStart:
            self.stage.setText('程序启动失败');self.status.setText(self.process.errorString());self.stop_button.setEnabled(False);self.open_button.setEnabled(True)

    def open_result(self):
        os.startfile(str(self.run or self.output))

    def capture(self, name):
        self.grab().save(str(self.validation_dir/(name+'.png')));self.snapshots.append(name)

    def closeEvent(self,event):
        if self.process.state()!=QProcess.ProcessState.NotRunning:
            self.cancel();event.ignore()
        else:event.accept()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    app=QApplication(sys.argv);window=Window(args.validate,args.smoke);window.show();sys.exit(app.exec())
