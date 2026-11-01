"""Create an APA Word scaffold; example values are never new study results."""
from pathlib import Path
import argparse,csv,json

HERE=Path(__file__).resolve().parent

def build(output):
    from docx import Document
    from docx.shared import Cm,Pt,RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    output=Path(output);output.mkdir(parents=True,exist_ok=True);doc=Document();md=[];section=doc.sections[0]
    section.page_width=Cm(21.59);section.page_height=Cm(27.94)
    section.top_margin=section.bottom_margin=section.left_margin=section.right_margin=Cm(2.54)
    normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(11);normal.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体');normal.paragraph_format.line_spacing=2;normal.paragraph_format.first_line_indent=Cm(.85)
    for level in [1,2,3]:
        style=doc.styles[f'Heading {level}'];style.font.name='Times New Roman';style.font.size=Pt(12);style.font.bold=True;style.font.color.rgb=RGBColor(0,0,0);style.element.rPr.rFonts.set(qn('w:eastAsia'),'黑体');style.paragraph_format.line_spacing=2;style.paragraph_format.keep_with_next=True;style.paragraph_format.first_line_indent=Cm(0)
        style.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER if level==1 else WD_ALIGN_PARAGRAPH.LEFT
    header=section.header.paragraphs[0];header.alignment=WD_ALIGN_PARAGRAPH.RIGHT;field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');header._p.append(field)
    def h(text,level=1):doc.add_heading(text,level);md.append('#'*level+' '+text+'\n')
    def p(text):
        paragraph=doc.add_paragraph(text);md.append(text+'\n');return paragraph
    def table(number,title,headers,rows,note):
        para=doc.add_paragraph();para.paragraph_format.first_line_indent=0;para.add_run('表 '+str(number)).bold=True;para.paragraph_format.keep_with_next=True
        para=doc.add_paragraph();para.paragraph_format.first_line_indent=0;para.add_run(title).italic=True;para.paragraph_format.keep_with_next=True
        t=doc.add_table(rows=1,cols=len(headers))
        for c,v in zip(t.rows[0].cells,headers):c.text=v
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in rows:
            for c,v in zip(t.add_row().cells,row):c.text=str(v)
        for i,row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for c in row.cells:
                for para in c.paragraphs:
                    para.paragraph_format.first_line_indent=0;para.paragraph_format.line_spacing=1.15
                    for run in para.runs:run.font.size=Pt(9);run.bold=i==0
                borders=OxmlElement('w:tcBorders')
                for edge in ['top','bottom','left','right']:
                    element=OxmlElement('w:'+edge);element.set(qn('w:val'),'single' if (edge=='top' and i==0) or (edge=='bottom' and i in [0,len(t.rows)-1]) else 'nil');element.set(qn('w:sz'),'6');borders.append(element)
                c._tc.get_or_add_tcPr().append(borders)
        doc.add_paragraph('注。'+note);md.extend([f'**表 {number}** *{title}*\n',' | '.join(headers),' | '.join(['---']*len(headers)),*[' | '.join(str(v).replace('\n','；') for v in row) for row in rows],'\n注。'+note+'\n'])
    def figure(number,title,note):
        para=doc.add_paragraph();para.paragraph_format.first_line_indent=0;para.paragraph_format.keep_with_next=True;para.add_run('图 '+str(number)).bold=True
        para=doc.add_paragraph();para.paragraph_format.first_line_indent=0;para.paragraph_format.keep_with_next=True;para.add_run(title).italic=True
        md.append(f'**图 {number}** *{title}*\n')
        p('[[插入本轮实际结果绘制的图；不得直接沿用匿名示例的数值或图形]]').paragraph_format.keep_with_next=True;p('注。'+note)
    h('[[研究标题]]');p('[[作者]]\n[[单位]]\n[[日期]]');p('填写说明：所有双中括号内容须用本轮可核实事实替换；不适用项目写明原因。匿名示例仅供版式和表达参考。')
    doc.add_page_break();h('方法');h('研究目的与实验设计',2)
    p('本研究考察[[研究问题及目标含义]]。实验采用[[组间/被试内/混合设计、条件和重复测量结构]]；独立统计单位为[[被试/其他已确认单位]]。样本来源、纳入与排除标准为[[依据及实际人数]]。未知信息应标为缺失，不从文件名推断。')
    h('数据与变量',2);p('数据表[[编号]]包含[[记录数]]行和[[原始特征数]]个特征，缺失单元数为[[数量]]。每行代表[[整段/片段/试次等]]；身份、目标与特征分别由[[可核实来源]]确定。目标为[[真实条件/已授权人工夹具]]；若为夹具，所有分数仅验证接口。正文使用来源编号，个人路径保留在研究侧来源台账，不写入公共模板。')
    h('划分、预处理与降维',2);p('数据按[[被试组/时间/其他合理规则]]划分为[[训练/验证/最终测试范围]]。填补、缩放、特征选择和降维仅在训练分区拟合，并在内部交叉验证中重新拟合。堆叠内层OOF同样尊重[[组边界]]（Pedregosa et al., 2011; Varoquaux et al., 2017）。')
    p('实际降维为[[未执行/方法]]，输入[[p]]维、输出[[k]]维，拟合范围为[[范围]]。未选择列不应称为学习型降维；全部特征与算法输入契约例外分别说明。')
    h('模型、差异与预算',2);p('使用[[实际模型及版本]]。线性、近邻、条件分布、树集成及神经网络的目标与假设不同；平均集成与逐阶段提升的训练依赖也不同（Breiman, 2001; Pedregosa et al., 2011）。搜索规则、候选预算和停止条件预先指定（Bergstra & Bengio, 2012）。完整算法差异见附录，未使用的方法标为不适用，不能沿用示例PASS。')
    p('[[按实际方法保留以下引用，未使用则删除对应句与文末条目。]] 外部提升树实现遵循各自接口与参数约束（XGBoost developers, n.d.; LightGBM developers, n.d.; CatBoost developers, n.d.）。深度模型及其适配分别依据PyTorch和skorch契约（PyTorch contributors, n.d.; Skorch developers, n.d.）。HMM与单变量预测使用各自观测和历史定义（Hmmlearn developers, n.d.; Statsmodels developers, n.d.）。TPE与通用评价接口分别见相应文档（Optuna contributors, n.d.; Scikit-learn developers, n.d.）；t-SNE和UMAP表示目标及假设不同（van der Maaten & Hinton, 2008; UMAP developers, n.d.）。')
    p('外层重复[[R]]次；这不等于[[内层折数]]折或max_iter。随机种子、训练迭代、深度epoch及所有输入契约见配置台账。HMM按发射分布、序列边界和有限似然核验；人工时序或符号不冒充真实时间或观测。')
    h('指标、解释与接续',2);p('分类报告accuracy、balanced accuracy、macro F1及适用AUC，正类/多分类平均口径为[[定义]]。回归报告MAE、RMSE、适用R²及目标域相关偏差。无监督和HMM按任务输出，不强行填写分类指标。重复种子的SD不等于独立样本置信区间。')
    p('特征排序使用[[系数/原生解释/验证集置换及定义]]，包括[[全部列数]]列，保留负值和不可估计原因。阶段状态仅记录true/false；日志、错误与代码/数据/环境指纹分别保存。最终测试状态为[[实际调用次数及是否冻结]]，不能因模板填充而重跑最终test。')
    h('结果');h('实际覆盖与数据范围',2);p('本轮预定[[流程数]]条方法×目标×来源流程，各[[次数]]次，共[[槽位数]]项；通过[[数量]]，未通过[[数量]]，未执行[[数量]]。PASS只说明相应检查通过，不证明收敛、质量或科学效度。')
    table(1,'数据与实际执行范围',['来源编号','行×原始列','独立单位','实际模型输入','目标/状态'],[['[[来源]]','[[n×p]]','[[单位数]]','[[维数及例外]]','[[真实/人工、状态]]']],'不同来源、目标和评价单位分开，未映射与未选择分别记录。')
    h('分类结果与特征排序',2);p('预先指定的[[模型]]在[[来源/目标]]上的accuracy为[[M]]（SD = [[SD]]），AUC为[[M]]（SD = [[SD]]）。若AUC不可估计，应报告类别/分数缺失等实际原因。')
    table(2,'逐模型分类验证指标',['模型','有效次数','accuracy M (SD)','AUC M (SD)','macro F1'],[['[[模型]]','[[n]]','[[值]]','[[值或原因]]','[[值]]']],'写明正类、多分类OvR/OvO及macro/weighted；失败模型也保留。')
    figure(1,'各模型准确率与AUC对比','同一目标、来源和划分口径。误差条须注明SD或适用区间；参考线含义明确。')
    table(3,'特征排序',['排名','匿名特征ID','排序量 M (SD)','有效次数'],[['[[序号]]','[[ID]]','[[值]]','[[n]]']],'完整排序和可授权的原始特征映射另存；排序不等于因果或生理显著性。')
    figure(2,'固定模型的特征排序','注明展示范围、完整排序位置和负值处理；不按最好分数挑示例。')
    h('回归、计数、窗口及预测结果',2)
    table(4,'任务特定验证误差',['目标/来源','模型','有效次数','MAE','RMSE','R²/原因'],[['[[目标]]','[[模型]]','[[n]]','[[值]]','[[值]]','[[值或原因]]']],'跨任务不混排；少于2个观测或常数目标时解释R²不可估计。巨大有限误差保留，使用科学计数法。')
    figure(3,'回归误差及预测—目标对比','明确指标单位、评价样本数和坐标变换；必要时对数或对称对数刻度，但不能隐藏极端值。')
    h('HMM、无监督与失败诊断',2);p('HMM的[[训练/验证似然、观测数与发射维数]]为[[结果]]，不同观测空间的似然不作总榜。聚类、异常与降维的[[标签数量/输出维数/其他适用输出]]为[[结果]]；描述性全量拟合不等于独立验证。')
    table(5,'未通过类型与处理',['方法/阶段','错误类别','实际原因','处理及剩余限制'],[['[[方法]]','[[程序/接口/参数/数值/依赖/资源]]','[[证据]]','[[修复或限制]]']],'未定义指标、警告和拟合失败分别记录；不以删方法、挑种子或强制补概率制造PASS。')
    h('简短讨论');h('主要结果及设计含义',2);p('针对[[研究问题]]，结果支持[[证据可支持的范围]]。与[[实验设计及独立单位]]结合解释；不把人工目标、窗口数量或重复种子当作真实效应或新增独立样本。')
    h('限制及后续工作',2);p('本轮限制包括[[样本量、高维、目标来源、优化预算、缺失、依赖及验证限制]]。后续需[[真实标签/独立数据/预先冻结协议等]]。引用须与实际使用的方法对应，删除未使用方法的文献，而非机械保留全部。')
    doc.add_page_break();h('参考文献')
    for text in (HERE/'references/apa-references.txt').read_text(encoding='utf-8').split('\n\n'):
        text=text.strip()
        if not text:continue
        para=doc.add_paragraph(text);para.paragraph_format.left_indent=Cm(1.27);para.paragraph_format.first_line_indent=Cm(-1.27)
        for span in ['Journal of Machine Learning Research, 13','Journal of Machine Learning Research, 12','Journal of Machine Learning Research, 9','Machine Learning, 45','NeuroImage, 145','How training is performed','Tutorial','Features','optuna.samplers.TPESampler','torch.nn','User guide','NeuralNetRegressor','Time series analysis tsa','How UMAP works','Introduction to boosted trees']:
            if span in text:
                first,last=text.split(span,1);para.clear();para.add_run(first);para.add_run(span).italic=True;para.add_run(last);break
        md.append(text+'\n')
    doc.add_page_break();h('附录：算法差异与本轮配置')
    with (HERE/'references/algorithm-reference.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    for prefix,label in [('model:','模型'),('search:','搜索'),('reduction:','训练前降维')]:
        h(label,2);table('A'+str(['model:','search:','reduction:'].index(prefix)+1),label+'差异',['ID与名称','机制差异','本轮配置与实际状态'],[[r['ID']+'\n'+r['名称'],r['机制'],'[[使用/不适用、参数、输入及实际PASS]]'] for r in rows if r['ID'].startswith(prefix)],'基于已核实的方法清单供选择；当前版本、适用条件和来源需在本轮复核。完整官方链接随algorithm-reference.csv提供。')
    props=doc.core_properties;props.title='APA分析报告填写模板';props.author='';props.last_modified_by='';props.comments='';props.keywords='';props.identifier=''
    from datetime import datetime,timezone
    props.created=props.modified=datetime(2000,1,1,tzinfo=timezone.utc)
    doc.save(output/'apa_report_template.docx');(output/'apa_report_template.md').write_text('\n'.join(md),encoding='utf-8')
    (output/'template-status.json').write_text(json.dumps({'status':'unfilled_template','contains_new_analysis':False,'model_fitting':'NOT RUN','final_test_calls':0},indent=2),encoding='utf-8')
    return output/'apa_report_template.docx'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();print(build(args.output))
