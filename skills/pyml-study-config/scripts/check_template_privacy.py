"""Scan skill text and document contents for local paths and private run IDs."""
from pathlib import Path
import argparse,json,re,zipfile

RULES={
    'absolute_windows_path':re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]'),
    'home_directory':re.compile(r'(?:/Users/|/home/|\\Users\\)[A-Za-z0-9_.-][^\s<>"/\\]*'),
    'private_run_id':re.compile(r'20\d{6}_\d{6}_[0-9a-fA-F]{6,}'),
}

def scan(root,forbidden=()):
    root=Path(root);issues=[];count=0;pdfs=0;docx=0
    def inspect(name,text):
        for rule,pattern in RULES.items():
            if pattern.search(text):issues.append({'file':name,'rule':rule})
        for phrase in forbidden:
            if phrase and phrase.lower() in text.lower():issues.append({'file':name,'rule':'forbidden_literal'})
    for file in sorted(root.rglob('*')):
        if not file.is_file() or '__pycache__' in file.parts:continue
        name=file.relative_to(root).as_posix();count+=1
        inspect(name,name)
        if file.suffix.lower() in ['.docx','.dotx']:
            docx+=1
            with zipfile.ZipFile(file) as archive:
                for part in archive.namelist():
                    if part.endswith(('.xml','.rels')):inspect(name+'!'+part,archive.read(part).decode('utf-8',errors='replace'))
        elif file.suffix.lower()=='.pdf':
            import pymupdf
            pdfs+=1
            with pymupdf.open(file) as pdf:
                inspect(name,json.dumps(pdf.metadata,ensure_ascii=False));inspect(name,pdf.get_xml_metadata())
                for page in pdf:inspect(name,page.get_text())
        elif file.suffix.lower() in ['.md','.txt','.py','.m','.json','.csv','.bib','.cmd','.ps1','.yml','.yaml'] or file.name=='.gitignore':
            data=file.read_bytes()
            inspect(name,data.decode('utf-16') if data.startswith((b'\xff\xfe',b'\xfe\xff')) else data.decode('utf-8-sig',errors='replace'))
    return {'status':'PASS' if not issues else 'FAIL','files_scanned':count,'docx_scanned':docx,'pdf_scanned':pdfs,'issues':issues}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--forbid',action='append',default=[]);args=parser.parse_args();result=scan(args.root,args.forbid);print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(bool(result['issues']))
