"""Build a phone-friendly flat upload set and a full auditable source archive."""
from pathlib import Path
import hashlib,shutil,zipfile

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent/'mobile_v27_flat'
OUT=ROOT.parent/'outputs'
OUT.mkdir(exist_ok=True)

def write_zip(path,files,base):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for file in sorted(files,key=lambda p:p.relative_to(base).as_posix()):
            z.write(file,file.relative_to(base).as_posix())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None

core=[]
for path in ROOT.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
    rel=path.relative_to(ROOT)
    if rel.parts[0]=='kosto' or rel.as_posix() in {'config/default.yaml','cli.py','START_HERE.txt','V27_MODEL_CARD.md'}:
        core.append(path)
core_zip=WORK/'alpharadar_core.zip';write_zip(core_zip,core,ROOT)

for name in ('requirements.txt','requirements-pipeline.txt','START_HERE.txt','MOBILE_UPLOAD.txt','pipeline_workflow_COPY.txt'):
    shutil.copy2(ROOT/name,WORK/name)

flat=[p for p in WORK.iterdir() if p.is_file() and p.name!='FLAT_MANIFEST.sha256']
manifest=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(flat))
(WORK/'FLAT_MANIFEST.sha256').write_text(manifest)
flat=[p for p in WORK.iterdir() if p.is_file()]
mobile=OUT/'AlphaRadar_V27_Mobile_Flat.zip';write_zip(mobile,flat,WORK)
with zipfile.ZipFile(mobile) as z:
    assert all('/' not in name.rstrip('/') for name in z.namelist())

excluded_roots={'artifacts','data','.git','.pytest_cache','.venv'}
source=[p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] not in excluded_roots
        and '__pycache__' not in p.relative_to(ROOT).parts and p.suffix!='.pyc' and p.name!='SOURCE_MANIFEST.sha256']
source_manifest=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in sorted(source))
(ROOT/'SOURCE_MANIFEST.sha256').write_text(source_manifest);source.append(ROOT/'SOURCE_MANIFEST.sha256')
full=OUT/'AlphaRadar_V27_PRO_Global_Full.zip';write_zip(full,source,ROOT)
print(f'mobile={mobile} flat_files={len(flat)} full={full} source_files={len(source)}')
