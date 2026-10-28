"""Flat mobile deployment CLI bootstrap for the existing GitHub workflow."""
from pathlib import Path
import hashlib,os,runpy,sys,tempfile,zipfile

DEPLOY=Path(__file__).resolve().parent;ARCHIVE=DEPLOY/'alpharadar_core.zip'
if not ARCHIVE.is_file():raise RuntimeError('缺少 alpharadar_core.zip')
digest=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()[:16]
target=Path(tempfile.gettempdir())/('alpharadar_v27_'+digest);marker=target/'.ready'
if not marker.is_file():
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        for x in z.infolist():
            if target.resolve() not in (target/x.filename).resolve().parents:raise RuntimeError('核心檔案路徑不合法')
        z.extractall(target)
    marker.write_text(digest)
sys.path.insert(0,str(target));os.environ['ALPHA_DESK_PROJECT_ROOT']=str(DEPLOY)
runpy.run_path(str(target/'cli.py'),run_name='__main__')
