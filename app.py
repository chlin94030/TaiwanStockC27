"""Flat mobile deployment bootstrap. Keep alpharadar_core.zip beside this file."""
from pathlib import Path
import hashlib,os,sys,tempfile,zipfile

DEPLOY=Path(__file__).resolve().parent
ARCHIVE=DEPLOY/'alpharadar_core.zip'
if not ARCHIVE.is_file():raise RuntimeError('缺少 alpharadar_core.zip，請將同層檔案全部上傳')
digest=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()[:16]
target=Path(tempfile.gettempdir())/('alpharadar_v27_'+digest);marker=target/'.ready'
if not marker.is_file():
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        members=z.infolist()
        if len(members)>150 or sum(x.file_size for x in members)>100_000_000:raise RuntimeError('核心檔案大小不合法')
        for x in members:
            out=(target/x.filename).resolve()
            if target.resolve() not in out.parents:raise RuntimeError('核心檔案路徑不合法')
        z.extractall(target)
    marker.write_text(digest)
sys.path.insert(0,str(target));os.environ['ALPHA_DESK_PROJECT_ROOT']=str(DEPLOY)
if (DEPLOY/'bundle.npz').is_file():os.environ.setdefault('ALPHA_DESK_BUNDLE',str(DEPLOY/'bundle.npz'))
from kosto.desk_ui import main
main()
