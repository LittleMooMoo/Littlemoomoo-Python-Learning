from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json
import os

ROOT=Path(__file__).resolve().parents[1]/'dist'
ROOT=ROOT.resolve()
BASE_PATH='/' + os.environ.get('PORTFOLIO_BASE_PATH', '').strip('/') if os.environ.get('PORTFOLIO_BASE_PATH', '').strip('/') else ''
ORIGIN=urlsplit(os.environ.get('PORTFOLIO_ORIGIN', 'https://jiale-jiang-engineering.aglowbard1.chatgpt.site').rstrip('/'))
class Inspect(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.ids=set();self.images=[];self.h1=0;self.title=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='h1':self.h1+=1
        if tag=='title':self.title+=1
        for key in ('href','src'):
            if a.get(key):self.links.append(a[key])
        if tag=='img' and a.get('src'):self.images.append(a)

errors=[];parsers={}
for p in ROOT.rglob('*.html'):
    q=Inspect();q.feed(p.read_text(encoding='utf-8'));parsers[p]=q
    if q.h1!=1:errors.append(str(p)+': incorrect h1 count')
    if q.title!=1:errors.append(str(p)+': incorrect title count')
    for a in q.images:
        if not a.get('alt'):errors.append(str(p)+': missing image alt')
        if not(a.get('width') and a.get('height')):errors.append(str(p)+': missing intrinsic image dimensions')
for p,q in parsers.items():
    for link in q.links:
        u=urlsplit(link)
        if u.scheme or u.netloc:
            if u.scheme!=ORIGIN.scheme or u.netloc!=ORIGIN.netloc:continue
        path=unquote(u.path)
        if path.startswith('/') and BASE_PATH:
            if path==BASE_PATH:path='/'
            elif path.startswith(BASE_PATH+'/'):path=path[len(BASE_PATH):]
            else:
                errors.append(str(p)+': URL missing deployment prefix '+link)
                continue
        target=ROOT/path.lstrip('/') if path.startswith('/') else p.parent/path
        if not path:target=p
        target=target.resolve()
        if not target.is_relative_to(ROOT):
            errors.append(str(p)+': link escapes website root '+link)
            continue
        if target.is_dir():target=target/'index.html'
        if not target.exists():errors.append(str(p)+': missing '+link)
        elif u.fragment and target.suffix=='.html' and (target not in parsers or unquote(u.fragment) not in parsers[target].ids):errors.append(str(p)+': missing anchor '+link)
for p in ROOT.rglob('*'):
    if p.suffix.lower() in {'.docx','.env','.py','.md'}:errors.append('Unexpected source document in public bundle: '+str(p))
    if p.suffix.lower()=='.pdf' and not p.is_relative_to(ROOT/'documents'):
        errors.append('PDF must be intentionally published under documents/: '+str(p))
    if p.name=='quadruped-hardware.webp':errors.append('Excluded reference image remains in public bundle')
for p in ROOT.rglob('*.html'):
    if 'quadruped-hardware' in p.read_text(encoding='utf-8'):
        errors.append('Excluded reference image remains in page: '+str(p))
report={'html_pages':len(parsers),'project_pages':len(list((ROOT/'projects').glob('*/index.html'))),'images':len(list((ROOT/'assets').rglob('*.webp'))),'errors':errors,'public_bundle_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())}
print(json.dumps(report,indent=2));(Path(__file__).parent/'structural-check.json').write_text(json.dumps(report,indent=2))
raise SystemExit(bool(errors))
