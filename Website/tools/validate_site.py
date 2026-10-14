from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json

ROOT=Path(__file__).resolve().parents[1]/'dist'
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
        if u.scheme or u.netloc:continue
        target=ROOT/unquote(u.path.lstrip('/')) if u.path.startswith('/') else p.parent/unquote(u.path)
        if not u.path:target=p
        if target.is_dir():target=target/'index.html'
        if not target.exists():errors.append(str(p)+': missing '+link)
        elif u.fragment and target.suffix=='.html' and u.fragment not in parsers[target].ids:errors.append(str(p)+': missing anchor '+link)
for p in ROOT.rglob('*'):
    if p.suffix.lower() in {'.docx','.pdf','.env','.py','.md'}:errors.append('Unexpected source document in public bundle: '+str(p))
report={'html_pages':len(parsers),'project_pages':len(list((ROOT/'projects').glob('*/index.html'))),'images':len(list((ROOT/'assets').glob('*.webp'))),'errors':errors,'public_bundle_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())}
print(json.dumps(report,indent=2));(Path(__file__).parent/'structural-check.json').write_text(json.dumps(report,indent=2))
raise SystemExit(bool(errors))
