"""Build Jiale Jiang's static, multi-page engineering portfolio.

All personal and project copy is curated from supplied documents and repository
evidence. Original materials remain outside this checkout and are never changed.
Run with Python 3; no build dependencies are required.
"""
from pathlib import Path
from html import escape
from urllib.parse import quote, urlsplit, urlunsplit, unquote
import json
import os
import re
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dist'
HOST_ORIGIN = os.environ.get('PORTFOLIO_ORIGIN', 'https://jiale-jiang-engineering.aglowbard1.chatgpt.site').rstrip('/')
origin_parts = urlsplit(HOST_ORIGIN)
if origin_parts.scheme not in ('http', 'https') or not origin_parts.netloc or origin_parts.path or origin_parts.query or origin_parts.fragment:
    raise ValueError('PORTFOLIO_ORIGIN must be an HTTP(S) origin without a path, query, or fragment.')
raw_base_path = os.environ.get('PORTFOLIO_BASE_PATH', '').strip()
if any(token in raw_base_path for token in ('://', '//', '?', '#', '\\')) or any(unquote(part) in ('.', '..') for part in raw_base_path.split('/')):
    raise ValueError('PORTFOLIO_BASE_PATH must be a URL path prefix without traversal, query, or fragment.')
BASE_PATH = '/' + raw_base_path.strip('/') if raw_base_path.strip('/') else ''
ORIGIN = HOST_ORIGIN + BASE_PATH
EMAIL = 'jiaxi9142@gmail.com'
LINKEDIN = 'https://www.linkedin.com/in/jiale-jiang-2277652a9/'
GITHUB = 'https://github.com/LittleMooMoo'
CYCLICAL_REPO = GITHUB + '/3D-Printed-17-1-Cycloidal-Robotic-Actuator'
ASSET_DIMENSIONS = json.loads((ROOT/'asset-dimensions.json').read_text()) if (ROOT/'asset-dimensions.json').exists() else {}

def site_path(url):
    """Mount internal links under a repository path without changing external links."""
    if not BASE_PATH or not url.startswith('/') or url.startswith('//'):
        return url
    parts = urlsplit(url)
    if parts.path == BASE_PATH or parts.path.startswith(BASE_PATH + '/'):
        return url
    return urlunsplit(('', '', BASE_PATH + parts.path, parts.query, parts.fragment))

def rewrite_site_urls(document):
    def dimensions(match):
        tag = match.group(0)
        src = re.search(r'''\bsrc=["']([^"']+)["']''', tag)
        if src and src[1].startswith('/assets/') and ' width=' not in tag:
            size = ASSET_DIMENSIONS.get(unquote(src[1][len('/assets/'):]))
            if size:
                tag = tag[:-1] + f' width="{size[0]}" height="{size[1]}">'
        return tag
    document = re.sub(r'<img\b[^>]*>', dimensions, document)
    if not BASE_PATH:
        return document
    return re.sub(r'''\b(href|src)=(["'])(/(?!/)[^"']*)\2''',
                  lambda match: match[1] + '=' + match[2] + site_path(match[3]) + match[2],
                  document)

def anchor(url, label, cls='line-link', external=False):
    extra = ' target="_blank" rel="noopener noreferrer"' if external else ''
    return f'<a class="{cls}" href="{escape(url, quote=True)}"{extra}>{label}<span aria-hidden="true">↗</span></a>'

def image(name, alt, eager=False, cls=''):
    loading = 'eager' if eager else 'lazy'
    priority = ' fetchpriority="high"' if eager else ''
    size = ASSET_DIMENSIONS.get(name+'.webp')
    dims = f' width="{size[0]}" height="{size[1]}"' if size else ''
    return f'<img class="{cls}" src="/assets/{name}.webp" alt="{escape(alt, quote=True)}" loading="{loading}" decoding="async"{priority}{dims}>'

def figure(name, caption, alt=None, photo=False, full=False):
    classes = 'evidence-image' + (' photo' if photo else '') + (' full-evidence' if full else '')
    return f'''<figure><div class="{classes}"><button class="image-button" data-lightbox data-caption="{escape(caption, quote=True)}" aria-label="Enlarge image: {escape(alt or caption, quote=True)}">{image(name,alt or caption)}<span class="zoom" aria-hidden="true">View larger ↗</span></button></div><figcaption class="figure-caption">{caption}</figcaption></figure>'''

def equation(label, formula, explanation):
    return f'<div class="equation-panel"><div class="equation-label">{label}</div><div class="equation" role="math" aria-label="{escape(label,quote=True)}">{formula}</div><p>{explanation}</p></div>'

def code(filename, language, content, url=None):
    link = anchor(url, 'Source', cls='', external=True) if url else f'<span>{language}</span>'
    return f'<div class="code-panel"><div class="code-header"><span>{filename}</span>{link}</div><pre><code>{escape(content)}</code></pre></div>'

def outcome(title, text):
    return f'<div class="outcome"><span class="outcome-label">PROJECT STATUS</span><h3>{title}</h3><p>{text}</p></div>'

def sources(items):
    return '<div class="source-links">' + ''.join(anchor(url, label,cls='',external=True) for label,url in items) + '</div>'

def header(active):
    items = [('projects','/projects/','Projects'),('notes','/engineering-notes/','Derivations'),('about','/about/','About & education'),('experience','/experience/','Experience'),('contact','/contact/','Contact')]
    links = ''.join(f'<a href="{href}"'+(' aria-current="page"' if active==key else '')+f'>{label}</a>' for key,href,label in items)
    return f'''<a class="skip" href="#main">Skip to content</a><header class="site-header"><div class="wrap nav-inner"><a href="/" class="brand" aria-label="Jiale Jiang home"><span class="brand-mark" aria-hidden="true">JJ</span><span>Jiale Jiang<span class="brand-sub">Engineering portfolio</span></span></a><button class="nav-toggle" aria-label="Open navigation" aria-expanded="false" aria-controls="main-nav"><svg width="21" height="18" viewBox="0 0 21 18" aria-hidden="true"><path d="M1 5h19M1 13h19" fill="none" stroke="currentColor" stroke-width="1.4"/></svg></button><nav class="nav-links" id="main-nav" aria-label="Main navigation">{links}<a class="nav-resume" href="/resume/">Résumé ↗</a></nav></div></header>'''

def footer():
    return f'''<footer class="site-footer"><div class="wrap"><div class="footer-inner"><span>© 2026 Jiale Jiang · Southern California</span><div class="footer-links"><a href="{LINKEDIN}" target="_blank" rel="noopener noreferrer">LinkedIn ↗</a><a href="{GITHUB}" target="_blank" rel="noopener noreferrer">GitHub ↗</a><a href="mailto:{EMAIL}">Email ↗</a></div></div><p class="footer-note">Project photographs, CAD, and technical work from my engineering practice. Team contributions and development stages are described on each project page.</p></div></footer>'''

def closing():
    return f'''<section class="closing dark"><div class="wrap closing-inner"><div><span class="eyebrow">NEXT CONVERSATION</span><h2>Let’s talk engineering.</h2><p>Robotics internships, research, and people who enjoy building things.</p></div>{anchor('/contact/','Get in touch',cls='pill')}</div></section>'''

def page(path,title,description,body,active='',include_closing=False):
    name = {'/':'home'}.get(path, path.strip('/'))
    editable = ROOT/'content'/'pages'/(name+'.html')
    if editable.is_file():
        body = editable.read_text(encoding='utf-8')
    body = body.replace('<!--PROJECT_CARDS-->', ''.join(card(p) for p in PROJECTS[:4]))
    entries = ''.join(f'<article class="resume-entry"><h3><a href="/projects/{p["slug"]}/">{p["title"]} ↗</a></h3><p class="date">{p["role"]} · {p["period"]}</p><p>{p["summary"]} {p["stage"]}.</p></article>' for p in PROJECTS)
    body = body.replace('<!--RESUME_PROJECTS-->', entries)
    dest=OUT / path.strip('/') / 'index.html' if path!='/' else OUT/'index.html'
    dest.parent.mkdir(parents=True,exist_ok=True)
    url=ORIGIN+path
    doc=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#ffffff"><title>{escape(title)} | Jiale Jiang</title><meta name="description" content="{escape(description,quote=True)}"><link rel="canonical" href="{url}"><meta property="og:type" content="website"><meta property="og:title" content="{escape(title,quote=True)} | Jiale Jiang"><meta property="og:description" content="{escape(description,quote=True)}"><meta property="og:url" content="{url}"><meta property="og:image" content="{ORIGIN}/assets/quadruped-hero.webp"><meta name="twitter:card" content="summary_large_image"><link rel="icon" href="/assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="/assets/site.css"><script src="/assets/site.js" defer></script></head><body>{header(active)}<main id="main">{body}</main>{closing() if include_closing else ''}{footer()}<dialog class="lightbox" aria-label="Project image viewer"><div class="lightbox-top"><button class="lightbox-close" autofocus>Close ×</button></div><img alt=""><p class="lightbox-caption"></p></dialog></body></html>'''
    dest.write_text(rewrite_site_urls(doc),encoding='utf-8')

PROJECTS = json.loads((ROOT / 'content' / 'projects.json').read_text(encoding='utf-8'))

def card(p):
    cls='project-card'+(' photo' if p.get('photo') else '')+(' black' if p.get('black') else '')
    return f'''<a class="{cls}" data-category="{p['category']}" href="/projects/{p['slug']}/"><div class="card-media">{image(p['image'],p['alt'])}<span class="card-status">{p['status']}</span></div><div class="card-meta"><span>{p['kind']}</span><span aria-hidden="true">0{PROJECTS.index(p)+1}</span></div><div class="card-title"><h3>{p['title']}</h3><span class="arrow" aria-hidden="true">↗</span></div><p>{p['summary']}</p></a>'''

def project_page(p,lede,hero_caption,sections):
    facts=''.join(f'<div class="fact"><span class="label">{label}</span><span class="value">{val}</span></div>' for label,val in [('My role',p['role']),('Timeline',p['period']),('Tools',p['tools']),('Stage',p['stage'])])
    toc=''.join(f'<a href="#{key}">{title}</a>' for key,title,_ in sections)
    content=''.join(f'<section id="{key}"><h2>{title}</h2>{html}</section>' for key,title,html in sections)
    hero_cls='project-main-image'+(' dark-media' if p.get('black') else '')
    hero=f'''<figure><div class="{hero_cls}"><button class="image-button" data-lightbox data-caption="{escape(hero_caption,quote=True)}" aria-label="Enlarge {escape(p['alt'],quote=True)}">{image(p['image'],p['alt'],eager=True)}<span class="zoom" aria-hidden="true">View larger ↗</span></button></div><figcaption class="figure-caption">{hero_caption}</figcaption></figure>'''
    nxt=PROJECTS[(PROJECTS.index(p)+1)%len(PROJECTS)]
    body=f'''<div class="wrap"><header class="project-hero"><div class="breadcrumbs"><a href="/projects/">Projects</a><span aria-hidden="true">/</span><span>{p['title']}</span></div><span class="eyebrow">{p['kind']}</span><h1>{p['title']}.</h1><p class="lede">{lede}</p><div class="project-tags">{''.join(f'<span>{tag}</span>' for tag in p['tools'].split(' · '))}</div></header>{hero}<div class="fact-row">{facts}</div><div class="project-layout"><aside class="project-toc" aria-label="On this page"><p>ON THIS PAGE</p>{toc}</aside><article class="project-body">{content}</article></div><section class="related"><div class="related-inner"><div><span class="eyebrow">NEXT PROJECT</span><h3>{nxt['title']}</h3></div>{anchor('/projects/'+nxt['slug']+'/','Explore project')}</div></section></div>'''
    page('/projects/'+p['slug']+'/',p['title'],lede,body,'projects')

def build_home():
    intro='''<section class="section"><div class="wrap"><div class="intro-grid"><div><span class="eyebrow">MY APPROACH</span><h2>Understanding the system.<br>Building the details.</h2></div><div><p>I’m a mechanical engineering student who learns by turning equations into things I can build and measure. Robotics gives me a reason to connect the mechanism, the circuit, and the code.</p><p class="muted">My projects document what I designed, what I tested, and what I’m still working to understand.</p></div></div><div class="capabilities"><div class="capability"><span class="num">01 / SYSTEMS REASONING</span><h3>Follow the whole problem.</h3><p>I trace issues across mechanical interfaces, wiring, coordinate conventions, and control software.</p></div><div class="capability"><span class="num">02 / MEASURED ITERATION</span><h3>Let tests guide the revision.</h3><p>Actuator prototypes and bench tests inform changes to bearings, clearances, geometry, and manufacturing.</p></div><div class="capability"><span class="num">03 / TECHNICAL COLLABORATION</span><h3>Make the work understandable.</h3><p>I document assumptions and help teammates take ownership of small, testable subsystems.</p></div></div></div></section>'''
    chosen=''.join(card(PROJECTS[i]) for i in [0,1,2,3])
    body=f'''<section class="home-hero"><div class="wrap"><div class="hero-grid"><div class="hero-copy"><span class="eyebrow">ROBOTICS & MECHANICAL ENGINEERING</span><h1>Jiale Jiang.</h1><p class="hero-role">Mechanical engineering student.<br>Building robotic systems.</p><p class="lede">I connect mechanical design, electronics, and control software through hands-on robotics projects.</p><div class="actions">{anchor('/projects/','Explore my work',cls='pill')}{anchor('/about/','About me',cls='pill secondary')}</div></div><div class="hero-art">{image('quadruped-hero','SolidWorks CAD of my revised modular quadruped, with exposed actuators and curved lower legs.',eager=True)}<span class="art-note">Modular quadruped · Revised mechanical CAD</span></div></div><div class="hero-index"><div class="index-items"><span><strong>Based in</strong> Southern California</span><span><strong>Studying at</strong> Mount San Antonio College</span></div><span class="availability"><span class="dot" aria-hidden="true"></span> Open to engineering opportunities</span></div></div></section>{intro}<section class="section wash"><div class="wrap"><div class="section-head"><div><span class="eyebrow">SELECTED WORK</span><h2>From a joint to a robot.</h2></div>{anchor('/projects/','All projects')}</div><div class="projects-grid">{chosen}</div><a class="featured-card" href="/projects/simulation-control/"><div class="feature-copy"><span class="eyebrow">MODELING & CONTROL</span><h3>From CAD to simulation.</h3><p>Analytical kinematics, corrected mass properties, and the foundations of a robot learning pipeline.</p><span class="line-link">Explore the model <span aria-hidden="true">↗</span></span></div>{image('robot-simulation','MuJoCo scene of the CAD-derived quadruped model.')}</a></div></section><section class="section"><div class="wrap intro-grid"><div><span class="eyebrow">THE FOUNDATION</span><h2>Learning that<br>reaches the workbench.</h2></div><div><p>Calculus, physics, and statics give me tools for understanding a mechanism. Building and testing it gives me new questions to bring back to those tools.</p><div class="actions">{anchor('/about/#education','Education & coursework')}{anchor('/experience/','Work experience')}</div></div></div></section>'''
    page('/','Robotics & Engineering Portfolio','Jiale Jiang is a mechanical engineering student building robotics projects across CAD, actuator design, electronics, and control software.',body,include_closing=True)

def build_projects():
    body=f'''<div class="wrap"><header class="page-heading"><span class="eyebrow">ENGINEERING PROJECTS</span><h1>The work,<br>and the reasoning behind it.</h1><p class="lede">Mechanisms, electronics, analysis, and control. Each project explains my contribution, the evidence, and the next engineering question.</p></header><section class="section-small" style="padding-top:8px"><div class="filters" role="group" aria-label="Filter projects"><button class="filter" data-filter="all" aria-pressed="true">All work</button><button class="filter" data-filter="mechanical" aria-pressed="false">Mechanical design</button><button class="filter" data-filter="electronics" aria-pressed="false">Electronics</button><button class="filter" data-filter="controls" aria-pressed="false">Control & simulation</button></div><p id="filter-status" class="skip" aria-live="polite"></p><div class="projects-grid">{''.join(card(p) for p in PROJECTS)}</div></section></div>'''
    page('/projects/','Engineering projects','Explore Jiale Jiang’s engineering case studies in quadruped robotics, actuator design, motor control, simulation, and underwater thrusters.',body,'projects',True)

def build_case_studies():
    for p in PROJECTS:
        sections = [(s['id'], s['title'], (ROOT/'content'/s['file']).read_text(encoding='utf-8')) for s in p['sections']]
        project_page(p, p['lede'], p['hero_caption'], sections)


def build_extra_pages():
    config = ROOT/'content'/'extra-pages.json'
    if not config.exists():
        return []
    pages = json.loads(config.read_text(encoding='utf-8'))
    for p in pages:
        body = (ROOT/'content'/p['file']).read_text(encoding='utf-8')
        page(p['path'], p['title'], p['description'], body, p.get('active','projects'))
    return [p['path'] for p in pages]


def build_about():
    body=f'''<div class="wrap"><header class="page-heading"><span class="eyebrow">ABOUT & EDUCATION</span><h1>I learn by building,<br>then asking why.</h1><p class="lede">Mechanical engineering student. Robotics systems developer. Interested in the point where mechanics, control, and real hardware meet.</p></header><section class="section-small" style="padding-top:20px"><div class="about-grid"><div><span class="small-label">Jiale Jiang · Southern California</span><h2>From an equation<br>to a working part.</h2></div><div class="story"><p>I learn best when an equation becomes something I can build and measure. That curiosity has taken me from a ducted ROV thruster to printed actuators and a twelve-joint quadruped.</p><p>Working across mechanics, electronics, and software has taught me to question assumptions, document failed tests, and make each subsystem understandable to the people working alongside me.</p><p>I’m continuing to develop those habits through coursework, independent technical study, and collaborative robotics work.</p></div></div></section><section class="section-small" id="education"><span class="eyebrow">ACADEMIC FOUNDATION</span><h2>Education & coursework.</h2><div class="education-card"><h3>Mount San Antonio College</h3><p class="school-meta">Mechanical engineering transfer preparation · Walnut, California<br>2023 — present</p><p class="recognition">President’s List · Fall 2025 and Spring 2026</p><div class="course-grid"><div><h4>Completed coursework</h4><ul><li>Calculus I, II, and III</li><li>Linear Algebra & Differential Equations</li><li>Engineering Physics I and II</li><li>Statics</li><li>Engineering Graphics</li><li>Chemistry</li></ul></div><div><h4>In progress · Fall 2026</h4><ul><li>Dynamics</li><li>MATLAB for Engineers</li><li>Properties of Materials</li><li>Critical Thinking and Writing · English C1001</li></ul><p class="semester">Course status reflects October 2026.</p></div></div></div></section><section class="section-small"><span class="eyebrow">TOOLS I USE IN MY PROJECTS</span><h2>Across the engineering stack.</h2><div class="tools-grid"><div><h3>Mechanical design</h3><p>SolidWorks, parametric CAD, FDM prototyping, assembly design, engineering drawings, structural and flow analysis.</p></div><div><h3>Electronics & embedded</h3><p>KiCad, STM32 development, magnetic encoder circuits, PCB assembly and rework, CAN communication, bench diagnostics.</p></div><div><h3>Software & modeling</h3><p>Python, C++, Eigen, Linux / SocketCAN, MATLAB, MuJoCo, analytical kinematics, numerical validation, learning-pipeline experiments.</p></div></div></section><section class="section-small"><div class="collaboration-note"><h3>Make room for other people to contribute.</h3><p>I’ve helped teammates learn soldering, assembly, CAN communication, and technical testing. Breaking a subsystem into small checks helped teammates own work such as the IMU integration, while keeping the interfaces understandable to the rest of the team.</p><div class="actions">{anchor('/projects/','See the projects')}{anchor('/experience/','Earlier work experience')}</div></div></section></div>'''
    page('/about/','About & education','Meet Jiale Jiang and explore his mechanical engineering studies, completed coursework, ongoing classes, and approach to collaborative robotics.',body,'about',True)

def build_experience():
    body=f'''<div class="wrap"><header class="page-heading"><span class="eyebrow">WORK EXPERIENCE</span><h1>Clear questions.<br>Practical tradeoffs.<br>Follow-through.</h1><p class="lede">My earlier sales roles taught me to understand a person’s needs, explain choices, and coordinate across teams. I bring those habits to engineering collaboration.</p></header><section class="section-small" style="padding-top:0"><article class="experience-row"><div><h2>Living Spaces</h2><p class="place">Monrovia, California</p><p class="date">May — November 2024</p></div><div><h3>Sales associate</h3><p>I translated customer needs into furniture options, explained product tradeoffs, and coordinated multi-item orders. Following through across teams was a regular part of helping a customer reach a workable solution.</p></div></article><article class="experience-row"><div><h2>Floor & Decor</h2><p class="place">San Gabriel, California</p><p class="date">January — May 2024</p></div><div><h3>Sales associate</h3><p>I helped customers with material selection, quantity planning, and installation considerations, while supporting inventory and team operations.</p></div></article></section><section class="section-small"><div class="intro-grid"><div><span class="eyebrow">WHERE THESE HABITS SHOW UP NOW</span><h2>Communication is part<br>of engineering.</h2></div><div><p class="lede">In robotics, I use the same habit of asking questions before proposing a solution. Clear assumptions, understandable test steps, and reliable handoffs make technical work easier to share.</p><div class="actions">{anchor('/projects/','Explore my engineering work')}{anchor('/resume/','View résumé')}</div></div></div></section></div>'''
    page('/experience/','Work experience','Jiale Jiang’s prior sales experience at Living Spaces and Floor & Decor, and the communication habits he brings to engineering.',body,'experience',True)

def build_contact():
    body=f'''<div class="wrap"><header class="page-heading"><span class="eyebrow">CONTACT</span><h1>Let’s build<br>something useful.</h1><p class="lede">I’m interested in engineering internships, robotics research, and conversations with people working across mechanics, electronics, and control.</p></header><section class="section-small" style="padding-top:12px"><div class="contact-grid"><div><h2>Start a conversation.</h2><div class="contact-links"><div class="contact-item"><div><p class="label">EMAIL</p><a href="mailto:{EMAIL}">{EMAIL}</a></div><button class="copy-email" data-copy-email="{EMAIL}">Copy email</button></div><div class="contact-item"><div><p class="label">PROFESSIONAL PROFILE</p><a href="{LINKEDIN}" target="_blank" rel="noopener noreferrer">LinkedIn</a></div><span aria-hidden="true">↗</span></div><div class="contact-item"><div><p class="label">PUBLIC PROJECTS</p><a href="{GITHUB}" target="_blank" rel="noopener noreferrer">GitHub</a></div><span aria-hidden="true">↗</span></div><div class="contact-item"><div><p class="label">LOCATION</p><p>Southern California</p></div></div></div></div><aside class="contact-panel"><span class="eyebrow">A QUICK OVERVIEW</span><h3>My background,<br>in one place.</h3><p>The résumé page brings together my engineering projects, education, technical tools, and previous work experience. You can print it or save it as a PDF.</p>{anchor('/resume/','View résumé',cls='pill')}</aside></div></section></div>'''
    page('/contact/','Contact','Contact Jiale Jiang about engineering internships, robotics research, and professional collaboration.',body,'contact')

def build_resume():
    entries=''.join(f'<article class="resume-entry"><h3><a href="/projects/{p["slug"]}/">{p["title"]} ↗</a></h3><p class="date">{p["role"]} · {p["period"]}</p><p>{p["summary"]} {p["stage"]}.</p></article>' for p in PROJECTS)
    body=f'''<div class="wrap resume-wrap section-small"><div class="resume-intro"><div><span class="eyebrow no-print">PROFESSIONAL OVERVIEW</span><h1>Jiale Jiang</h1><p class="lede" style="margin-top:14px">Mechanical engineering student · Robotics systems developer</p><p class="muted" style="font-size:13px;margin-top:14px">Southern California · <a href="mailto:{EMAIL}">{EMAIL}</a><br><a href="{LINKEDIN}">LinkedIn</a> · <a href="{GITHUB}">GitHub</a></p><p class="print-only">Portfolio: {ORIGIN}</p></div><div class="no-print"><button class="pill" data-print>Print / save PDF ↗</button></div></div><section class="resume-section"><h2>Profile</h2><p>I develop robotics projects across mechanical design, printed hardware, electronics, and control software. I use physical tests and numerical checks to guide revisions, and help teammates understand the interfaces between subsystems.</p></section><section class="resume-section"><h2>Engineering projects</h2>{entries}<p style="margin-top:16px">Explore the project pages for design details, testing evidence, and current development stages.</p></section><section class="resume-section"><h2>Education</h2><h3>Mount San Antonio College</h3><p>Mechanical engineering transfer preparation · 2023 — present</p><p>President’s List: Fall 2025 and Spring 2026.</p><p><strong>Completed:</strong> Calculus I–III; Linear Algebra & Differential Equations; Engineering Physics I–II; Statics; Engineering Graphics; Chemistry.</p><p><strong>In progress, Fall 2026:</strong> Dynamics; MATLAB for Engineers; Properties of Materials; Critical Thinking and Writing.</p></section><section class="resume-section"><h2>Technical tools</h2><p>SolidWorks · FDM fabrication · KiCad · STM32 development · Python · C++ / Eigen · Linux / SocketCAN · MATLAB · MuJoCo · Numerical modeling and verification.</p></section><section class="resume-section"><h2>Work experience</h2><article class="resume-entry"><h3>Living Spaces · Sales associate</h3><p class="date">Monrovia, CA · May — November 2024</p><p>Explained product tradeoffs, translated customer needs into furniture solutions, and coordinated multi-item orders across teams.</p></article><article class="resume-entry"><h3>Floor & Decor · Sales associate</h3><p class="date">San Gabriel, CA · January — May 2024</p><p>Supported material selection, quantity planning, installation considerations, inventory, and team operations.</p></article></section></div>'''
    page('/resume/','Résumé','A printable professional overview of Jiale Jiang’s engineering projects, education, skills, and work experience.',body)

def build_404():
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Page not found | Jiale Jiang</title><link rel="stylesheet" href="/assets/site.css"></head><body>{header('')}<main id="main" class="wrap section"><span class="eyebrow">404</span><h1>That page isn’t here.</h1><p class="lede" style="margin-top:24px">You can return to the portfolio or explore the project pages.</p><div class="actions">{anchor('/','Home',cls='pill')}{anchor('/projects/','Projects',cls='pill secondary')}</div></main>{footer()}<script src="/assets/site.js" defer></script></body></html>'''
    (OUT/'404.html').write_text(rewrite_site_urls(doc),encoding='utf-8')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'public').exists():
        shutil.copytree(ROOT/'public', OUT, dirs_exist_ok=True)
    build_home();build_projects();build_case_studies();build_about();build_experience();build_contact();build_resume();build_404()
    extra_paths = build_extra_pages()
    paths=['/','/projects/','/about/','/experience/','/contact/','/resume/']+['/projects/'+p['slug']+'/' for p in PROJECTS] + extra_paths
    urls=''.join(f'<url><loc>{ORIGIN+p}</loc><lastmod>2026-10-21</lastmod></url>' for p in paths)
    (OUT/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+urls+'</urlset>',encoding='utf-8')
    (OUT/'robots.txt').write_text('User-agent: *\nAllow: /\nSitemap: '+ORIGIN+'/sitemap.xml\n',encoding='utf-8')
    print(json.dumps({'pages':len(paths),'project_pages':len(PROJECTS),'output':str(OUT)}))

if __name__=='__main__':main()
