# Jiale Jiang engineering portfolio

This folder contains the editable source and complete static website, copied
into the Python learning repository for editing and future publishing.
It includes the home page, project index, six project case studies, education,
work experience, contact, a printable résumé, and a 404 page.

The copy is local. Copying these files does not publish a new website or update
the existing public site.

## Files to edit

| File or directory | Purpose |
| --- | --- |
| `build_portfolio.py` | Text, project information, page templates, and website URLs. |
| `dist/assets/site.css` | Typography, colors, spacing, and responsive layout. |
| `dist/assets/site.js` | Navigation, project filters, image viewer, and other interactions. |
| `dist/assets/` | Images, favicon, CSS, and JavaScript required by the website. |
| `asset-dimensions.json` | Image dimensions used to reserve space while images load. |
| `tools/validate_site.py` | Checks local links, anchors, images, and page structure. |
| `.openai/hosting.json` | Optional metadata identifying the existing Sites deployment. |

Edit page content in the Python builder. Rebuilding replaces generated HTML,
the sitemap, and robots file. Keep the assets folder: the builder does not
regenerate images, CSS, or JavaScript.

## Rebuild and check

From this repository's root, open PowerShell and run:

```powershell
Set-Location Website
python build_portfolio.py
python tools/validate_site.py
```

The builder and validator use only Python's standard library. If `python` is
not available on your PATH, the learning repository's interpreter can be used
from inside this folder instead:

```powershell
..\.venv\Scripts\python.exe build_portfolio.py
..\.venv\Scripts\python.exe tools\validate_site.py
```

## Preview locally

From inside this folder, run:

```powershell
python -m http.server 4173 --directory dist
```

Open http://localhost:4173 and keep the terminal running while you preview.
Press Ctrl+C in that terminal to stop the server. This address serves the site
on your own computer; use the eventual published URL when sharing it.

## Add images

Save a new image as `.webp` in `dist/assets/`, then reference its filename
without the extension inside the appropriate page's HTML string:

```python
{figure('actuator-test', 'Bench testing the revised actuator.',
        'Photo of the actuator mounted on the test fixture.', photo=True)}
```

Add its pixel width and height to `asset-dimensions.json`, using the existing
entries as a guide. Rebuild and preview the affected page.

## Publish later

The directory to publish is `Website/dist` relative to the repository root.
Configure a static host to serve that directory as the website root, or upload
its contents. Python is used to generate the pages; the published site runs
as ordinary HTML, CSS, JavaScript, and images.

Before using a new domain, update `ORIGIN` near the top of `build_portfolio.py`
and rebuild. That setting controls canonical links, social-preview URLs, the
sitemap, robots metadata, and the portfolio URL printed on the résumé.

Navigation and asset URLs currently begin with `/` and expect the website at
the domain root. For a GitHub Pages project URL such as
`https://username.github.io/Littlemoomoo-Python-Learning/`, those URLs also need
the repository prefix. Changing `ORIGIN` alone does not make subdirectory
hosting work. A root-domain Pages site or a custom domain avoids that prefix.

The `.openai/hosting.json` file contains only the original Sites project ID
and static-directory setting; it contains no credentials. It can be retained
as deployment metadata and is not required by other static hosts. Only `dist`
needs to be served publicly.

Original attachments, private research notes, screenshots of browser checks,
and the original site's Git metadata are excluded from this folder. Project
copy distinguishes built hardware, CAD studies, simulation checks, and work
still in development.
