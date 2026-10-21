# Jiale Jiang engineering portfolio

The live GitHub Pages site is https://littlemoomoo.github.io/Littlemoomoo-Python-Learning/ .

## Edit without changing Python or HTML

1. Double-click **OPEN_EDITOR.cmd** in this folder. Keep its terminal open while editing.
2. The local browser editor opens. Choose a page or project section from the left.
3. Click the text to edit it. Use the toolbar for headings, lists, and links.
4. Expand **Add a photograph, drawing, or original PDF** to choose a file. Add a caption and alternative text. A handwritten preview can include its original PDF link.
5. Click **Save & build**, then **Preview**. This saves the source files and rebuilds the local website.
6. In **GitHub Desktop**, select this repository, review and select the **Website** changes, write a commit message, commit to main, and **Push origin**. Leave unrelated learning-project changes unselected. The **Publish engineering portfolio** action rebuilds and publishes the website.

To change a project title, summary, main image, tools, or status, choose the project under **Project settings**, then **Edit project details**. Use **Add a project section** for a new group of text, equations, or images.

The editor runs only on your computer. Saving locally does not update the public site until you push. Its local preview address is not a sharing URL. Close the terminal to stop the editor. Your public GitHub Pages site keeps working independently.

The launcher uses the repository's Python environment when available, then the bundled Python or system Python. No packages or account subscription are needed. If a browser does not open automatically, copy the local URL printed in the terminal.

## Where your content lives

| File or folder | What you can edit |
| --- | --- |
| `content/projects.json` | Project cards, titles, summaries, metadata, and section order. |
| `content/projects/<project>/*.html` | Each project section, editable visually in the local editor. |
| `content/pages/*.html` | Home, about, experience, contact, resume, and detailed walkthroughs. |
| `content/extra-pages.json` | Paths and titles for the detailed walkthroughs and derivation library. |
| `content/evidence.json` | Handwritten preview-to-original-PDF records and excluded media. |
| `public/assets/notes/` | Full-page handwritten previews. |
| `public/assets/evidence/` | Selected CAD revisions and physical component photographs. |
| `public/assets/uploads/` | Pictures added through the editor. |
| `public/documents/uploads/` | Original PDFs added through the editor. |
| `dist/assets/site.css` and `site.js` | The shared visual style and browser interactions. |
| `build_portfolio.py` | Rebuilds the website from the editable content. |

Do not edit generated `dist/*.html` files: rebuilding replaces them. Original engineering files outside Website are unchanged. Large existing handwritten PDFs are linked from the public Learning material folder in this repository, rather than duplicated or loaded on every page visit.

Home and resume source contain `<!--PROJECT_CARDS-->` and `<!--RESUME_PROJECTS-->` markers. The builder uses these to insert the current project summaries. Change those summaries through Project settings.

## Preview or build from a terminal

From this repository root:

```powershell
.\.venv\Scripts\python.exe Website\tools\editor.py
```

For just a build and structural validation:

```powershell
.\.venv\Scripts\python.exe Website\build_portfolio.py
.\.venv\Scripts\python.exe Website\tools\validate_site.py
```

The editor and builder use Python's standard library. The editor handles image dimensions and file uploads itself.

## Publication

GitHub Pages publishes only `Website/dist`, using `.github/workflows/portfolio-pages.yml`. Pushes to main that change Website trigger it automatically. GitHub Actions supplies the deployment origin and repository path so navigation, image URLs, and canonical links work at the project URL.

To publish to a different static host, set `PORTFOLIO_ORIGIN` to its origin and `PORTFOLIO_BASE_PATH` to its path prefix before building, then publish dist. Normal local previews use an empty base path. The local editor is a development tool and is never served by the static public website.

The older ChatGPT-hosted copy is updated separately. Use GitHub Pages as the ongoing publication workflow when editing this repository yourself.

## Evidence and technical status

The rejected turquoise quadruped image is removed from source content and the public assets. The validator rejects that filename if it is accidentally reintroduced. Each handwritten preview is paired with its original PDF and page number. Team CAD, commercial development boards, reference architectures, exploratory measurements, saved numerical results, and unfinished hardware work are identified in their context.
