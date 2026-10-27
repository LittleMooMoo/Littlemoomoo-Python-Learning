# Jiale Jiang engineering portfolio

The live GitHub Pages site is https://littlemoomoo.github.io/Littlemoomoo-Python-Learning/ .

## Edit without writing code

1. In File Explorer, open this repository's **Website** folder and double-click **OPEN_EDITOR.cmd**. A terminal opens, then the browser editor. Keep the terminal open while you work.
2. Choose **Home**, **About & education**, or a project section from the list on the left. Click the text in the white editing area to change it. Use the toolbar for headings, lists, and links.
3. To add a picture, put the cursor where it belongs and expand **Add a photograph, drawing, or original PDF**. Enter a caption and an image description, then choose **Upload & insert image**.
4. For handwritten notes, upload the original PDF first. The editor inserts its link and fills in **Original PDF link (optional)**. Upload the preview image with that link attached so readers can open the full notes.
5. Click **Save & build** and wait for **Saved. Build and site checks passed.**, then click **Preview**. Check the page and its links. If checks fail, open **Build details** and resolve the issue before publishing. Saving changes the files on your computer; publishing comes next.
6. Open **GitHub Desktop** and select **Littlemoomoo-Python-Learning**. Select only the **Website** changes you want to publish, write a short commit message, click **Commit to main**, then **Push origin**. The **Publish engineering portfolio** workflow publishes the update. Unrelated learning-project files can stay unselected.

For a project title, summary, main image, tools, or status, choose the project under **Project settings** and click **Edit project details**. Use **Add a project section** when you need another section on that project's page.

If the browser does not open, copy the local editor URL printed in the terminal. That address works only on your computer while the terminal is open. Share the public GitHub Pages URL with employers and professors.

Close the terminal when you finish editing. The public website stays online. The launcher uses the repository's Python environment, the bundled Python, or system Python; it needs no extra packages.

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

Home source contains a `<!--PROJECT_CARDS-->` marker. The builder uses it to insert the current project summaries. Change those summaries through Project settings.

The résumé page uses the concise selection in `content/pages/resume.html`. Its **Download résumé (PDF)** button serves `public/documents/Jiale_Jiang_Robotics_Resume2027_Concise.pdf`. To update it later, replace that PDF and edit the Résumé page to match, then save, preview, and publish. The builder copies the PDF into the public website without changing its contents.

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
