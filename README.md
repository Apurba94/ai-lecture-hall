# The AI Lecture Hall

© Janin A Apurba (CSE, AUST) — Advanced ICT Officer, CNRS-UNHCR. All rights reserved.

A static website with **273 lectures** on Artificial Intelligence, Machine Learning and Deep Learning,
organised into 10 learning tracks.

## Live site and automatic publishing

- Live website: https://ai-lecture-hall.vercel.app
- Source code: GitHub repository `ai-lecture-hall`
- Vercel is connected to the GitHub repository: **every push to the `main` branch rebuilds and redeploys the site automatically.**

To add or edit a lecture:

1. Edit or add a file in `content/` (see "Add or edit lectures" below).
2. Optionally preview locally with `node build.js` then `node serve.js 8080`.
3. Double-click **`publish.cmd`** (or run `publish "Describe your change"`).
   It checks the build, commits, and pushes to GitHub; Vercel publishes within about a minute.

You can also edit a lecture directly on github.com — committing there triggers the same automatic deploy.

## Upload the website manually (other hosts)

The ready-to-upload website is the **`site/`** folder. Upload its *contents* to any static host:

- **cPanel / shared hosting**: upload everything inside `site/` into `public_html/`.
- **GitHub Pages**: push the contents of `site/` to a repository and enable Pages.
- **Netlify / Cloudflare Pages / Vercel**: drag and drop the `site/` folder.

No database or server-side code is needed. Pages load KaTeX (maths), highlight.js (code) and Google
Fonts from public CDNs.

## Preview locally

```bash
node serve.js 8080
```

Then open http://localhost:8080

## Customise

Edit the `SITE` block at the top of `build.js` (site name, tagline, author details). Set `baseUrl`
to your domain (for example `https://ai.example.com`) so the sitemap and canonical links are correct.
Colours and layout live in `src/style.css`; interactive behaviour in `src/app.js`.

Rebuild after any change:

```bash
node build.js
```

## Add or edit lectures

Lectures live in `content/*.md`. Each file holds several lectures separated by `=== POST ===`:

```text
=== POST ===
slug: my-new-lecture
title: My New Lecture
category: deep-learning          # one of the 10 track slugs defined in build.js
level: Intermediate              # Beginner | Intermediate | Advanced
tags: tag one, tag two
summary: One or two sentences shown on cards and in search results.
---
Lecture body in Markdown. Supports ## headings, lists, tables, ```code```, $inline$ and $$display$$ maths,
and callouts:

:::note
A professor's note.
:::
```

Callout types: `note`, `tip`, `warning`, `exercise`, `takeaway`, `example`, `definition`.
Lectures are numbered in track order, then file order.

## Project layout

```text
build.js        static site generator (Node.js, no dependencies)
serve.js        local preview server
content/        lecture sources (Markdown)
src/            stylesheet, script and favicon (copied to site/assets)
site/           generated website — upload this
```
