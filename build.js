#!/usr/bin/env node
/*
 * The AI Lecture Hall — static site builder
 * © Janin A Apurba (CSE, AUST) — Advanced ICT Officer, CNRS-UNHCR. All rights reserved.
 *
 * Usage:  node build.js
 * Reads every content/*.md file (each holds several lectures separated by "=== POST ===")
 * and writes a complete, upload-ready website into ./site
 */
'use strict';
const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------- configuration
const SITE = {
  name: 'The AI Lecture Hall',
  tagline: 'Machine Learning · Deep Learning · Artificial Intelligence — taught like a PhD classroom',
  author: 'Janin A Apurba',
  authorShort: 'J. A. Apurba',
  credentials: 'CSE, AUST',
  role: 'Advanced ICT Officer, CNRS-UNHCR',
  year: new Date().getFullYear(),
  baseUrl: 'https://ai-lecture-hall.vercel.app', // used for sitemap/canonical links; change if you add a custom domain
};

const CATEGORIES = [
  { slug: 'foundations', name: 'AI Foundations', icon: '🧠', c1: '#7c3aed', c2: '#2563eb', desc: 'Search, logic, knowledge, uncertainty and the ideas that started the field.' },
  { slug: 'math', name: 'Mathematics for ML', icon: '∑', c1: '#0891b2', c2: '#4f46e5', desc: 'Linear algebra, calculus, probability, statistics, information theory and optimisation.' },
  { slug: 'machine-learning', name: 'Machine Learning', icon: '📈', c1: '#059669', c2: '#0284c7', desc: 'Regression, classification, trees, ensembles, clustering, evaluation and learning theory.' },
  { slug: 'deep-learning', name: 'Deep Learning', icon: '🔗', c1: '#db2777', c2: '#7c3aed', desc: 'Neural networks, backpropagation, optimisers, normalisation, CNNs, RNNs and training at scale.' },
  { slug: 'computer-vision', name: 'Computer Vision', icon: '👁️', c1: '#ea580c', c2: '#db2777', desc: 'From pixels to perception: classification, detection, segmentation, ViTs and 3D vision.' },
  { slug: 'nlp', name: 'NLP & Transformers', icon: '💬', c1: '#2563eb', c2: '#06b6d4', desc: 'Language models, embeddings, attention, BERT, GPT, speech and multilingual NLP.' },
  { slug: 'generative-ai', name: 'Generative AI & LLMs', icon: '✨', c1: '#9333ea', c2: '#f43f5e', desc: 'VAEs, GANs, diffusion, large language models, RAG, fine-tuning and AI agents.' },
  { slug: 'reinforcement-learning', name: 'Reinforcement Learning', icon: '🎮', c1: '#16a34a', c2: '#ca8a04', desc: 'MDPs, dynamic programming, Q-learning, policy gradients, PPO and AlphaZero.' },
  { slug: 'mlops', name: 'MLOps & Engineering', icon: '⚙️', c1: '#475569', c2: '#0ea5e9', desc: 'Pipelines, reproducibility, serving, monitoring and shipping ML systems to production.' },
  { slug: 'ethics', name: 'AI Ethics, Society & Careers', icon: '⚖️', c1: '#b45309', c2: '#be123c', desc: 'Fairness, explainability, privacy, safety, regulation, humanitarian AI and your career.' },
];
const CAT = Object.fromEntries(CATEGORIES.map(c => [c.slug, c]));

const ROOT = __dirname;
const CONTENT = path.join(ROOT, 'content');
const OUT = path.join(ROOT, 'site');
const SRC = path.join(ROOT, 'src');

// ---------------------------------------------------------------- helpers
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const slugify = s => String(s).toLowerCase().replace(/<[^>]+>/g, '').replace(/&[a-z]+;/g, '').replace(/[^a-z0-9\s-]/g, '').trim().replace(/\s+/g, '-').replace(/-+/g, '-');
const hash = s => { let h = 2166136261; for (const ch of s) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; };
const pad = n => String(n).padStart(3, '0');
function writeFile(rel, data) {
  const p = path.join(OUT, rel);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, data);
}

// ---------------------------------------------------------------- markdown
// A compact Markdown dialect: headings, paragraphs, lists, fenced code, tables,
// blockquotes, hr, $$display math$$, inline $math$, and :::callout blocks.
function inline(text) {
  const stash = [];
  const keep = html => `\u0000${stash.push(html) - 1}\u0000`;
  text = text.replace(/`([^`]+)`/g, (_, c) => keep(`<code>${esc(c)}</code>`));
  text = text.replace(/\\\((.+?)\\\)/g, (m) => keep(esc(m)));
  text = text.replace(/\$([^$\n]+?)\$/g, (m) => keep(esc(m)));
  text = esc(text);
  text = text.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, a, u) => keep(`<img src="${u}" alt="${a}" loading="lazy">`));
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) => {
    const ext = /^https?:/.test(u);
    return `<a href="${u}"${ext ? ' target="_blank" rel="noopener"' : ''}>${t}</a>`;
  });
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/(^|[^\w*])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>');
  text = text.replace(/(^|[^\w])_([^_\n]+)_(?!\w)/g, '$1<em>$2</em>');
  return text.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[+i]);
}

const CALLOUTS = {
  note: { icon: '🎓', label: "Professor's Note" },
  tip: { icon: '💡', label: 'Tip' },
  warning: { icon: '⚠️', label: 'Common Pitfall' },
  exercise: { icon: '📝', label: 'Exercise' },
  takeaway: { icon: '🎯', label: 'Key Takeaways' },
  example: { icon: '🔍', label: 'Worked Example' },
  definition: { icon: '📘', label: 'Definition' },
};

function markdown(src, toc) {
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let i = 0;
  const isBlockStart = l => /^(#{1,6}\s|```|\$\$|:::|>\s?|\s*[-*+]\s|\s*\d+[.)]\s|\|.*\||---+\s*$|\*\*\*+\s*$)/.test(l);
  while (i < lines.length) {
    let line = lines[i];
    if (!line.trim()) { i++; continue; }

    let m;
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      const level = m[1].length; const txt = m[2].trim();
      let id = slugify(txt) || `section-${i}`;
      while (toc && toc.ids.has(id)) id += '-x';
      if (toc) { toc.ids.add(id); if (level === 2 || level === 3) toc.items.push({ level, id, text: inline(txt).replace(/<[^>]+>/g, '') }); }
      out.push(`<h${level} id="${id}">${inline(txt)}<a class="anchor" href="#${id}" aria-label="Link to section">#</a></h${level}>`);
      i++; continue;
    }
    if ((m = line.match(/^```\s*([\w+-]*)/))) {
      const lang = m[1] || 'plaintext'; const buf = []; i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push(`<div class="code-wrap"><div class="code-head"><span class="dots"><i></i><i></i><i></i></span><span class="lang">${esc(lang)}</span><button class="copy-btn" type="button">Copy</button></div><pre><code class="language-${esc(lang)}">${esc(buf.join('\n'))}</code></pre></div>`);
      continue;
    }
    if (/^\$\$/.test(line.trim())) {
      const t = line.trim();
      if (t.length > 4 && t.endsWith('$$')) { out.push(`<div class="math-block">${esc(t)}</div>`); i++; continue; }
      const buf = [t]; i++;
      while (i < lines.length && !lines[i].trim().endsWith('$$')) buf.push(lines[i++]);
      if (i < lines.length) buf.push(lines[i++].trim());
      out.push(`<div class="math-block">${esc(buf.join('\n'))}</div>`);
      continue;
    }
    if ((m = line.match(/^:::(\w+)\s*(.*)$/))) {
      const type = m[1]; const meta = CALLOUTS[type] || CALLOUTS.note; const title = m[2].trim() || meta.label;
      const buf = []; i++;
      while (i < lines.length && !/^:::\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push(`<aside class="callout callout-${esc(type)}"><div class="callout-title"><span>${meta.icon}</span>${inline(title)}</div><div class="callout-body">${markdown(buf.join('\n'))}</div></aside>`);
      continue;
    }
    if (/^>\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ''));
      out.push(`<blockquote>${markdown(buf.join('\n'))}</blockquote>`);
      continue;
    }
    if (/^(---+|\*\*\*+)\s*$/.test(line)) { out.push('<hr>'); i++; continue; }
    if (/^\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(lines[i + 1])) {
      const split = l => l.trim().replace(/^\|/, '').replace(/\|$/, '').split(/(?<!\\)\|/).map(c => c.trim());
      const head = split(line); const aligns = split(lines[i + 1]).map(c => c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : '');
      i += 2; const rows = [];
      while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) rows.push(split(lines[i++]));
      const al = k => aligns[k] ? ` style="text-align:${aligns[k]}"` : '';
      out.push(`<div class="table-wrap"><table><thead><tr>${head.map((h, k) => `<th${al(k)}>${inline(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c, k) => `<td${al(k)}>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
      continue;
    }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const items = [];
      while (i < lines.length) {
        const l = lines[i];
        if (/^\s*([-*+]|\d+[.)])\s+/.test(l) && !/^\s{2,}/.test(l)) { items.push([l.replace(/^\s*([-*+]|\d+[.)])\s+/, '')]); i++; }
        else if (/^\s{2,}\S/.test(l) && items.length) { items[items.length - 1].push(l.replace(/^\s{2,4}/, '')); i++; }
        else if (!l.trim() && i + 1 < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i + 1]) && ordered === /^\s*\d+[.)]\s+/.test(lines[i + 1])) { i++; }
        else break;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map(it => {
        const [first, ...rest] = it;
        const nested = rest.length ? markdown(rest.join('\n')) : '';
        return `<li>${inline(first)}${nested}</li>`;
      }).join('')}</${tag}>`);
      continue;
    }
    const buf = [];
    while (i < lines.length && lines[i].trim() && !(buf.length && isBlockStart(lines[i]))) buf.push(lines[i++]);
    out.push(`<p>${inline(buf.join(' '))}</p>`);
  }
  return out.join('\n');
}

// ---------------------------------------------------------------- load content
function loadPosts() {
  const files = fs.readdirSync(CONTENT).filter(f => f.endsWith('.md')).sort();
  const posts = [];
  const seen = new Set();
  for (const f of files) {
    const raw = fs.readFileSync(path.join(CONTENT, f), 'utf8').replace(/\r\n/g, '\n');
    const chunks = raw.split(/^=== POST ===\s*$/m).map(s => s.trim()).filter(Boolean);
    for (const chunk of chunks) {
      const sep = chunk.indexOf('\n---\n');
      if (sep < 0) throw new Error(`Missing front-matter separator in ${f}: ${chunk.slice(0, 80)}`);
      const meta = {};
      for (const l of chunk.slice(0, sep).split('\n')) {
        const k = l.indexOf(':'); if (k > 0) meta[l.slice(0, k).trim()] = l.slice(k + 1).trim().replace(/^"(.*)"$/, '$1');
      }
      const body = chunk.slice(sep + 5);
      if (!meta.slug || !meta.title || !meta.category) throw new Error(`Bad front matter in ${f}: ${JSON.stringify(meta)}`);
      if (!CAT[meta.category]) throw new Error(`Unknown category "${meta.category}" in ${f}`);
      if (seen.has(meta.slug)) throw new Error(`Duplicate slug ${meta.slug} in ${f}`);
      seen.add(meta.slug);
      const toc = { items: [], ids: new Set() };
      const html = markdown(body, toc);
      const words = body.split(/\s+/).filter(Boolean).length;
      posts.push({
        ...meta,
        tags: (meta.tags || '').split(',').map(t => t.trim()).filter(Boolean),
        level: meta.level || 'Intermediate',
        html, toc: toc.items, words,
        minutes: Math.max(3, Math.round(words / 170)),     // technical reading pace
        file: f,
      });
    }
  }
  // order: by category order, then file order
  const catIndex = Object.fromEntries(CATEGORIES.map((c, k) => [c.slug, k]));
  posts.forEach((p, k) => (p._k = k));
  posts.sort((a, b) => catIndex[a.category] - catIndex[b.category] || a._k - b._k);
  posts.forEach((p, k) => (p.n = k + 1));
  for (const c of CATEGORIES) {
    const list = posts.filter(p => p.category === c.slug);
    list.forEach((p, k) => { p.inCat = k + 1; p.prev = list[k - 1]; p.next = list[k + 1]; });
    c.posts = list;
  }
  return posts;
}

// ---------------------------------------------------------------- generative cover art
function cover(p, cls = 'cover') {
  const c = CAT[p.category]; const h = hash(p.slug); const v = h % 5;
  const rnd = (() => { let s = h || 1; return () => ((s = Math.imul(s ^ (s >>> 15), 2246822507) ^ Math.imul(s ^ (s >>> 13), 3266489909)) >>> 0) / 4294967296; })();
  const gid = `g${h.toString(36)}`;
  let art = '';
  if (v === 0) { // neural network
    const layers = [3, 5, 5, 2]; const pts = layers.map((n, L) => Array.from({ length: n }, (_, k) => [60 + L * 90, 30 + (k + 0.5) * (150 / n)]));
    for (let L = 0; L < pts.length - 1; L++) for (const a of pts[L]) for (const b of pts[L + 1]) art += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#fff" stroke-opacity="${(0.08 + rnd() * 0.22).toFixed(2)}"/>`;
    for (const layer of pts) for (const a of layer) art += `<circle cx="${a[0]}" cy="${a[1]}" r="7" fill="#fff" fill-opacity="${(0.5 + rnd() * 0.5).toFixed(2)}"/>`;
  } else if (v === 1) { // rings
    const cx = 250 + rnd() * 80, cy = 40 + rnd() * 100;
    for (let r = 20; r < 260; r += 22) art += `<circle cx="${cx.toFixed(0)}" cy="${cy.toFixed(0)}" r="${r}" fill="none" stroke="#fff" stroke-opacity="${(0.35 - r / 900).toFixed(2)}" stroke-width="2"/>`;
  } else if (v === 2) { // waves
    for (let k = 0; k < 9; k++) {
      const a = 8 + rnd() * 22, f = 0.015 + rnd() * 0.02, ph = rnd() * 6, y0 = 20 + k * 20;
      let d = `M0 ${y0}`; for (let x = 0; x <= 400; x += 10) d += ` L${x} ${(y0 + a * Math.sin(f * x + ph)).toFixed(1)}`;
      art += `<path d="${d}" fill="none" stroke="#fff" stroke-opacity="${(0.12 + k * 0.03).toFixed(2)}" stroke-width="2"/>`;
    }
  } else if (v === 3) { // dot matrix
    for (let x = 20; x < 400; x += 24) for (let y = 16; y < 200; y += 24) { const on = rnd() > 0.72; art += `<circle cx="${x}" cy="${y}" r="${on ? 5 : 2.5}" fill="#fff" fill-opacity="${on ? 0.85 : 0.2}"/>`; }
  } else { // bar chart / signal
    for (let x = 16; x < 400; x += 18) { const hh = 20 + rnd() * 130; art += `<rect x="${x}" y="${(190 - hh).toFixed(0)}" width="10" height="${hh.toFixed(0)}" rx="3" fill="#fff" fill-opacity="${(0.15 + rnd() * 0.45).toFixed(2)}"/>`; }
  }
  return `<svg class="${cls}" viewBox="0 0 400 200" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><defs><linearGradient id="${gid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${c.c1}"/><stop offset="1" stop-color="${c.c2}"/></linearGradient></defs><rect width="400" height="200" fill="url(#${gid})"/>${art}<text x="22" y="182" font-family="Space Grotesk, sans-serif" font-weight="700" font-size="15" fill="#fff" fill-opacity=".9">LECTURE ${pad(p.n)}</text></svg>`;
}

// ---------------------------------------------------------------- layout
const FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">';

function layout({ title, desc, rel = '', body, extraHead = '', page = '', canonical = '' }) {
  const nav = [['index.html', 'Home', 'home'], ['tracks.html', 'Learning Tracks', 'tracks'], ['index.html#lectures', 'All Lectures', 'lectures'], ['about.html', 'About the Author', 'about']];
  return `<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<meta name="description" content="${esc(desc)}">
<meta name="author" content="${esc(SITE.author)}">
<meta name="copyright" content="© ${SITE.year} ${esc(SITE.author)}. All rights reserved.">
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(desc)}">
<meta property="og:type" content="${page === 'post' ? 'article' : 'website'}">
${canonical && SITE.baseUrl ? `<link rel="canonical" href="${SITE.baseUrl}/${canonical}">` : ''}
<meta name="theme-color" content="#070b1a">
<link rel="icon" href="${rel}assets/favicon.svg" type="image/svg+xml">
${FONTS}
<link rel="stylesheet" href="${rel}assets/style.css">
<script>try{var t=localStorage.getItem('alh-theme');if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}</script>
${extraHead}
</head>
<body class="page-${page}">
<div class="bg-scene" aria-hidden="true"><span class="blob b1"></span><span class="blob b2"></span><span class="blob b3"></span><span class="grid-overlay"></span></div>
<div class="progress" id="progress"></div>
<header class="topbar">
  <div class="container nav-inner">
    <a class="brand" href="${rel}index.html"><span class="brand-mark"><svg viewBox="0 0 32 32" width="30" height="30"><defs><linearGradient id="bm" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#a78bfa"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs><rect width="32" height="32" rx="9" fill="url(#bm)"/><g stroke="#fff" stroke-width="1.6" opacity=".9"><line x1="9" y1="10" x2="16" y2="16"/><line x1="9" y1="22" x2="16" y2="16"/><line x1="16" y1="16" x2="23" y2="10"/><line x1="16" y1="16" x2="23" y2="22"/></g><g fill="#fff"><circle cx="9" cy="10" r="2.6"/><circle cx="9" cy="22" r="2.6"/><circle cx="16" cy="16" r="3.2"/><circle cx="23" cy="10" r="2.6"/><circle cx="23" cy="22" r="2.6"/></g></svg></span><span class="brand-text">${esc(SITE.name)}<small>by ${esc(SITE.author)}</small></span></a>
    <nav class="nav-links" id="navLinks">${nav.map(([h, t, k]) => `<a href="${rel}${h}"${k === page ? ' class="active"' : ''}>${t}</a>`).join('')}</nav>
    <div class="nav-actions">
      <button class="icon-btn" id="themeToggle" type="button" aria-label="Toggle colour theme"><svg class="i-sun" viewBox="0 0 24 24" width="20" height="20"><circle cx="12" cy="12" r="4.5" fill="currentColor"/><g stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></g></svg><svg class="i-moon" viewBox="0 0 24 24" width="20" height="20"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z" fill="currentColor"/></svg></button>
      <button class="icon-btn menu-btn" id="menuBtn" type="button" aria-label="Open menu"><svg viewBox="0 0 24 24" width="22" height="22"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button>
    </div>
  </div>
</header>
<main id="main">
${body}
</main>
<footer class="footer">
  <div class="container footer-grid">
    <div>
      <a class="brand" href="${rel}index.html"><span class="brand-text">${esc(SITE.name)}<small>by ${esc(SITE.author)}</small></span></a>
      <p class="muted">A free, structured university-level curriculum in Artificial Intelligence, Machine Learning and Deep Learning — written as lectures for students everywhere.</p>
    </div>
    <div>
      <h4>Learning Tracks</h4>
      <ul>${CATEGORIES.map(c => `<li><a href="${rel}track/${c.slug}.html">${c.icon} ${esc(c.name)}</a></li>`).join('')}</ul>
    </div>
    <div>
      <h4>The Author</h4>
      <p><strong>${esc(SITE.author)}</strong><br>B.Sc. in ${esc(SITE.credentials)}<br>${esc(SITE.role)}</p>
      <p><a href="${rel}about.html">About the author →</a></p>
    </div>
  </div>
  <div class="container copyright">
    <p>© ${SITE.year} <strong>${esc(SITE.author)}</strong>, ${esc(SITE.credentials)} · ${esc(SITE.role)}. All rights reserved.</p>
    <p class="muted">All lectures, text, code examples and site design are the intellectual property of ${esc(SITE.author)}. Reproduction or redistribution without written permission is prohibited.</p>
  </div>
</footer>
<button class="to-top" id="toTop" type="button" aria-label="Back to top">↑</button>
<script src="${rel}assets/app.js" defer></script>
</body>
</html>`;
}

function card(p, rel) {
  const c = CAT[p.category];
  return `<article class="card" data-cat="${p.category}" data-level="${esc(p.level)}" data-n="${p.n}" style="--c1:${c.c1};--c2:${c.c2}">
  <a class="card-link" href="${rel}posts/${p.slug}.html" aria-label="${esc(p.title)}"></a>
  <div class="card-cover">${cover(p)}<span class="card-cat">${c.icon} ${esc(c.name)}</span></div>
  <div class="card-body">
    <h3>${esc(p.title)}</h3>
    <p>${esc(p.summary || '')}</p>
    <div class="card-meta"><span class="lvl lvl-${esc(p.level.toLowerCase())}">${esc(p.level)}</span><span>⏱ ${p.minutes} min</span><span>#${pad(p.n)}</span></div>
  </div>
</article>`;
}

// ---------------------------------------------------------------- pages
function buildIndex(posts) {
  const totalMin = posts.reduce((a, p) => a + p.minutes, 0);
  const featuredSlugs = ['what-is-artificial-intelligence', 'backpropagation-derived', 'transformer-architecture-explained', 'diffusion-models', 'q-learning-and-sarsa', 'retrieval-augmented-generation'];
  const featured = featuredSlugs.map(s => posts.find(p => p.slug === s)).filter(Boolean);
  while (featured.length < 6 && featured.length < posts.length) { const p = posts[(featured.length * 37) % posts.length]; if (!featured.includes(p)) featured.push(p); else break; }
  const index = posts.map(p => ({ s: p.slug, t: p.title, c: p.category, l: p.level, g: p.tags.join(' '), d: p.summary || '' }));
  const body = `
<section class="hero">
  <canvas id="heroCanvas" aria-hidden="true"></canvas>
  <div class="container hero-inner">
    <span class="eyebrow"><span class="pulse"></span> ${posts.length} free lectures · ${CATEGORIES.length} learning tracks</span>
    <h1>Learn <span class="grad">Artificial Intelligence</span><br>the way it is taught in a <span class="grad-2">PhD classroom</span></h1>
    <p class="lead">Rigorous, intuitive lectures on Machine Learning, Deep Learning and AI — from the mathematics of gradient descent to transformers, diffusion models and reinforcement learning. Written by <strong>${esc(SITE.author)}</strong>.</p>
    <div class="hero-cta">
      <a class="btn btn-primary" href="posts/${posts[0].slug}.html">Start Lecture 001 →</a>
      <a class="btn btn-ghost" href="tracks.html">Explore the curriculum</a>
    </div>
    <div class="stats">
      <div><b data-count="${posts.length}">${posts.length}</b><span>Lectures</span></div>
      <div><b data-count="${CATEGORIES.length}">${CATEGORIES.length}</b><span>Tracks</span></div>
      <div><b data-count="${Math.round(totalMin / 60)}">${Math.round(totalMin / 60)}</b><span>Hours of reading</span></div>
      <div><b data-count="${posts.filter(p => /```/.test(p.html) || p.html.includes('code-wrap')).length}">${posts.filter(p => p.html.includes('code-wrap')).length}</b><span>With code</span></div>
    </div>
  </div>
</section>

<section class="section container">
  <div class="section-head"><div><span class="kicker">Curriculum</span><h2>Ten tracks, one coherent journey</h2></div><a class="more" href="tracks.html">See full roadmap →</a></div>
  <div class="tracks-grid">
    ${CATEGORIES.map((c, k) => `<a class="track" href="track/${c.slug}.html" style="--c1:${c.c1};--c2:${c.c2}"><span class="track-num">${String(k + 1).padStart(2, '0')}</span><span class="track-icon">${c.icon}</span><h3>${esc(c.name)}</h3><p>${esc(c.desc)}</p><span class="track-count">${c.posts.length} lectures →</span></a>`).join('')}
  </div>
</section>

<section class="section container">
  <div class="section-head"><div><span class="kicker">Editor's picks</span><h2>Featured lectures</h2></div></div>
  <div class="cards featured">${featured.map(p => card(p, '')).join('')}</div>
</section>

<section class="section container" id="lectures">
  <div class="section-head"><div><span class="kicker">Library</span><h2>All lectures</h2></div><span class="muted" id="resultCount">${posts.length} lectures</span></div>
  <div class="filters">
    <div class="search"><svg viewBox="0 0 24 24" width="18" height="18"><circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2"/><path d="M20 20l-4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><input id="q" type="search" placeholder="Search lectures, e.g. “attention”, “SVM”, “Bayes”…" autocomplete="off"></div>
    <select id="level" aria-label="Filter by level"><option value="">All levels</option><option>Beginner</option><option>Intermediate</option><option>Advanced</option></select>
  </div>
  <div class="chips" id="chips"><button class="chip active" data-cat="">All</button>${CATEGORIES.map(c => `<button class="chip" data-cat="${c.slug}" style="--c1:${c.c1};--c2:${c.c2}">${c.icon} ${esc(c.name)}</button>`).join('')}</div>
  <div class="cards" id="grid">${posts.map(p => card(p, '')).join('')}</div>
  <div class="empty" id="empty" hidden>No lectures match your search. Try another keyword.</div>
  <div class="load-more-wrap"><button class="btn btn-ghost" id="loadMore" type="button">Load more lectures</button></div>
</section>

<section class="section container">
  <div class="author-band">
    <div class="avatar big">JA</div>
    <div>
      <span class="kicker">Your instructor</span>
      <h2>${esc(SITE.author)}</h2>
      <p class="muted">B.Sc. in Computer Science &amp; Engineering, Ahsanullah University of Science and Technology (AUST) · ${esc(SITE.role)}</p>
      <p>“I wrote these lectures the way I would deliver them in a classroom: starting from intuition, building the mathematics carefully, and ending with code you can run. My goal is simple — to put a complete, rigorous AI education within reach of every student.”</p>
      <a class="btn btn-primary" href="about.html">Meet the author</a>
    </div>
  </div>
</section>
<script id="search-data" type="application/json">${JSON.stringify(index).replace(/</g, '\\u003c')}</script>`;
  writeFile('index.html', layout({ title: `${SITE.name} — ML, Deep Learning & AI Lectures by ${SITE.author}`, desc: SITE.tagline, body, page: 'home', canonical: 'index.html' }));
}

// Animated companion lessons from AI in Motion (data/animated-lessons.json, generated by that site's build)
const ANIM = fs.existsSync(path.join(ROOT, 'data', 'animated-lessons.json')) ? JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'animated-lessons.json'), 'utf8')) : {};
function animBox(slug) {
  const list = ANIM[slug]; if (!list || !list.length) return '';
  return `<aside class="anim-box"><div class="anim-ic" aria-hidden="true">▶</div><div><div class="anim-kicker">Watch it animated · AI in Motion</div>${list.map(v => `<a href="${esc(v.url)}" target="_blank" rel="noopener">${esc(v.title)} <span>· ${v.minutes} min video ↗</span></a>`).join('')}<small>A short narrated animation of the key ideas — watch it first, then read the full lecture.</small></div></aside>`;
}

function buildPost(p, posts) {
  const c = CAT[p.category]; const rel = '../';
  const related = posts.filter(q => q.category === p.category && q !== p)
    .map(q => ({ q, s: q.tags.filter(t => p.tags.includes(t)).length * 3 - Math.abs(q.n - p.n) / 10 }))
    .sort((a, b) => b.s - a.s).slice(0, 3).map(x => x.q);
  const tocHtml = p.toc.length ? `<nav class="toc" id="toc"><div class="toc-title">On this page</div><ol>${p.toc.map(t => `<li class="lvl${t.level}"><a href="#${t.id}">${esc(t.text)}</a></li>`).join('')}</ol></nav>` : '';
  const ld = { '@context': 'https://schema.org', '@type': 'Article', headline: p.title, description: p.summary, author: { '@type': 'Person', name: SITE.author, jobTitle: SITE.role }, copyrightHolder: { '@type': 'Person', name: SITE.author }, copyrightYear: SITE.year, keywords: p.tags.join(', '), articleSection: c.name };
  const extraHead = `
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script type="application/ld+json">${JSON.stringify(ld).replace(/</g, '\\u003c')}</script>`;
  const body = `
<header class="post-hero" style="--c1:${c.c1};--c2:${c.c2}">
  ${cover(p, 'post-hero-art')}
  <div class="container post-hero-inner">
    <nav class="crumbs"><a href="${rel}index.html">Home</a><span>/</span><a href="${rel}track/${c.slug}.html">${esc(c.name)}</a><span>/</span><span>Lecture ${pad(p.n)}</span></nav>
    <span class="pill">${c.icon} ${esc(c.name)} · Lecture ${p.inCat} of ${c.posts.length}</span>
    <h1>${esc(p.title)}</h1>
    <p class="lead">${esc(p.summary || '')}</p>
    <div class="post-meta">
      <span class="avatar">JA</span>
      <span><strong>${esc(SITE.author)}</strong><br><small>${esc(SITE.role)}</small></span>
      <span class="sep"></span>
      <span class="lvl lvl-${esc(p.level.toLowerCase())}">${esc(p.level)}</span>
      <span>⏱ ${p.minutes} min read</span>
    </div>
  </div>
</header>
<div class="container post-layout">
  <article class="prose" id="article">
    ${animBox(p.slug)}
    ${p.html}
    <div class="post-tags">${p.tags.map(t => `<span>#${esc(t)}</span>`).join('')}</div>
    <div class="copyright-note">© ${SITE.year} ${esc(SITE.author)} (${esc(SITE.credentials)}), ${esc(SITE.role)}. All rights reserved. This lecture may not be republished without the author's written permission.</div>
    <div class="author-box">
      <div class="avatar big">JA</div>
      <div><span class="kicker">Written by</span><h3>${esc(SITE.author)}</h3><p class="muted">B.Sc. in ${esc(SITE.credentials)} · ${esc(SITE.role)}. Teaching AI, ML and Deep Learning to the next generation of engineers and researchers.</p></div>
    </div>
    <nav class="pager">
      ${p.prev ? `<a class="prev" href="${p.prev.slug}.html"><small>← Previous lecture</small><span>${esc(p.prev.title)}</span></a>` : '<span></span>'}
      ${p.next ? `<a class="next" href="${p.next.slug}.html"><small>Next lecture →</small><span>${esc(p.next.title)}</span></a>` : `<a class="next" href="${rel}tracks.html"><small>Track complete 🎉</small><span>Choose your next track</span></a>`}
    </nav>
  </article>
  <aside class="sidebar"><div class="sidebar-inner">
    ${tocHtml}
    <div class="side-card" style="--c1:${c.c1};--c2:${c.c2}">
      <div class="side-title">${c.icon} ${esc(c.name)}</div>
      <div class="side-progress"><span style="width:${Math.round(p.inCat / c.posts.length * 100)}%"></span></div>
      <small class="muted">Lecture ${p.inCat} of ${c.posts.length} in this track</small>
      <a href="${rel}track/${c.slug}.html">View syllabus →</a>
    </div>
  </div></aside>
</div>
<section class="container section">
  <div class="section-head"><div><span class="kicker">Keep learning</span><h2>Related lectures</h2></div></div>
  <div class="cards">${related.map(q => card(q, rel)).join('')}</div>
</section>`;
  writeFile(`posts/${p.slug}.html`, layout({ title: `${p.title} | ${SITE.name}`, desc: p.summary || p.title, rel, body, extraHead, page: 'post', canonical: `posts/${p.slug}.html` }));
}

function buildTrack(c) {
  const rel = '../';
  const mins = c.posts.reduce((a, p) => a + p.minutes, 0);
  const body = `
<header class="post-hero track-hero" style="--c1:${c.c1};--c2:${c.c2}">
  <div class="container post-hero-inner">
    <nav class="crumbs"><a href="${rel}index.html">Home</a><span>/</span><a href="${rel}tracks.html">Tracks</a><span>/</span><span>${esc(c.name)}</span></nav>
    <span class="track-icon xl">${c.icon}</span>
    <h1>${esc(c.name)}</h1>
    <p class="lead">${esc(c.desc)}</p>
    <div class="post-meta"><span>📚 ${c.posts.length} lectures</span><span>⏱ ~${Math.round(mins / 60 * 10) / 10} hours</span></div>
  </div>
</header>
<section class="container section">
  <ol class="syllabus">
    ${c.posts.map(p => `<li style="--c1:${c.c1};--c2:${c.c2}"><a href="${rel}posts/${p.slug}.html"><span class="syl-num">${String(p.inCat).padStart(2, '0')}</span><span class="syl-body"><strong>${esc(p.title)}</strong><small>${esc(p.summary || '')}</small></span><span class="syl-meta"><span class="lvl lvl-${esc(p.level.toLowerCase())}">${esc(p.level)}</span><small>${p.minutes} min</small></span></a></li>`).join('')}
  </ol>
</section>`;
  writeFile(`track/${c.slug}.html`, layout({ title: `${c.name} — Learning Track | ${SITE.name}`, desc: c.desc, rel, body, page: 'tracks', canonical: `track/${c.slug}.html` }));
}

function buildTracks() {
  const body = `
<header class="page-hero container">
  <span class="kicker">The roadmap</span>
  <h1>A complete AI curriculum, <span class="grad">in the right order</span></h1>
  <p class="lead">Follow the tracks from top to bottom for a full degree-style journey, or jump straight to the area you need. Each track builds on the mathematics and ideas of the ones before it.</p>
</header>
<section class="container section roadmap">
  ${CATEGORIES.map((c, k) => `<div class="road-step" style="--c1:${c.c1};--c2:${c.c2}">
    <div class="road-dot">${String(k + 1).padStart(2, '0')}</div>
    <div class="road-card">
      <div class="road-head"><span class="track-icon">${c.icon}</span><div><h2><a href="track/${c.slug}.html">${esc(c.name)}</a></h2><p class="muted">${esc(c.desc)}</p></div></div>
      <ul class="road-list">${c.posts.map(p => `<li><a href="posts/${p.slug}.html"><span>${String(p.inCat).padStart(2, '0')}</span>${esc(p.title)}</a></li>`).join('')}</ul>
      <a class="more" href="track/${c.slug}.html">Open track syllabus →</a>
    </div>
  </div>`).join('')}
</section>`;
  writeFile('tracks.html', layout({ title: `Learning Tracks & Roadmap | ${SITE.name}`, desc: 'The complete AI, ML and Deep Learning curriculum in recommended order.', body, page: 'tracks', canonical: 'tracks.html' }));
}

function buildAbout(posts) {
  const body = `
<header class="page-hero container about-hero">
  <div class="avatar huge">JA</div>
  <div>
    <span class="kicker">About the author</span>
    <h1>${esc(SITE.author)}</h1>
    <p class="lead">B.Sc. in Computer Science &amp; Engineering — Ahsanullah University of Science and Technology (AUST)<br>${esc(SITE.role)}</p>
  </div>
</header>
<section class="container section prose narrow">
  <h2>Why this lecture hall exists</h2>
  <p>Artificial Intelligence is reshaping science, medicine, humanitarian response and everyday life — yet a rigorous, structured education in the field is still out of reach for many students. <strong>${esc(SITE.name)}</strong> is my answer: ${posts.length} lectures that take a learner from the definition of intelligence all the way to transformers, diffusion models, reinforcement learning and responsible deployment.</p>
  <p>Every lecture follows the rhythm of a good university class. We start from <em>intuition</em>, formalise the idea with <em>mathematics</em>, test it with a <em>worked example</em>, and finish with <em>code</em> and <em>exercises</em>. Professor's notes highlight the subtle points that examiners — and real-world systems — care about.</p>
  <h2>About me</h2>
  <p>I am a Computer Science &amp; Engineering graduate of AUST, currently serving as an <strong>Advanced ICT Officer at CNRS-UNHCR</strong>, where I work at the intersection of technology and people and teach students the digital skills that open doors. I believe technology education is one of the most powerful tools we have for creating opportunity.</p>
  <h2>How to use this site</h2>
  <ul>
    <li><strong>Beginners</strong> should start with <a href="track/foundations.html">AI Foundations</a> and <a href="track/math.html">Mathematics for ML</a>.</li>
    <li><strong>Practitioners</strong> can jump to <a href="track/deep-learning.html">Deep Learning</a>, <a href="track/nlp.html">NLP</a> or <a href="track/generative-ai.html">Generative AI</a>.</li>
    <li><strong>Everyone</strong> should read the <a href="track/ethics.html">Ethics, Society &amp; Careers</a> track — building AI responsibly is part of building it well.</li>
  </ul>
  <h2>Copyright</h2>
  <p>© ${SITE.year} ${esc(SITE.author)}, ${esc(SITE.credentials)} — ${esc(SITE.role)}. All lectures, text, figures, code examples and the design of this website are the intellectual property of the author. All rights reserved. Students are welcome to read, learn from and cite these lectures with attribution; republishing requires written permission.</p>
</section>`;
  writeFile('about.html', layout({ title: `About ${SITE.author} | ${SITE.name}`, desc: `About ${SITE.author}, ${SITE.role}`, body, page: 'about', canonical: 'about.html' }));
}

function build404() {
  const body = `<section class="page-hero container center"><span class="kicker">Error 404</span><h1>This neuron <span class="grad">didn't fire</span></h1><p class="lead">The page you are looking for does not exist. Let's get you back to class.</p><a class="btn btn-primary" href="index.html">Back to the lecture hall</a></section>`;
  writeFile('404.html', layout({ title: `Page not found | ${SITE.name}`, desc: 'Page not found', body, page: '404' }));
}

// ---------------------------------------------------------------- run
function main() {
  const posts = loadPosts();
  fs.rmSync(OUT, { recursive: true, force: true });
  fs.mkdirSync(OUT, { recursive: true });
  fs.mkdirSync(path.join(OUT, 'assets'), { recursive: true });
  for (const f of fs.readdirSync(SRC)) fs.copyFileSync(path.join(SRC, f), path.join(OUT, 'assets', f));
  buildIndex(posts);
  posts.forEach(p => buildPost(p, posts));
  CATEGORIES.forEach(buildTrack);
  buildTracks();
  buildAbout(posts);
  build404();
  const urls = ['index.html', 'tracks.html', 'about.html', ...CATEGORIES.map(c => `track/${c.slug}.html`), ...posts.map(p => `posts/${p.slug}.html`)];
  writeFile('sitemap.xml', `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.map(u => `  <url><loc>${SITE.baseUrl || ''}/${u}</loc></url>`).join('\n')}\n</urlset>\n`);
  writeFile('robots.txt', `User-agent: *\nAllow: /\n${SITE.baseUrl ? `Sitemap: ${SITE.baseUrl}/sitemap.xml\n` : ''}`);
  const words = posts.reduce((a, p) => a + p.words, 0);
  console.log(`Built ${posts.length} lectures (${words.toLocaleString()} words) into ./site`);
  for (const c of CATEGORIES) console.log(`  ${c.name.padEnd(30)} ${c.posts.length}`);
  const short = posts.filter(p => p.words < 450);
  if (short.length) console.log(`  Short lectures (<450 words): ${short.map(p => `${p.slug}(${p.words})`).join(', ')}`);
}
main();
