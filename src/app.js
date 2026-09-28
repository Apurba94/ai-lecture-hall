/* The AI Lecture Hall — © Janin A Apurba (CSE, AUST), Advanced ICT Officer, CNRS-UNHCR. All rights reserved. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } }
  };

  // ---- theme
  var root = document.documentElement;
  var toggle = $('#themeToggle');
  if (toggle) toggle.addEventListener('click', function () {
    var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', next);
    store.set('alh-theme', next);
  });

  // ---- mobile menu
  var menuBtn = $('#menuBtn'), links = $('#navLinks');
  if (menuBtn && links) {
    menuBtn.addEventListener('click', function () { links.classList.toggle('open'); });
    links.addEventListener('click', function (e) { if (e.target.tagName === 'A') links.classList.remove('open'); });
  }

  // ---- reading progress + back to top
  var bar = $('#progress'), toTop = $('#toTop'), article = $('#article');
  function onScroll() {
    var y = window.scrollY;
    if (bar) {
      var total;
      if (article) {
        var r = article.getBoundingClientRect();
        total = Math.min(1, Math.max(0, -r.top / Math.max(1, r.height - window.innerHeight)));
      } else {
        total = y / Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      }
      bar.style.width = (total * 100).toFixed(2) + '%';
    }
    if (toTop) toTop.classList.toggle('show', y > 700);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  if (toTop) toTop.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: 'smooth' }); });

  // ---- reveal on scroll
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -40px 0px' });
    $$('.track, .featured .card, .road-step, .syllabus li, .author-band').forEach(function (el) { el.classList.add('reveal'); io.observe(el); });
  }

  // ---- hero neural-network canvas
  var cv = $('#heroCanvas');
  if (cv && cv.getContext) {
    var ctx = cv.getContext('2d'), nodes = [], W = 0, H = 0, dpr = Math.min(2, window.devicePixelRatio || 1);
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var resize = function () {
      W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var n = Math.round(Math.min(90, W * H / 14000));
      nodes = [];
      for (var i = 0; i < n; i++) nodes.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35, r: 1.2 + Math.random() * 2.2, h: Math.random() });
    };
    var mouse = { x: -999, y: -999 };
    cv.parentElement.addEventListener('mousemove', function (e) { var r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; });
    cv.parentElement.addEventListener('mouseleave', function () { mouse.x = mouse.y = -999; });
    var draw = function () {
      ctx.clearRect(0, 0, W, H);
      var light = root.getAttribute('data-theme') === 'light';
      for (var i = 0; i < nodes.length; i++) {
        var a = nodes[i];
        if (!reduce) { a.x += a.vx; a.y += a.vy; if (a.x < 0 || a.x > W) a.vx *= -1; if (a.y < 0 || a.y > H) a.vy *= -1; }
        for (var j = i + 1; j < nodes.length; j++) {
          var b = nodes[j], dx = a.x - b.x, dy = a.y - b.y, d = dx * dx + dy * dy;
          if (d < 16000) {
            var al = (1 - d / 16000) * .45;
            ctx.strokeStyle = light ? 'rgba(109,40,217,' + al * .6 + ')' : 'rgba(167,139,250,' + al + ')';
            ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
        var md = Math.hypot(a.x - mouse.x, a.y - mouse.y);
        if (md < 160) {
          ctx.strokeStyle = light ? 'rgba(8,145,178,' + (1 - md / 160) * .6 + ')' : 'rgba(34,211,238,' + (1 - md / 160) * .8 + ')';
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
        }
        ctx.fillStyle = a.h > .66 ? (light ? '#db2777' : '#f472b6') : a.h > .33 ? (light ? '#0891b2' : '#22d3ee') : (light ? '#6d28d9' : '#a78bfa');
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2); ctx.fill();
      }
      if (!reduce) requestAnimationFrame(draw);
    };
    resize(); window.addEventListener('resize', resize); draw();
  }

  // ---- stat counters
  $$('.stats b[data-count]').forEach(function (el) {
    var target = +el.getAttribute('data-count'), t0 = null;
    var step = function (ts) { if (!t0) t0 = ts; var p = Math.min(1, (ts - t0) / 1400); el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))); if (p < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  });

  // ---- library filtering / search / pagination
  var grid = $('#grid');
  if (grid) {
    var cards = $$('.card', grid), data = {};
    try { JSON.parse($('#search-data').textContent).forEach(function (d) { data[d.s] = (d.t + ' ' + d.g + ' ' + d.d + ' ' + d.c).toLowerCase(); }); } catch (e) { /* ignore */ }
    var q = $('#q'), level = $('#level'), chips = $$('#chips .chip'), more = $('#loadMore'), count = $('#resultCount'), empty = $('#empty');
    var PAGE = 24, shown = PAGE, cat = '';
    var slugOf = function (c) { var h = c.querySelector('.card-link').getAttribute('href'); return h.slice(h.lastIndexOf('/') + 1, -5); };
    var apply = function () {
      var terms = (q.value || '').toLowerCase().split(/\s+/).filter(Boolean), lv = level.value, matched = 0;
      cards.forEach(function (c) {
        var text = data[slugOf(c)] || c.textContent.toLowerCase();
        var ok = (!cat || c.getAttribute('data-cat') === cat) && (!lv || c.getAttribute('data-level') === lv) && terms.every(function (t) { return text.indexOf(t) >= 0; });
        if (ok) matched++;
        c.classList.toggle('hide', !ok || matched > shown);
      });
      count.textContent = matched + (matched === 1 ? ' lecture' : ' lectures');
      empty.hidden = matched !== 0;
      more.parentElement.style.display = matched > shown ? '' : 'none';
    };
    q.addEventListener('input', function () { shown = PAGE; apply(); });
    level.addEventListener('change', function () { shown = PAGE; apply(); });
    chips.forEach(function (ch) {
      ch.addEventListener('click', function () {
        chips.forEach(function (x) { x.classList.remove('active'); }); ch.classList.add('active');
        cat = ch.getAttribute('data-cat'); shown = PAGE; apply();
      });
    });
    more.addEventListener('click', function () { shown += PAGE; apply(); });
    var hashCat = (location.hash.match(/cat=([\w-]+)/) || [])[1];
    if (hashCat) chips.forEach(function (ch) { if (ch.getAttribute('data-cat') === hashCat) ch.click(); });
    apply();
  }

  // ---- post page: TOC scrollspy, code copy, math + syntax highlighting
  var toc = $('#toc');
  if (toc && 'IntersectionObserver' in window) {
    var tocLinks = $$('a', toc), map = {};
    tocLinks.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting && map[e.target.id]) { tocLinks.forEach(function (a) { a.classList.remove('active'); }); map[e.target.id].classList.add('active'); }
      });
    }, { rootMargin: '-80px 0px -70% 0px' });
    $$('.prose h2[id], .prose h3[id]').forEach(function (h) { spy.observe(h); });
  }
  $$('.copy-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var code = btn.closest('.code-wrap').querySelector('code').innerText;
      var done = function () { btn.textContent = 'Copied ✓'; setTimeout(function () { btn.textContent = 'Copy'; }, 1600); };
      if (navigator.clipboard) navigator.clipboard.writeText(code).then(done, function () {}); else done();
    });
  });
  window.addEventListener('load', function () {
    if (window.renderMathInElement && article) {
      window.renderMathInElement(article, {
        delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '$', right: '$', display: false }, { left: '\\(', right: '\\)', display: false }],
        throwOnError: false, ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      });
    }
    if (window.hljs) $$('.code-wrap pre code').forEach(function (el) { window.hljs.highlightElement(el); });
  });
})();
