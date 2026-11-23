"use strict";

const $ = (id) => document.getElementById(id);
const state = { mode: "ask", backend: "none", busy: false, context: [] };

const ICON_TRASH = '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>';
const ICON_WARN = '<svg viewBox="0 0 24 24"><path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>';
const TYPE_LABEL = { text: "Text", table: "Table", chart: "Chart" };

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  return body;
}

/* ---------- status & documents ---------- */

async function loadStatus() {
  try {
    const s = await api("/api/status");
    state.backend = s.backend;
    const llm = { claude: "Claude API", bedrock: "Amazon Bedrock" }[s.backend];
    $("status").innerHTML = llm
      ? `<span class="dot"></span><span>${esc(llm)} · ${esc(s.model)}</span><span class="sub">Hybrid retrieval: ${esc(s.dense_model.split("/").pop())} + ${esc(s.sparse_model.split("/").pop())}</span>`
      : `<span class="dot off"></span><span>No Claude credentials: search works, Ask and chart vision are off. Set ANTHROPIC_API_KEY and restart.</span>`;
    const vision = $("vision");
    vision.disabled = !llm;
    if (!llm) vision.checked = false;
    $("vision-toggle").classList.toggle("disabled", !llm);
    if (!llm && state.mode === "ask") setMode("search");
  } catch {
    $("status").innerHTML = '<span class="dot off"></span><span>Cannot reach the finrag server.</span>';
  }
}

async function loadDocs() {
  const docs = await api("/api/documents").catch(() => []);
  $("doc-count").textContent = docs.length;
  $("docs-empty").hidden = docs.length > 0;
  $("docs").innerHTML = docs.map((d) => {
    const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
    const photos = d.figures_not_data ? `, ${d.figures_not_data} photos left out` : "";
    const failed = d.figures_failed ? `, ${d.figures_failed} skipped` : "";
    const figs = d.figures ? ` · ${d.figures_described}/${d.figures} charts described${photos}${failed}` : "";
    return `<li class="doc">
      <div class="info">
        <div class="title">${esc(d.name)}</div>
        <div class="meta">${plural(d.pages, "page")} · ${plural(d.chunks, "chunk")} · ${plural(d.tables, "table")}${figs}</div>
      </div>
      <button class="icon danger" data-delete="${esc(d.name)}" aria-label="Delete ${esc(d.name)}">${ICON_TRASH}</button>
    </li>`;
  }).join("");
}

$("docs").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-delete]");
  if (!btn) return;
  const name = btn.dataset.delete;
  if (!confirm(`Remove ${name} and its chunks from the index?`)) return;
  try {
    await api(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
  } catch (err) {
    alert(err.message);
  }
  loadDocs();
  loadStatus();
});

/* ---------- upload ---------- */

const dz = $("dropzone");
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); } });
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => [...e.dataTransfer.files].forEach(upload));
$("file-input").addEventListener("change", (e) => { [...e.target.files].forEach(upload); e.target.value = ""; });

function jobCard(id, name) {
  const el = document.createElement("div");
  el.className = "job";
  el.id = `job-${id}`;
  el.innerHTML = `<div class="name">${esc(name)}</div><div class="stage"><span class="spinner"></span><span class="txt">Uploading</span></div>`;
  $("jobs").prepend(el);
  return el;
}

function failJob(el, message) {
  el.classList.add("failed");
  el.querySelector(".stage").innerHTML = `<span>${esc(message)}</span>`;
  el.addEventListener("click", () => el.remove(), { once: true });
  el.title = "Click to dismiss";
}

async function upload(file) {
  const el = jobCard(Math.random().toString(36).slice(2), file.name);
  const form = new FormData();
  form.append("file", file);
  form.append("vision", $("vision").checked ? "true" : "false");
  let job;
  try {
    job = await api("/api/documents", { method: "POST", body: form });
  } catch (err) {
    return failJob(el, err.message);
  }
  const poll = async () => {
    try {
      job = await api(`/api/jobs/${job.id}`);
    } catch (err) {
      return failJob(el, err.message);
    }
    if (job.status === "running") {
      el.querySelector(".txt").textContent = job.stage;
      return setTimeout(poll, 1200);
    }
    if (job.status === "failed") return failJob(el, job.error || "Ingestion failed");
    el.remove();
    loadDocs();
    loadStatus();
  };
  poll();
}

/* ---------- query ---------- */

function setMode(mode) {
  if (mode === "ask" && state.backend === "none") {
    renderMessage("Ask needs Claude credentials. Set ANTHROPIC_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) and restart the server. Search works without them.");
    mode = "search";
  }
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.mode === mode;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on);
  });
  $("submit").textContent = mode === "ask" ? "Ask" : "Search";
  $("query").placeholder = mode === "ask"
    ? "Ask about revenue, margins, guidance, charts…"
    : "Search text, tables and chart descriptions…";
  $("mode-hint").textContent = mode === "ask"
    ? "Answers use only the indexed reports and cite the exact page for every claim."
    : "Hybrid retrieval: dense semantic vectors fused with BM25 keywords. No LLM call.";
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));

const q = $("query");
q.addEventListener("input", () => { q.style.height = "auto"; q.style.height = `${q.scrollHeight}px`; });
q.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("query-form").requestSubmit(); }
});
$("suggestions").addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") return;
  q.value = e.target.textContent;
  q.dispatchEvent(new Event("input"));
  $("query-form").requestSubmit();
});

$("query-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("submit").disabled = true;
  $("suggestions").hidden = true;
  $("results").innerHTML = `<div class="card"><p class="question">${esc(text)}</p>
    <div class="skeleton" style="width:92%"></div><div class="skeleton" style="width:78%"></div><div class="skeleton" style="width:60%"></div></div>`;
  try {
    if (state.mode === "ask") {
      renderAnswer(await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text }) }));
    } else {
      renderSearch(await api("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: text }) }));
    }
  } catch (err) {
    renderMessage(err.message, true);
  } finally {
    state.busy = false;
    $("submit").disabled = false;
  }
});

function renderMessage(msg, isError = false) {
  $("results").innerHTML = `<div class="card"><p class="${isError ? "error" : "muted"}" style="margin:0">${esc(msg)}</p></div>`;
}

function snippet(c) {
  // Flatten Markdown tables into "a · b · c" rows so cards stay readable; the drawer shows the real table.
  if (c.element_type !== "table") return c.text;
  return c.text.split("\n").filter((l) => !/^\|?[\s|:-]+\|?$/.test(l.trim()))
    .map((l) => l.trim().replace(/^\||\|$/g, "").split("|").map((x) => x.trim()).filter(Boolean).join(" · "))
    .filter(Boolean).join(" / ");
}

function sourceButton(c, dim = false) {
  return `<button class="source${dim ? " dim" : ""}" data-n="${c.n}">
    <span class="n">[${c.n}]</span>
    <span class="body">
      <span class="where"><span class="type ${esc(c.element_type)}">${TYPE_LABEL[c.element_type] || esc(c.element_type)}</span>
        <span>${esc(c.source)} · page ${c.page}</span><span class="score">score ${c.score.toFixed(3)}</span></span>
      <span class="snippet">${esc(snippet(c))}</span>
    </span>
  </button>`;
}

function withCitations(answer, max) {
  // [1], [1, 3] and [1][2] all become clickable chips; numbers outside the context stay plain text.
  return esc(answer).replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (m, list) => {
    const nums = list.split(",").map((n) => parseInt(n, 10));
    if (!nums.every((n) => n >= 1 && n <= max)) return m;
    return nums.map((n) => `<button class="cite" data-n="${n}" aria-label="Source ${n}">${n}</button>`).join("");
  });
}

function renderAnswer(r) {
  state.context = r.context;
  const cited = r.context.filter((c) => r.cited.includes(c.n));
  const others = r.context.filter((c) => !r.cited.includes(c.n));
  const banner = r.insufficient_context
    ? `<div class="banner">${ICON_WARN}<span>The indexed reports don't fully answer this. Here is what they do say.</span></div>` : "";
  $("results").innerHTML = `
    <div class="card">
      <p class="question">${esc(r.question)}</p>
      ${banner}
      <div class="answer">${withCitations(r.answer, r.context.length)}</div>
    </div>
    ${cited.length ? `<div class="card"><h3>Sources cited</h3><div class="sources">${cited.map((c) => sourceButton(c)).join("")}</div></div>` : ""}
    ${others.length ? `<div class="card"><details><summary>${others.length} more retrieved chunk${others.length > 1 ? "s" : ""} not cited</summary>
      <div class="sources">${others.map((c) => sourceButton(c, true)).join("")}</div></details></div>` : ""}`;
}

function renderSearch(r) {
  state.context = r.results;
  if (!r.results.length) return renderMessage("No results. Add a report first, or try different wording.");
  $("results").innerHTML = `<div class="card"><h3>${r.results.length} results for “${esc(r.query)}”</h3>
    <div class="sources">${r.results.map((c) => sourceButton(c)).join("")}</div></div>`;
}

$("results").addEventListener("click", (e) => {
  const el = e.target.closest("[data-n]");
  if (!el) return;
  const c = state.context.find((x) => x.n === Number(el.dataset.n));
  if (c) openDrawer(c);
});

/* ---------- source drawer ---------- */

function renderChunk(text) {
  // Tables are stored as Markdown; render them as real tables, everything else as escaped text.
  const lines = text.split("\n");
  let html = "", table = [];
  const flush = () => {
    if (!table.length) return;
    const rows = table.filter((l) => !/^\|\s*-{3}/.test(l) && !/^\|(-{3}\|)+$/.test(l))
      .map((l) => l.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
    // PDF tables often carry spacer columns; drop columns that are empty in every row.
    const width = Math.max(...rows.map((r) => r.length));
    const keep = [...Array(width).keys()].filter((i) => rows.some((r) => (r[i] || "").length));
    const [head, ...body] = rows.map((r) => keep.map((i) => r[i] || ""));
    html += `<table><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${
      body.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    table = [];
  };
  for (const line of lines) {
    if (line.trim().startsWith("|")) { table.push(line.trim()); continue; }
    flush();
    html += `${esc(line)}\n`;
  }
  flush();
  return html.trim();
}

function openDrawer(c) {
  const doc = encodeURIComponent(c.source);
  $("drawer-title").textContent = `[${c.n}] ${c.source}`;
  $("drawer-sub").textContent = `Page ${c.page} · ${TYPE_LABEL[c.element_type] || c.element_type} · score ${c.score.toFixed(3)}`;
  $("drawer-text").innerHTML = renderChunk(c.text);
  const img = $("drawer-img");
  img.removeAttribute("src");
  img.alt = `Page ${c.page} of ${c.source}`;
  img.onerror = () => { img.alt = "Page image unavailable (this report was ingested from the CLI, so the server has no copy of the PDF)."; };
  img.src = `/api/documents/${doc}/pages/${c.page}.png`;
  $("drawer-pdf").href = `/api/documents/${doc}/file#page=${c.page}`;
  $("drawer").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
  $("drawer-backdrop").hidden = false;
  $("drawer-close").focus();
}

function closeDrawer() {
  $("drawer").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  $("drawer-backdrop").hidden = true;
}
$("drawer-close").addEventListener("click", closeDrawer);
$("drawer-backdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

loadStatus();
loadDocs();
