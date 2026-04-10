let currentData = [];
let currentThreshold = 5.5;

document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('dataset-select');
    select.addEventListener('change', (e) => loadData(e.target.value));

    document.getElementById('threshold-apply').addEventListener('click', () => {
        currentThreshold = parseFloat(document.getElementById('threshold-input').value) || 5.5;
        renderSidebar();
        // Re-render current paper if one is selected
        const active = document.querySelector('.paper-item.active');
        if (active) active.click();
    });

    loadData(select.value);
});

async function loadData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const text = await response.text();

        currentData = text.trim().split('\n')
            .filter(line => line.trim())
            .map(line => { try { return JSON.parse(line); } catch { return null; } })
            .filter(Boolean);

        // Pick up threshold from data if present
        if (currentData.length > 0 && currentData[0].aggregation?.threshold) {
            currentThreshold = currentData[0].aggregation.threshold;
            document.getElementById('threshold-input').value = currentThreshold;
        }

        renderSidebar();
        if (currentData.length > 0) selectPaper(0);
        else showEmptyState("No papers found.");

    } catch (e) {
        console.error("Load error:", e);
        showEmptyState(`Error: ${e.message}. Run via HTTP server (python -m http.server).`);
    }
}

// ── Sidebar ──────────────────────────────────────────────────────────────────

function renderSidebar() {
    const list = document.getElementById('paper-list');
    list.innerHTML = '';

    // Re-evaluate correctness at current threshold
    currentData.forEach(p => {
        const score = p.aggregation?.avg_overall ?? 0;
        p._pred = score >= currentThreshold ? 'Accept' : 'Reject';
        p._correct = (p.label === 'ACCEPT' && p._pred === 'Accept') ||
                     (p.label === 'REJECT' && p._pred === 'Reject');
    });

    const correctCount = currentData.filter(d => d._correct).length;
    const statsDiv = document.getElementById('global-stats');
    const acc = currentData.length ? (correctCount / currentData.length * 100).toFixed(1) : 0;
    statsDiv.innerHTML = `
        <div>Accuracy: <strong>${correctCount}/${currentData.length}</strong> (${acc}%)</div>
        <div style="margin-top:4px">Threshold: <strong>${currentThreshold}</strong></div>
    `;

    currentData.forEach((paper, index) => {
        const div = document.createElement('div');
        div.className = 'paper-item';
        div.onclick = () => selectPaper(index);

        const matchClass = paper._correct ? 'match-true' : 'match-false';
        const matchIcon = paper._correct ? '\u2713' : '\u2717';
        const score = (paper.aggregation?.avg_overall ?? 0).toFixed(1);

        div.innerHTML = `
            <div class="paper-item-title">${esc(paper.title || "Untitled")}</div>
            <div class="paper-meta-tags">
                <span class="match-tag ${matchClass}">${matchIcon}</span>
                <span class="score-tag">${score}</span>
                <span class="label-tag label-${paper.label?.toLowerCase()}">${paper.label || '?'}</span>
                <span class="id-tag">${paper.paper_id || ''}</span>
            </div>
        `;
        list.appendChild(div);
    });
}

function selectPaper(index) {
    document.querySelectorAll('.paper-item').forEach((el, i) => {
        el.classList.toggle('active', i === index);
    });

    const paper = currentData[index];
    if (!paper) return;

    document.getElementById('welcome-state').classList.add('hidden');
    document.getElementById('rubric-view').classList.remove('hidden');

    renderHeader(paper);
    renderScoreOverview(paper);
    renderPersonas(paper);
    renderReviews(paper);
    renderDiscussion(paper);
    renderAggregation(paper);
}

function showEmptyState(msg) {
    document.getElementById('welcome-state').classList.remove('hidden');
    document.getElementById('rubric-view').classList.add('hidden');
    if (msg) document.querySelector('#welcome-state p').textContent = msg;
}

// ── Render functions ─────────────────────────────────────────────────────────

function renderHeader(p) {
    const header = document.getElementById('paper-header');
    const gt = (p.label || '?').toUpperCase();
    const pred = p._pred.toUpperCase();

    const timeBlock = p.elapsed_seconds != null ? `
            <div class="verdict-item">
                <span class="verdict-label">Eval Time</span>
                <span class="verdict-value" style="color:var(--accent-muted, #a78bfa)" title="${p.elapsed_seconds}s total">${fmtTime(p.elapsed_seconds)}</span>
            </div>` : '';

    header.innerHTML = `
        <h2>${esc(p.title || "Untitled")}</h2>
        <div class="header-meta">
            <span><a href="${p.forum_url || '#'}" target="_blank" class="link-accent">ID: ${p.paper_id}</a></span>
            <span class="sep"></span>
            <span>Method: <strong>${p.method || 'rubric_council'}</strong></span>
            <span class="sep"></span>
            <span>Profiler: <strong>${p.profiler_model || '?'}</strong></span>
        </div>

        <div class="verdict-box">
            <div class="verdict-item">
                <span class="verdict-label">Ground Truth</span>
                <span class="verdict-value verdict-${gt.toLowerCase()}">${gt}</span>
            </div>
            <div class="verdict-item">
                <span class="verdict-label">Prediction (t=${currentThreshold})</span>
                <span class="verdict-value ${p._correct ? 'verdict-correct' : 'verdict-incorrect'}">${pred}</span>
            </div>
            <div class="verdict-item">
                <span class="verdict-label">Avg Overall</span>
                <span class="verdict-value" style="color:white">${(p.aggregation?.avg_overall ?? 0).toFixed(2)}</span>
            </div>
            ${timeBlock}
        </div>

        <div class="model-list">
            ${(p.reviewer_models || []).map((m, i) => `<span class="model-tag">R${i+1}: ${m}</span>`).join('')}
        </div>
    `;
}

function renderScoreOverview(p) {
    const container = document.getElementById('score-overview');
    const agg = p.aggregation || {};
    const initial = p.initial_reviews || [];
    const final_ = p.final_scores || [];

    // Build comparison table: initial → final (post-discussion)
    const dims = ['overall', 'soundness', 'contribution', 'clarity', 'confidence'];
    const dimLabels = { overall: 'Overall (1-10)', soundness: 'Soundness (1-4)',
                        contribution: 'Contribution (1-4)', clarity: 'Clarity (1-4)',
                        confidence: 'Confidence (1-5)' };

    let tableHtml = `<table class="score-table">
        <thead><tr><th>Dimension</th>`;
    for (let i = 0; i < initial.length; i++) tableHtml += `<th>R${i+1} Initial</th><th>R${i+1} Final</th>`;
    tableHtml += `<th>Weighted Avg</th></tr></thead><tbody>`;

    for (const dim of dims) {
        tableHtml += `<tr><td class="dim-name">${dimLabels[dim] || dim}</td>`;
        for (let i = 0; i < initial.length; i++) {
            const initScore = initial[i]?.scores?.[dim] ?? '?';
            const finalScore = final_[i]?.[dim] ?? '?';
            const changed = initScore !== finalScore && initScore !== '?' && finalScore !== '?';
            const changeClass = changed ? (finalScore > initScore ? 'score-up' : 'score-down') : '';
            tableHtml += `<td>${initScore}</td>`;
            tableHtml += `<td class="${changeClass}">${finalScore}${changed ? (finalScore > initScore ? ' ↑' : ' ↓') : ''}</td>`;
        }
        const avgKey = 'avg_' + dim;
        const avgVal = agg[avgKey] !== undefined ? agg[avgKey].toFixed(2) : '—';
        tableHtml += `<td class="avg-col">${avgVal}</td>`;
        tableHtml += `</tr>`;
    }
    tableHtml += `</tbody></table>`;

    container.innerHTML = tableHtml;
}

function renderPersonas(p) {
    const grid = document.getElementById('personas-grid');
    const personas = p.personas || [];
    const colors = ['var(--r1)', 'var(--r2)', 'var(--r3)'];

    grid.innerHTML = personas.map((per, i) => `
        <div class="role-card" style="border-left-color: ${colors[i]}">
            <div class="role-header" style="color: ${colors[i]}">R${i+1}</div>
            <div class="role-name">${esc(per.persona || '—')}</div>
            <div class="role-focus">Focus: ${esc(per.focus_area || '—')}</div>
            <div class="role-instruction">${esc(per.instruction || '—')}</div>
        </div>
    `).join('');
}

function renderReviews(p) {
    const grid = document.getElementById('reviews-grid');
    const reviews = p.initial_reviews || [];
    const colors = ['var(--r1)', 'var(--r2)', 'var(--r3)'];

    grid.innerHTML = reviews.map((rev, i) => {
        const scores = rev.scores || {};
        const scoreBadges = Object.entries(scores)
            .map(([k, v]) => `<span class="score-badge">${k}: <strong>${v}</strong></span>`)
            .join('');

        return `
            <div class="role-card" style="border-left-color: ${colors[i]}">
                <div class="role-header" style="color: ${colors[i]}">
                    R${i+1} — ${esc(rev.persona || '')}
                    <span class="model-tag-sm">${rev.model || '?'}</span>
                </div>
                <div class="score-badges">${scoreBadges}</div>
                <div class="parse-status ${rev.parse_success ? 'parse-ok' : 'parse-fail'}">
                    ${rev.parse_success ? 'Parsed OK' : 'PARSE FAILED'}
                </div>
                <details class="review-details">
                    <summary>Full Review</summary>
                    <div class="markdown-body">${marked.parse(rev.raw_review || 'No review text.')}</div>
                </details>
            </div>
        `;
    }).join('');
}

function renderDiscussion(p) {
    const grid = document.getElementById('discussion-grid');
    const discussions = p.discussion_reviews || [];
    const colors = ['var(--r1)', 'var(--r2)', 'var(--r3)'];

    if (!discussions.length) {
        grid.innerHTML = '<p class="muted">No discussion phase recorded.</p>';
        return;
    }

    grid.innerHTML = discussions.map((disc, i) => {
        const revised = disc.revised_scores || {};
        const changes = disc.score_changes || {};
        const hasChanges = Object.keys(changes).length > 0;

        const scoreBadges = Object.entries(revised)
            .map(([k, v]) => {
                const delta = changes[k];
                let arrow = '';
                if (delta && delta !== 0) arrow = delta > 0 ? ` <span class="score-up">+${delta}</span>` : ` <span class="score-down">${delta}</span>`;
                return `<span class="score-badge">${k}: <strong>${v}</strong>${arrow}</span>`;
            }).join('');

        return `
            <div class="role-card" style="border-left-color: ${colors[i]}">
                <div class="role-header" style="color: ${colors[i]}">
                    R${i+1} — ${esc(disc.persona || '')}
                    ${hasChanges ? '<span class="badge badge-revised">REVISED</span>' : '<span class="badge badge-no-change">NO CHANGE</span>'}
                </div>
                <div class="score-badges">${scoreBadges}</div>
                <details class="review-details">
                    <summary>Discussion Text</summary>
                    <div class="markdown-body">${marked.parse(disc.raw_discussion || 'No discussion text.')}</div>
                </details>
            </div>
        `;
    }).join('');
}

function renderAggregation(p) {
    const panel = document.getElementById('aggregation-panel');
    const agg = p.aggregation || {};
    const weights = agg.weights || [];
    const individuals = agg.individual_scores || [];

    const vetoSound = (agg.avg_soundness ?? 0) < 2.0;
    const vetoContrib = (agg.avg_contribution ?? 0) < 2.0;
    const hasVeto = vetoSound || vetoContrib;

    const timeMetric = p.elapsed_seconds != null ? `
        <div class="agg-metric">
            <div class="agg-label">Eval Time</div>
            <div class="agg-value" style="color:var(--accent-muted, #a78bfa)" title="${p.elapsed_seconds}s total">${fmtTime(p.elapsed_seconds)}</div>
        </div>` : '';

    panel.innerHTML = `
        <div class="agg-grid">
            <div class="agg-metric">
                <div class="agg-label">Weighted Avg Overall</div>
                <div class="agg-value">${(agg.avg_overall ?? 0).toFixed(2)}</div>
            </div>
            <div class="agg-metric">
                <div class="agg-label">Threshold</div>
                <div class="agg-value">${currentThreshold}</div>
            </div>
            <div class="agg-metric">
                <div class="agg-label">Decision</div>
                <div class="agg-value ${p._correct ? 'verdict-correct' : 'verdict-incorrect'}">${p._pred}</div>
            </div>
            <div class="agg-metric">
                <div class="agg-label">Soundness Avg</div>
                <div class="agg-value ${vetoSound ? 'veto-active' : ''}">${(agg.avg_soundness ?? 0).toFixed(2)} ${vetoSound ? '(VETO)' : ''}</div>
            </div>
            <div class="agg-metric">
                <div class="agg-label">Contribution Avg</div>
                <div class="agg-value ${vetoContrib ? 'veto-active' : ''}">${(agg.avg_contribution ?? 0).toFixed(2)} ${vetoContrib ? '(VETO)' : ''}</div>
            </div>
            <div class="agg-metric">
                <div class="agg-label">Clarity Avg</div>
                <div class="agg-value">${(agg.avg_clarity ?? 0).toFixed(2)}</div>
            </div>
            ${timeMetric}
        </div>
        <div class="weights-row">
            Confidence weights: ${weights.map((w, i) => `R${i+1}=${w}`).join(', ')}
        </div>
    `;
}

// ── Utility ──────────────────────────────────────────────────────────────────

function esc(unsafe) {
    if (!unsafe) return "";
    return String(unsafe)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function fmtTime(secs) {
    if (secs == null) return null;
    const m = Math.floor(secs / 60);
    const s = Math.round(secs % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
}
