let currentData = [];

document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('dataset-select');
    select.addEventListener('change', (e) => loadData(e.target.value));
    
    // Load initial data
    loadData(select.value);
});

async function loadData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const text = await response.text();
        
        currentData = text.trim().split('\n')
            .filter(line => line.trim())
            .map(line => {
                try { return JSON.parse(line); } catch (e) { return null; }
            }).filter(item => item !== null);
            
        renderSidebar();
        
        // Auto-select first item if exists
        if (currentData.length > 0) {
            selectPaper(0);
        } else {
            showEmptyState("No papers found in this dataset.");
        }
        
    } catch (e) {
        console.error("Failed to load dataset:", e);
        showEmptyState(`Error loading dataset: ${e.message}. Make sure you are running via HTTP server.`);
    }
}

function renderSidebar() {
    const list = document.getElementById('paper-list');
    list.innerHTML = '';
    
    // Update stats
    const correctCount = currentData.filter(d => d.correct).length;
    const statsDiv = document.getElementById('global-stats');
    statsDiv.innerHTML = `
        <div>🎯 Accuracy: <strong>${correctCount}/${currentData.length}</strong></div>
        <div style="margin-top:4px">📄 Papers: <strong>${currentData.length}</strong></div>
    `;

    currentData.forEach((paper, index) => {
        const div = document.createElement('div');
        div.className = 'paper-item';
        div.onclick = () => selectPaper(index);
        
        const isCorrect = paper.correct;
        const matchClass = isCorrect ? 'match-true' : 'match-false';
        const matchIcon = isCorrect ? '✓' : '✗';
        
        div.innerHTML = `
            <div class="paper-item-title">${paper.title || "Untitled Paper"}</div>
            <div class="paper-meta-tags">
                <span class="match-tag ${matchClass}">${matchIcon}</span>
                <span class="id-tag">${paper.paper_id || 'unknown'}</span>
            </div>
        `;
        list.appendChild(div);
    });
}

function selectPaper(index) {
    // Update active state in sidebar
    document.querySelectorAll('.paper-item').forEach((el, i) => {
        if (i === index) el.classList.add('active');
        else el.classList.remove('active');
    });

    const paper = currentData[index];
    if (!paper) return;

    document.getElementById('welcome-state').classList.add('hidden');
    document.getElementById('debate-view').classList.remove('hidden');
    
    renderHeader(paper);
    renderPersonas(paper);
    renderInitialAnalysis(paper);
    renderDebateRounds(paper);
    renderVerdict(paper);
}

function showEmptyState(msg) {
    document.getElementById('welcome-state').classList.remove('hidden');
    document.getElementById('debate-view').classList.add('hidden');
    if (msg) {
        document.querySelector('#welcome-state p').textContent = msg;
    }
}

// ── Rendering Helpers ────────────────────────────────────────────────────────

function renderHeader(p) {
    const header = document.getElementById('paper-header');
    const gt = (p.decision || p.label || "?").toUpperCase();
    const pred = (p.prediction || "None").toUpperCase();
    const isCorrect = p.correct;
    
    header.innerHTML = `
        <h2>${p.title || "Untitled Paper"}</h2>
        <div class="header-meta">
            <span><a href="${p.forum_url || '#'}" target="_blank" style="color:var(--proponent)">ID: ${p.paper_id}</a></span>
            <span>•</span>
            <span>Domain: <strong style="color:white">${p.domain || 'N/A'}</strong></span>
            <span>•</span>
            <span>Rounds: <strong style="color:white">${p.rounds_completed || (p.debate_transcript ? p.debate_transcript.length : 0)}</strong></span>
        </div>
        
        <div class="verdict-box">
            <div class="verdict-item">
                <span class="verdict-label">Ground Truth</span>
                <span class="verdict-value">${gt}</span>
            </div>
            <div class="verdict-item ${isCorrect ? 'verdict-correct' : 'verdict-incorrect'}">
                <span class="verdict-label">Council Prediction</span>
                <span class="verdict-value">${pred}</span>
            </div>
        </div>
        
        <div style="margin-top:16px; font-size:12px; opacity:0.6; display:flex; gap:16px;">
            <span>Proponent: ${p.proponent_model || '?'}</span>
            <span>Critic: ${p.critic_model || '?'}</span>
            <span>Chair: ${p.chair_model || '?'}</span>
        </div>
    `;
}

function renderPersonas(p) {
    const grid = document.getElementById('personas-grid');
    const pro = p.proponent_persona || {};
    const crit = p.critic_persona || {};
    
    grid.innerHTML = `
        <div class="role-card proponent">
            <div class="role-header">⚗️ Proponent</div>
            <div class="role-name">${pro.name || '—'}</div>
            <div class="role-mandate">${pro.mandate || '—'}</div>
        </div>
        <div class="role-card critic">
            <div class="role-header">🔬 Critic</div>
            <div class="role-name">${crit.name || '—'}</div>
            <div class="role-mandate">${crit.mandate || '—'}</div>
        </div>
    `;
}

function getVerdictBadge(text) {
    if (!text) return '';
    const upper = String(text).toUpperCase();
    if (upper.includes('[VALID]')) return `<span class="badge badge-valid">VALID</span>`;
    if (upper.includes('[INVALID]')) return `<span class="badge badge-invalid">INVALID</span>`;
    if (upper.includes('[WEAK]')) return `<span class="badge badge-weak">WEAK</span>`;
    return '';
}

function buildAnalysisList(items) {
    if (!items || items.length === 0) return '<p style="opacity:0.6; margin-top:20px; font-style:italic">No data</p>';
    
    // Handle PARSE ERROR fallback
    if (items.length === 1 && items[0].claim === "PARSE_ERROR") {
        let raw = items[0].verdict || "";
        try {
            const clean = raw.replace(/```json/g, '').replace(/```/g, '').trim();
            const start = clean.indexOf('[');
            const end = clean.lastIndexOf(']') + 1;
            items = JSON.parse(clean.substring(start, end));
        } catch(e) {
            return `<div class="markdown-body" style="margin-top:20px">${marked.parse(raw)}</div>`;
        }
    }

    return items.map((item, i) => {
        const claim = item.claim || '—';
        const evidence = item.evidence || item.proof || '—';
        const verdict = item.verdict || item.evaluation || '—';
        const badge = getVerdictBadge(verdict);
        
        let extraHtml = '';
        Object.keys(item).forEach(k => {
            if (!['claim', 'evidence', 'proof', 'verdict', 'evaluation'].includes(k) && item[k]) {
                extraHtml += `<div class="detail-row"><span class="detail-label">${k}</span><div style="opacity:0.9">${item[k]}</div></div>`;
            }
        });
        
        return `
            <div class="analysis-item">
                <div class="claim-title">#${i + 1} — ${escapeHtml(claim)}</div>
                
                <div class="detail-row">
                    <span class="detail-label">Evidence</span>
                    <div style="opacity:0.85">${escapeHtml(evidence)}</div>
                </div>
                
                <div class="detail-row">
                    <span class="detail-label" style="display:flex; align-items:center; gap:10px">Verdict ${badge}</span>
                    <div class="markdown-body" style="font-family:inherit">${marked.parse(verdict)}</div>
                </div>
                
                ${extraHtml}
            </div>
        `;
    }).join("");
}

function renderInitialAnalysis(p) {
    const grid = document.getElementById('initial-analysis-grid');
    grid.innerHTML = `
        <div>
            <div style="color:var(--proponent); font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:1px">⚗️ Proponent — ${(p.proponent_persona || {}).name || ''}</div>
            ${buildAnalysisList(p.proponent_analysis)}
        </div>
        <div>
            <div style="color:var(--critic); font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:1px">🔬 Critic — ${(p.critic_persona || {}).name || ''}</div>
            ${buildAnalysisList(p.critic_analysis)}
        </div>
    `;
}

function buildPositionChanges(changes) {
    if (!changes || changes.length === 0) return `<div style="opacity:0.5; font-style:italic">No position changes this round.</div>`;
    
    return changes.map(c => `
        <div class="change-card">
            <span class="badge badge-revised">REVISED</span>
            <div class="change-claim">${escapeHtml(c.claim || '?')}</div>
            <div class="change-arrows">
                <div><span style="opacity:0.5">was</span> &rarr; ${escapeHtml(c.original_verdict || '—')}</div>
                <div><span style="opacity:0.5">now</span> &rarr; <span style="font-weight:600; color:white">${escapeHtml(c.revised_verdict || '—')}</span></div>
            </div>
            <div style="font-size:13px; margin-top:8px"><span style="opacity:0.5; text-transform:uppercase; font-size:11px; margin-right:8px">Reason</span> <em>${escapeHtml(c.reason || '—')}</em></div>
        </div>
    `).join("");
}

function renderDebateRounds(p) {
    const container = document.getElementById('debate-rounds');
    const rounds = p.debate_transcript || [];
    
    if (rounds.length === 0) {
        container.innerHTML = '<p style="opacity:0.6; font-style:italic">No debate rounds recorded.</p>';
        return;
    }
    
    container.innerHTML = rounds.map((rnd, i) => {
        const proR = rnd.proponent || {};
        const critR = rnd.critic || {};
        const proS = proR.structured || {};
        const critS = critR.structured || {};
        
        const proChanges = proS.position_changes || [];
        const critChanges = critS.position_changes || [];
        
        return `
            <h4 class="round-title">
                Round ${i + 1} &nbsp;&mdash;&nbsp; 
                Proponent revised ${proChanges.length} position(s) &nbsp;|&nbsp; 
                Critic revised ${critChanges.length} position(s)
            </h4>
            <div class="grid-2">
                <div class="role-card proponent">
                    <div class="role-header" style="opacity:0.7">⚗️ Proponent Rebuttal</div>
                    <div class="markdown-body">${marked.parse(proR.rebuttal || '')}</div>
                    
                    <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin: 20px 0;">
                    
                    <div class="detail-label">Position Changes</div>
                    ${buildPositionChanges(proChanges)}
                    
                    <div class="round-summary">${escapeHtml(proS.round_summary || '—')}</div>
                </div>
                
                <div class="role-card critic">
                    <div class="role-header" style="opacity:0.7">🔬 Critic Rebuttal</div>
                    <div class="markdown-body">${marked.parse(critR.rebuttal || '')}</div>
                    
                    <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin: 20px 0;">
                    
                    <div class="detail-label">Position Changes</div>
                    ${buildPositionChanges(critChanges)}
                    
                    <div class="round-summary">${escapeHtml(critS.round_summary || '—')}</div>
                </div>
            </div>
        `;
    }).join("");
}

function renderVerdict(p) {
    const panel = document.getElementById('verdict-panel');
    panel.innerHTML = `<div class="markdown-body">${marked.parse(p.verdict || 'No verdict recorded.')}</div>`;
}

// Utility to escape HTML and prevent XSS or render breaks
function escapeHtml(unsafe) {
    if (!unsafe) return "";
    return String(unsafe)
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
