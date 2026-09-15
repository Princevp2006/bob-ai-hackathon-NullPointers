/**
 * PowerGuard AI — Maintenance Plan Page JS
 * Fetches the prioritised maintenance plan and renders a ranked work-order table.
 *
 * Endpoints used:
 *   GET /api/weather/maintenance              → current pending orders
 *   GET /api/weather/maintenance?regenerate=true → regenerate and reload
 */

'use strict';

// ── Helpers ───────────────────────────────────────────────────────────────────

function badgeHtml(level) {
    const l = (level || 'unscored').toLowerCase();
    return `<span class="badge badge--${l}">${level || '—'}</span>`;
}

function riskBarHtml(score, level) {
    if (score == null) return '—';
    const l = (level || 'unscored').toLowerCase();
    const pct = Math.min(score * 100, 100).toFixed(1);
    return `
        <div class="risk-bar-wrap">
            <div class="risk-bar">
                <div class="risk-bar__fill risk-bar__fill--${l}" style="width:${pct}%"></div>
            </div>
            <span class="risk-bar__label">${score.toFixed(3)}</span>
        </div>`;
}

function fmtDate(iso) {
    if (!iso) return '—';
    return iso;   // already "YYYY-MM-DD"
}

function statusBadge(status) {
    const map = { pending: 'medium', in_progress: 'high', done: 'low' };
    const cls = map[status] || 'unscored';
    return `<span class="badge badge--${cls}">${status}</span>`;
}

function showStatus(msg, type = 'info') {
    const el = document.getElementById('maint-status');
    if (el) el.innerHTML = `<div class="alert alert--${type}">${msg}</div>`;
}

// ── KPI counters ──────────────────────────────────────────────────────────────

function updateKpis(plan) {
    const actionMap = {
        'Emergency': 0, 'Priority': 0, 'Scheduled': 0, 'Routine': 0,
    };
    let totalHours = 0;

    plan.forEach(o => {
        const a = o.action_type || '';
        if (a.startsWith('Emergency'))  actionMap['Emergency']++;
        else if (a.startsWith('Priority')) actionMap['Priority']++;
        else if (a.startsWith('Scheduled')) actionMap['Scheduled']++;
        else actionMap['Routine']++;
        totalHours += o.estimated_hours || 0;
    });

    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('mk-emergency', actionMap['Emergency']);
    set('mk-priority',  actionMap['Priority']);
    set('mk-scheduled', actionMap['Scheduled']);
    set('mk-routine',   actionMap['Routine']);
    set('mk-hours',     totalHours.toFixed(0));
}

// ── Table render ──────────────────────────────────────────────────────────────

function renderTable(plan) {
    const tbody = document.getElementById('maint-body');
    if (!tbody) return;

    if (!plan || plan.length === 0) {
        tbody.innerHTML = `
            <tr><td colspan="10" class="table-empty">
                No maintenance orders yet.
                Click "Run All Predictions" on the dashboard first,
                then click "Regenerate Plan" above.
            </td></tr>`;
        return;
    }

    tbody.innerHTML = plan.map(o => {
        const level = o.risk_level || 'LOW';
        const score = o.combined_risk_score != null ? o.combined_risk_score : (o.priority_score || null);
        const rankCls = o.priority_rank <= 3 ? `rank-cell--${o.priority_rank}` : '';
        return `
        <tr>
            <td><span class="rank-cell ${rankCls}">${o.priority_rank}</span></td>
            <td class="fw-600 nowrap">${o.asset_name || '—'}</td>
            <td class="text-muted nowrap">${o.region || '—'}</td>
            <td>${badgeHtml(level)}</td>
            <td>${riskBarHtml(score, level)}</td>
            <td class="nowrap" style="max-width:240px;white-space:normal;font-size:.82rem">${o.action_type}</td>
            <td class="nowrap">${fmtDate(o.recommended_date)}</td>
            <td>${o.crew_size} crew</td>
            <td>${o.estimated_hours} h</td>
            <td>${statusBadge(o.status || 'pending')}</td>
        </tr>`;
    }).join('');
}

// ── Load plan ─────────────────────────────────────────────────────────────────

async function loadPlan(regenerate = false) {
    const url = regenerate
        ? '/api/weather/maintenance?regenerate=true'
        : '/api/weather/maintenance';
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`API ${res.status}`);
        const data = await res.json();
        const plan = data.plan || [];
        updateKpis(plan);
        renderTable(plan);
        if (regenerate) {
            showStatus(`Plan regenerated: ${plan.length} work orders.`, 'info');
        }
    } catch (err) {
        const tbody = document.getElementById('maint-body');
        if (tbody) tbody.innerHTML =
            `<tr><td colspan="10" class="table-empty">Error loading plan: ${err.message}</td></tr>`;
        showStatus(`Failed to load plan: ${err.message}`, 'error');
    }
}

// ── Buttons ───────────────────────────────────────────────────────────────────

document.getElementById('btn-reload')?.addEventListener('click', () => loadPlan(false));

document.getElementById('btn-regenerate')?.addEventListener('click', async () => {
    showStatus('Regenerating plan from latest predictions…', 'info');
    await loadPlan(true);
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => loadPlan(false));
