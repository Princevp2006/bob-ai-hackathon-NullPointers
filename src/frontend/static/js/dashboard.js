/**
 * PowerGuard AI — Dashboard JS
 * Fetches live data from the Flask API and renders Chart.js visualisations.
 *
 * Endpoints used:
 *   GET  /api/dashboard/summary          → KPI cards + top-risk table
 *   GET  /api/dashboard/feature-importance → Model A feature importance bar chart
 *   POST /api/predict/all               → "Run All Predictions" button
 *   GET  /api/weather/maintenance?regenerate=true → "Generate Plan" button
 */

'use strict';

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Build a coloured risk badge element. */
function badgeEl(level) {
    const span = document.createElement('span');
    span.className = `badge badge--${(level || 'unscored').toLowerCase()}`;
    span.textContent = level || 'UNSCORED';
    return span;
}

/** Format a number with thousands separator. */
function fmtNum(n) {
    if (n == null) return '—';
    return Number(n).toLocaleString();
}

/** Format an ISO timestamp to a readable short form. */
function fmtTs(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
        + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

/** Show an inline status message. */
function showStatus(elId, msg, type = 'info') {
    const el = document.getElementById(elId);
    if (!el) return;
    el.innerHTML = `<div class="alert alert--${type}">${msg}</div>`;
}
function clearStatus(elId) {
    const el = document.getElementById(elId);
    if (el) el.innerHTML = '';
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────

function populateKpis(data) {
    const rc = data.risk_counts || {};
    document.getElementById('kpi-total').textContent    = fmtNum(data.total_assets);
    document.getElementById('kpi-critical').textContent = fmtNum(rc.CRITICAL || 0);
    document.getElementById('kpi-high').textContent     = fmtNum(rc.HIGH     || 0);
    document.getElementById('kpi-medium').textContent   = fmtNum(rc.MEDIUM   || 0);
    document.getElementById('kpi-low').textContent      = fmtNum(rc.LOW      || 0);
    document.getElementById('kpi-customers').textContent =
        fmtNum(data.total_at_risk_customers);
}

// ── Risk Distribution Doughnut ─────────────────────────────────────────────────

let _riskChart = null;
function renderRiskDist(rc) {
    const ctx = document.getElementById('chart-risk-dist');
    if (!ctx) return;
    if (_riskChart) { _riskChart.destroy(); }

    const labels = ['Critical', 'High', 'Medium', 'Low', 'Unscored'];
    const values = [
        rc.CRITICAL || 0, rc.HIGH || 0, rc.MEDIUM || 0, rc.LOW || 0, rc.UNSCORED || 0,
    ];
    const colors = ['#7c3aed', '#dc2626', '#d97706', '#16a34a', '#94a3b8'];

    _riskChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{ data: values, backgroundColor: colors, borderWidth: 2, borderColor: '#fff' }],
        },
        options: {
            responsive: true,
            cutout: '62%',
            plugins: {
                legend: { position: 'bottom', labels: { font: { size: 12 }, padding: 12 } },
                tooltip: { callbacks: { label: c => ` ${c.label}: ${c.parsed}` } },
            },
        },
    });
}

// ── Feature Importance Bar Chart ──────────────────────────────────────────────

let _featChart = null;
function renderFeatImp(items) {
    const ctx = document.getElementById('chart-feat-imp');
    if (!ctx) return;
    if (_featChart) { _featChart.destroy(); }

    // Show top 10 only
    const top = items.slice(0, 10);
    const labels = top.map(i => i.feature.replace(/_/g, ' '));
    const values = top.map(i => +(i.importance * 100).toFixed(2));

    _featChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Importance (%)',
                data: values,
                backgroundColor: values.map((_, i) =>
                    i < 3 ? '#2563eb' : '#93c5fd'),
                borderRadius: 4,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => ` ${c.parsed.x.toFixed(2)}%` } },
            },
            scales: {
                x: { grid: { color: '#f1f5f9' }, ticks: { font: { size: 11 } } },
                y: { grid: { display: false }, ticks: { font: { size: 11 } } },
            },
        },
    });
}

// ── Top-Risk Table ─────────────────────────────────────────────────────────────

function populateTopRiskTable(topRisks, allAssets) {
    const tbody = document.getElementById('top-risk-body');
    if (!tbody) return;

    // Merge allAssets (which has latest_risk) with topRisks from summary
    // topRisks is already sorted; if empty, fall back to allAssets sorted by score
    let rows = topRisks.length > 0 ? topRisks : buildFallbackRows(allAssets);

    if (rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="table-empty">No predictions yet. Click "Run All Predictions".</td></tr>';
        return;
    }

    tbody.innerHTML = rows.map((r, idx) => {
        const level = r.risk_level || 'UNSCORED';
        const score = r.combined_risk_score != null
            ? r.combined_risk_score.toFixed(3) : '—';
        const ts = fmtTs(r.predicted_at);
        return `
        <tr>
            <td><span class="rank-cell rank-cell--${idx + 1}">${idx + 1}</span></td>
            <td class="fw-600 nowrap">${r.asset_name || r.name || '—'}</td>
            <td class="text-muted nowrap">${r.region || '—'}</td>
            <td>${r.voltage_kv || '—'}</td>
            <td><span class="badge badge--${level.toLowerCase()}">${level}</span></td>
            <td>
                <div class="risk-bar-wrap">
                    <div class="risk-bar">
                        <div class="risk-bar__fill risk-bar__fill--${level.toLowerCase()}"
                             style="width:${((r.combined_risk_score || 0) * 100).toFixed(1)}%"></div>
                    </div>
                    <span class="risk-bar__label">${score}</span>
                </div>
            </td>
            <td>${fmtNum(r.customers_served)}</td>
            <td><span class="badge badge--${(r.model_a_risk_class||'unscored').toLowerCase()}">${r.model_a_risk_class||'—'}</span></td>
            <td><span class="badge badge--${(r.model_b_risk_class||'unscored').toLowerCase()}">${r.model_b_risk_class||'—'}</span></td>
            <td class="text-muted nowrap">${ts}</td>
        </tr>`;
    }).join('');
}

function buildFallbackRows(allAssets) {
    return allAssets
        .filter(a => a.latest_risk)
        .map(a => ({
            asset_name: a.name, region: a.region, voltage_kv: a.voltage_kv,
            customers_served: a.customers_served,
            risk_level: a.latest_risk.risk_level,
            combined_risk_score: a.latest_risk.combined_risk_score,
            model_a_risk_class: a.latest_risk.model_a_risk_class,
            model_b_risk_class: a.latest_risk.model_b_risk_class,
            predicted_at: a.latest_risk.predicted_at,
        }))
        .sort((a, b) => (b.combined_risk_score || 0) - (a.combined_risk_score || 0))
        .slice(0, 10);
}

// ── Data loading ──────────────────────────────────────────────────────────────

async function loadDashboard() {
    try {
        const [summaryRes, featRes, assetsRes] = await Promise.all([
            fetch('/api/dashboard/summary'),
            fetch('/api/dashboard/feature-importance'),
            fetch('/api/assets/'),
        ]);

        if (!summaryRes.ok) throw new Error(`Summary API error ${summaryRes.status}`);
        const summary = await summaryRes.json();
        populateKpis(summary);
        renderRiskDist(summary.risk_counts || {});
        populateTopRiskTable(summary.top_risks || [], []);

        if (featRes.ok) {
            const featData = await featRes.json();
            renderFeatImp(featData.feature_importances || []);
        }

        if (assetsRes.ok) {
            const assetsData = await assetsRes.json();
            // Re-populate table with full asset list if top_risks was empty
            if ((summary.top_risks || []).length === 0) {
                populateTopRiskTable([], assetsData.assets || []);
            }
        }
    } catch (err) {
        console.error('Dashboard load error:', err);
        showStatus('action-status', `Failed to load dashboard data: ${err.message}`, 'error');
    }
}

// ── Button handlers ───────────────────────────────────────────────────────────

document.getElementById('btn-run-all')?.addEventListener('click', async () => {
    showStatus('action-status', 'Running predictions for all assets…', 'info');
    try {
        const res = await fetch('/api/predict/all', { method: 'POST' });
        const data = await res.json();
        showStatus('action-status',
            `Predictions complete: ${data.count} assets. ` +
            `Critical: ${data.risk_summary.CRITICAL}, High: ${data.risk_summary.HIGH}, ` +
            `Medium: ${data.risk_summary.MEDIUM}, Low: ${data.risk_summary.LOW}`,
            'info');
        await loadDashboard();
    } catch (err) {
        showStatus('action-status', `Prediction error: ${err.message}`, 'error');
    }
});

document.getElementById('btn-gen-plan')?.addEventListener('click', async () => {
    showStatus('action-status', 'Generating maintenance plan…', 'info');
    try {
        const res = await fetch('/api/weather/maintenance?regenerate=true');
        const data = await res.json();
        showStatus('action-status',
            `Maintenance plan generated: ${data.count} work orders. ` +
            '<a href="/maintenance">View plan →</a>', 'info');
    } catch (err) {
        showStatus('action-status', `Plan error: ${err.message}`, 'error');
    }
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', loadDashboard);
