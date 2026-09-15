/**
 * PowerGuard AI — Assets Page JS
 * Fetches all assets from the API, renders the table with filters,
 * and opens a detail modal with sensor history sparkline on row click.
 *
 * Endpoints used:
 *   GET /api/assets/               → full asset list + latest_risk
 *   GET /api/assets/<id>           → single asset detail + sensor readings
 *   POST /api/predict/<id>         → run prediction from modal
 */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let _allAssets = [];
let _modalAssetId = null;
let _sensorChart = null;

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtNum(n) {
    if (n == null) return '—';
    return Number(n).toLocaleString();
}

function fmtTs(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function badgeHtml(level) {
    const l = (level || 'unscored').toLowerCase();
    return `<span class="badge badge--${l}">${level || 'UNSCORED'}</span>`;
}

function riskBarHtml(score, level) {
    if (score == null) return '—';
    const l = (level || 'unscored').toLowerCase();
    const pct = (score * 100).toFixed(1);
    return `
        <div class="risk-bar-wrap">
            <div class="risk-bar">
                <div class="risk-bar__fill risk-bar__fill--${l}" style="width:${pct}%"></div>
            </div>
            <span class="risk-bar__label">${score.toFixed(3)}</span>
        </div>`;
}

// ── Render table ──────────────────────────────────────────────────────────────

function renderTable(assets) {
    const tbody = document.getElementById('asset-body');
    const countEl = document.getElementById('filter-count');
    if (!tbody) return;

    if (countEl) countEl.textContent = `${assets.length} asset${assets.length !== 1 ? 's' : ''}`;

    if (assets.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="table-empty">No assets match the current filters.</td></tr>';
        return;
    }

    tbody.innerHTML = assets.map(a => {
        const r = a.latest_risk;
        const level = r ? r.risk_level : 'UNSCORED';
        return `
        <tr data-asset-id="${a.id}" style="cursor:pointer">
            <td class="fw-600">${a.name}</td>
            <td class="text-muted">${a.asset_type}</td>
            <td class="text-muted">${a.region}</td>
            <td>${a.state}</td>
            <td>${a.voltage_kv}</td>
            <td>${a.install_year}</td>
            <td>${fmtNum(a.customers_served)}</td>
            <td>${badgeHtml(level)}</td>
            <td>${r ? riskBarHtml(r.combined_risk_score, level) : '—'}</td>
            <td>${(a.criticality_score * 100).toFixed(1)}%</td>
            <td>
                <button class="btn btn--outline btn--sm"
                        onclick="openModal(${a.id}); event.stopPropagation()">
                    Detail
                </button>
            </td>
        </tr>`;
    }).join('');

    // Row click also opens modal
    tbody.querySelectorAll('tr[data-asset-id]').forEach(tr => {
        tr.addEventListener('click', () => openModal(+tr.dataset.assetId));
    });
}

// ── Filters ───────────────────────────────────────────────────────────────────

function applyFilters() {
    const q     = document.getElementById('search-input')?.value.toLowerCase() || '';
    const state = document.getElementById('filter-state')?.value  || '';
    const level = document.getElementById('filter-level')?.value  || '';

    const filtered = _allAssets.filter(a => {
        const matchQ = !q || a.name.toLowerCase().includes(q) || a.region.toLowerCase().includes(q);
        const matchS = !state || a.state === state;
        const matchL = !level || (a.latest_risk && a.latest_risk.risk_level === level)
                               || (!a.latest_risk && level === 'UNSCORED');
        return matchQ && matchS && matchL;
    });
    renderTable(filtered);
}

function populateStateFilter(assets) {
    const sel = document.getElementById('filter-state');
    if (!sel) return;
    const states = [...new Set(assets.map(a => a.state))].sort();
    states.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s; opt.textContent = s;
        sel.appendChild(opt);
    });
}

// ── Modal ─────────────────────────────────────────────────────────────────────

async function openModal(assetId) {
    _modalAssetId = assetId;
    const backdrop = document.getElementById('asset-modal');
    if (backdrop) backdrop.classList.add('open');

    try {
        const res = await fetch(`/api/assets/${assetId}`);
        if (!res.ok) throw new Error(`API ${res.status}`);
        const a = await res.json();

        document.getElementById('modal-title').textContent = a.name;

        // Asset info
        const infoEl = document.getElementById('modal-info');
        if (infoEl) {
            infoEl.innerHTML = `
                <dt>Type</dt>      <dd>${a.asset_type}</dd>
                <dt>Region</dt>    <dd>${a.region}</dd>
                <dt>State</dt>     <dd>${a.state}</dd>
                <dt>Voltage</dt>   <dd>${a.voltage_kv} kV</dd>
                <dt>Installed</dt> <dd>${a.install_year} (${a.age_years} yr)</dd>
                <dt>Customers</dt> <dd>${fmtNum(a.customers_served)}</dd>
                <dt>Criticality</dt><dd>${(a.criticality_score * 100).toFixed(1)}%</dd>
                <dt>Lat / Lon</dt> <dd>${a.latitude}, ${a.longitude}</dd>`;
        }

        // Risk score
        const riskEl = document.getElementById('modal-risk');
        if (riskEl) {
            const r = a.risk_scores && a.risk_scores[0];
            if (r) {
                riskEl.innerHTML = `
                    <dt>Risk Level</dt>   <dd>${r.risk_level}</dd>
                    <dt>Combined Score</dt><dd>${r.combined_risk_score}</dd>
                    <dt>Model A</dt>      <dd>${r.model_a_risk_class} (p_high=${r.model_a_proba_high})</dd>
                    <dt>Model B</dt>      <dd>${r.model_b_risk_class} (p_high=${r.model_b_proba_high})</dd>
                    <dt>Predicted At</dt> <dd>${fmtTs(r.predicted_at)}</dd>`;
            } else {
                riskEl.innerHTML = '<dt>Status</dt><dd>No predictions yet.</dd>';
            }
        }

        // Sensor history sparkline (hydrogen_ppm as proxy for health degradation)
        renderSensorSparkline(a.sensor_readings || []);

    } catch (err) {
        document.getElementById('modal-info').innerHTML =
            `<dt>Error</dt><dd>${err.message}</dd>`;
    }
}

function renderSensorSparkline(readings) {
    const ctx = document.getElementById('modal-sensor-chart');
    if (!ctx) return;
    if (_sensorChart) { _sensorChart.destroy(); }

    // Reverse to chronological order
    const sorted = [...readings].reverse();
    const labels = sorted.map(r => fmtTs(r.recorded_at));
    const h2Data = sorted.map(r => r.hydrogen_ppm || 0);
    const acetData = sorted.map(r => r.acetylene_ppm || 0);

    _sensorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'H₂ (ppm)',
                    data: h2Data,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37,99,235,.08)',
                    tension: 0.3, fill: true, pointRadius: 3,
                },
                {
                    label: 'C₂H₂ (ppm)',
                    data: acetData,
                    borderColor: '#dc2626',
                    backgroundColor: 'rgba(220,38,38,.06)',
                    tension: 0.3, fill: false, pointRadius: 3,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'bottom', labels: { font: { size: 11 } } } },
            scales: {
                x: { ticks: { font: { size: 10 } } },
                y: { ticks: { font: { size: 10 } }, title: { display: true, text: 'ppm' } },
            },
        },
    });
}

function closeModal() {
    document.getElementById('asset-modal')?.classList.remove('open');
    _modalAssetId = null;
}

// ── Modal predict button ──────────────────────────────────────────────────────

document.getElementById('modal-predict-btn')?.addEventListener('click', async () => {
    if (!_modalAssetId) return;
    const btn = document.getElementById('modal-predict-btn');
    btn.disabled = true;
    btn.textContent = 'Running…';
    try {
        const res = await fetch(`/api/predict/${_modalAssetId}`, { method: 'POST' });
        if (!res.ok) throw new Error(`API ${res.status}`);
        await openModal(_modalAssetId);   // re-load modal with fresh data
    } catch (err) {
        console.error('Predict error:', err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '&#9654; Run Prediction Now';
    }
});

// ── Close modal on backdrop click ─────────────────────────────────────────────

document.getElementById('asset-modal')?.addEventListener('click', e => {
    if (e.target === document.getElementById('asset-modal')) closeModal();
});
document.getElementById('modal-close')?.addEventListener('click', closeModal);

// ── Filter event listeners ────────────────────────────────────────────────────

['search-input', 'filter-state', 'filter-level'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', applyFilters);
});
document.getElementById('btn-clear-filters')?.addEventListener('click', () => {
    document.getElementById('search-input').value = '';
    document.getElementById('filter-state').value = '';
    document.getElementById('filter-level').value = '';
    applyFilters();
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────

async function loadAssets() {
    try {
        const res = await fetch('/api/assets/');
        if (!res.ok) throw new Error(`API ${res.status}`);
        const data = await res.json();
        _allAssets = data.assets || [];
        populateStateFilter(_allAssets);
        renderTable(_allAssets);
    } catch (err) {
        const tbody = document.getElementById('asset-body');
        if (tbody) tbody.innerHTML =
            `<tr><td colspan="11" class="table-empty">Failed to load assets: ${err.message}</td></tr>`;
    }
}

document.addEventListener('DOMContentLoaded', loadAssets);
