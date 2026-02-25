/*
 * AuraSDK Memory Management UI Logic
 * Vanilla JS
 */

const API_BASE = ''; // Same origin

// DOM Elements
const views = document.querySelectorAll('.view');
const navItems = document.querySelectorAll('.nav-item');
const pageTitle = document.getElementById('page-title');
const refreshBtn = document.getElementById('refresh-btn');
const toastContainer = document.getElementById('toast-container');

// State
let currentView = 'dashboard';
let memoriesPagination = { offset: 0, limit: 20, total: 0 };

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initDashboard();
    initMemoriesView();
    initRecallView();
    initProcessView();
    initEditModal();
    
    // Load initial data
    checkHealth();
    loadDashboardStats();
    
    // Handle hash routing on load
    if (window.location.hash) {
        const target = window.location.hash.substring(1);
        if (target) navigateTo(target);
    }
});

refreshBtn.addEventListener('click', () => {
    refreshBtn.classList.add('pulse');
    setTimeout(() => refreshBtn.classList.remove('pulse'), 500);
    
    switch(currentView) {
        case 'dashboard': loadDashboardStats(); break;
        case 'memories': loadMemories(); break;
        case 'recall': 
            const recallInput = document.getElementById('recall-input');
            if (recallInput.value.trim()) document.getElementById('run-recall-btn').click();
            break;
    }
});

// --- Navigation ---
function initNavigation() {
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.getAttribute('data-target');
            navigateTo(target);
        });
    });
}

function navigateTo(targetId) {
    if (!document.getElementById(`view-${targetId}`)) return;
    
    // Update active nav
    navItems.forEach(nav => {
        if (nav.getAttribute('data-target') === targetId) {
            nav.classList.add('active');
            pageTitle.textContent = nav.textContent.trim();
        } else {
            nav.classList.remove('active');
        }
    });

    // Toggle views
    views.forEach(view => {
        if (view.id === `view-${targetId}`) {
            view.classList.add('active');
        } else {
            view.classList.remove('active');
        }
    });
    
    window.location.hash = targetId;
    currentView = targetId;

    // Trigger loads if needed
    if (targetId === 'dashboard') loadDashboardStats();
    if (targetId === 'memories') loadMemories();
}

// --- Utils ---
async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };
    
    try {
        const response = await fetch(url, { ...options, headers });
        if (!response.ok) {
            const err = await response.text();
            throw new Error(err || `HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const iconClass = type === 'success' ? 'ph-check-circle' : 'ph-warning-circle';
    
    toast.innerHTML = `
        <span><i class="ph ${iconClass}" style="margin-right: 8px;"></i>${message}</span>
        <button class="close-modal" style="margin-left: 10px; background: none; border: none; color: inherit; cursor: pointer;">
            <i class="ph ph-x"></i>
        </button>
    `;
    
    const closeBtn = toast.querySelector('button');
    closeBtn.addEventListener('click', () => removeToast(toast));
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => removeToast(toast), 4000);
}

function removeToast(toast) {
    if (toast.parentNode) {
        toast.classList.add('closing');
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 300);
    }
}

// Format relative time (e.g. "2 mins ago")
function formatTime(timestamp) {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

function getDnaBadgeHTML(dna) {
    let cls = 'dna-general';
    let text = dna;
    
    if (dna === 'user_core' || dna === '"user_core"') { cls = 'dna-identity'; text = 'Identity'; }
    else if (dna === 'domain' || dna === '"domain"') { cls = 'dna-domain'; text = 'Domain'; }
    else if (dna === 'decisions' || dna === '"decisions"') { cls = 'dna-decisions'; text = 'Decisions'; }
    else if (dna === 'working' || dna === '"working"') { cls = 'dna-working'; text = 'Working'; }
    else if (dna === 'phantom') { cls = 'dna-general'; text = 'Phantom'; }
    
    return `<span class="dna-badge ${cls}">${text}</span>`;
}

// --- 1. Dashboard ---
async function checkHealth() {
    try {
        await apiCall('/health');
        document.getElementById('connection-status').textContent = 'Server Online';
        document.querySelector('.server-status').classList.add('online');
    } catch (e) {
        document.getElementById('connection-status').textContent = 'Server Offline';
        document.querySelector('.server-status').classList.remove('online');
    }
}

async function loadDashboardStats() {
    try {
        const [stats, analytics] = await Promise.all([
            apiCall('/stats'),
            apiCall('/analytics')
        ]);
        
        // Update basic numbers
        document.getElementById('stat-total-memories').textContent = stats.total_memories.toLocaleString();
        document.getElementById('stat-plasticity-boosts').textContent = stats.plasticity_boosts.toLocaleString();
        document.getElementById('stat-plasticity-decays').textContent = stats.plasticity_decays.toLocaleString();
        document.getElementById('stat-phantoms').textContent = stats.phantom_count.toLocaleString();
        
        document.getElementById('license-info').textContent = stats.license;
        document.getElementById('version-badge').textContent = stats.version;
        
        // Update Analytics details
        document.getElementById('param-oldest').textContent = formatTime(analytics.oldest);
        document.getElementById('param-newest').textContent = formatTime(analytics.newest);
        
        // Render DNA distribution
        const dnaList = document.getElementById('dna-list');
        dnaList.innerHTML = '';
        
        if (Object.keys(analytics.by_dna).length === 0) {
            dnaList.innerHTML = '<li class="text-muted">No memories stored yet.</li>';
        } else {
            // Sort by count descending
            const sortedDna = Object.entries(analytics.by_dna).sort((a, b) => b[1] - a[1]);
            
            for (const [dna, count] of sortedDna) {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span>${getDnaBadgeHTML(dna)}</span>
                    <strong>${count.toLocaleString()}</strong>
                `;
                dnaList.appendChild(li);
            }
        }
        
    } catch (e) {
        console.error("Failed to load dashboard stats", e);
    }
}

// --- 2. Memories Explorer ---
function initMemoriesView() {
    document.getElementById('prev-page').addEventListener('click', () => {
        if (memoriesPagination.offset >= memoriesPagination.limit) {
            memoriesPagination.offset -= memoriesPagination.limit;
            loadMemories();
        }
    });
    
    document.getElementById('next-page').addEventListener('click', () => {
        if (memoriesPagination.offset + memoriesPagination.limit < memoriesPagination.total) {
            memoriesPagination.offset += memoriesPagination.limit;
            loadMemories();
        }
    });
    
    document.getElementById('dna-filter').addEventListener('change', () => {
        memoriesPagination.offset = 0;
        loadMemories();
    });
    
    // Batch UI
    const selectAll = document.getElementById('select-all');
    selectAll.addEventListener('change', (e) => {
        const checkboxes = document.querySelectorAll('.mem-checkbox:not(:disabled)');
        checkboxes.forEach(cb => cb.checked = e.target.checked);
        updateBatchDeleteButton();
    });
    
    document.getElementById('batch-delete-btn').addEventListener('click', async () => {
        const selected = Array.from(document.querySelectorAll('.mem-checkbox:checked')).map(cb => cb.value);
        if (selected.length === 0) return;
        
        if (confirm(`Are you sure you want to completely erase ${selected.length} memories? This cannot be undone.`)) {
            try {
                const btn = document.getElementById('batch-delete-btn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Deleting...';
                btn.disabled = true;
                
                const res = await apiCall('/batch-delete', {
                    method: 'POST',
                    body: JSON.stringify({ ids: selected })
                });
                showToast(`Successfully erased ${res.deleted} memories.`, 'success');
                selectAll.checked = false;
                loadMemories();
            } catch (e) {
                showToast(`Failed to branch delete: ${e.message}`, 'error');
            } finally {
                const btn = document.getElementById('batch-delete-btn');
                btn.innerHTML = '<i class="ph ph-trash"></i> Delete Selected';
                btn.disabled = false;
            }
        }
    });
}

function updateBatchDeleteButton() {
    const selected = document.querySelectorAll('.mem-checkbox:checked').length;
    const btn = document.getElementById('batch-delete-btn');
    if (selected > 0) {
        btn.disabled = false;
        btn.innerHTML = `<i class="ph ph-trash"></i> Delete ${selected}`;
    } else {
        btn.disabled = true;
        btn.innerHTML = `<i class="ph ph-trash"></i> Delete Selected`;
    }
}

async function loadMemories() {
    const tbody = document.getElementById('memories-table-body');
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    const pageInfo = document.getElementById('page-info');
    const filter = document.getElementById('dna-filter').value;
    
    tbody.innerHTML = '<tr><td colspan="6" class="text-center loading-text"><i class="ph ph-spinner ph-spin"></i> Fetching cognitive data...</td></tr>';
    
    try {
        const res = await apiCall(`/memories?offset=${memoriesPagination.offset}&limit=${memoriesPagination.limit}&dna=${filter}`);
        memoriesPagination.total = res.total;
        
        tbody.innerHTML = '';
        
        if (res.memories.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No memories found.</td></tr>';
        } else {
            res.memories.forEach(mem => {
                const tr = document.createElement('tr');
                const intensityPercent = Math.min(100, (mem.intensity / 100) * 100); // Usually intensity starts at 100+
                
                tr.innerHTML = `
                    <td><input type="checkbox" class="mem-checkbox" value="${mem.id}"></td>
                    <td>
                        <div class="cell-content" title="${mem.text.replace(/"/g, '&quot;')}">${mem.text}</div>
                    </td>
                    <td>${getDnaBadgeHTML(mem.dna)}</td>
                    <td>
                        <div class="intensity-wrapper" title="${mem.intensity.toFixed(2)}">
                            <div class="intensity-fill" style="width: ${intensityPercent}%;"></div>
                        </div>
                    </td>
                    <td style="font-size: 0.85rem; color: var(--text-secondary);">${formatTime(mem.timestamp)}</td>
                    <td>
                        <div class="actions-cell">
                            <button class="btn icon-btn edit-mem-btn" data-id="${mem.id}" data-text="${mem.text.replace(/"/g, '&quot;')}" title="Edit">
                                <i class="ph ph-pencil-simple"></i>
                            </button>
                            <button class="btn icon-btn delete-mem-btn" data-id="${mem.id}" title="Delete">
                                <i class="ph ph-trash"></i>
                            </button>
                        </div>
                    </td>
                `;
                tbody.appendChild(tr);
            });
            
            // Re-bind events
            document.querySelectorAll('.mem-checkbox').forEach(cb => {
                cb.addEventListener('change', updateBatchDeleteButton);
            });
            
            document.querySelectorAll('.delete-mem-btn').forEach(btn => {
                btn.addEventListener('click', () => deleteSingleMemory(btn.getAttribute('data-id')));
            });
            
            document.querySelectorAll('.edit-mem-btn').forEach(btn => {
                btn.addEventListener('click', () => openEditModal(btn.getAttribute('data-id'), btn.getAttribute('data-text')));
            });
        }
        
        // Update pagination UI
        const currentPage = Math.floor(memoriesPagination.offset / memoriesPagination.limit) + 1;
        const totalPages = Math.ceil(memoriesPagination.total / memoriesPagination.limit) || 1;
        
        pageInfo.textContent = `Page ${currentPage} of ${totalPages} (Total: ${memoriesPagination.total})`;
        prevBtn.disabled = memoriesPagination.offset === 0;
        nextBtn.disabled = memoriesPagination.offset + memoriesPagination.limit >= memoriesPagination.total;
        
        document.getElementById('select-all').checked = false;
        updateBatchDeleteButton();
        
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted" style="color: var(--danger)">Error loading memories: ${e.message}</td></tr>`;
    }
}

async function deleteSingleMemory(id) {
    if (!confirm("Delete this memory forever?")) return;
    
    try {
        await apiCall('/delete', {
            method: 'POST',
            body: JSON.stringify({ id })
        });
        showToast("Memory deleted.", "success");
        loadMemories();
    } catch(e) {
        showToast(`Failed to delete: ${e.message}`, "error");
    }
}

// --- 3. Recall / Search ---
function initRecallView() {
    const input = document.getElementById('recall-input');
    const btn = document.getElementById('run-recall-btn');
    
    const doRecall = async () => {
        const query = input.value.trim();
        if (!query) return;
        
        const top_k = parseInt(document.getElementById('recall-top-k').value) || 5;
        const resultsContainer = document.getElementById('recall-results');
        
        btn.disabled = true;
        btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i>';
        
        try {
            const res = await apiCall('/retrieve', {
                method: 'POST',
                body: JSON.stringify({ query, top_k })
            });
            
            resultsContainer.innerHTML = '';
            
            if (!res.results || res.results.length === 0) {
                resultsContainer.innerHTML = `
                    <div class="empty-state">
                        <i class="ph ph-ghost"></i>
                        <p>No cognitive resonance found for "${query}".</p>
                    </div>`;
            } else {
                res.results.forEach((r, idx) => {
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    card.style.animationDelay = `${idx * 0.05}s`;
                    
                    card.innerHTML = `
                        <div class="result-header">
                            <div>
                                ${getDnaBadgeHTML(r.dna)}
                                <span style="margin-left: 12px; font-size: 0.8rem">${formatTime(r.timestamp)}</span>
                            </div>
                            <div class="result-score">
                                <i class="ph ph-target"></i> ${(r.score * 100).toFixed(1)}% Match
                            </div>
                        </div>
                        <div class="result-text">${r.text}</div>
                    `;
                    resultsContainer.appendChild(card);
                });
            }
        } catch (e) {
            resultsContainer.innerHTML = `<div class="empty-state" style="color: var(--danger)"><i class="ph ph-warning"></i><p>Recall failed: ${e.message}</p></div>`;
        } finally {
            btn.disabled = false;
            btn.innerHTML = 'Recall';
        }
    };
    
    btn.addEventListener('click', doRecall);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') doRecall();
    });
}

// --- 4. Process / Ingest ---
function initProcessView() {
    const submitBtn = document.getElementById('submit-memory-btn');
    const batchBtn = document.getElementById('submit-batch-btn');
    
    submitBtn.addEventListener('click', async () => {
        const textInput = document.getElementById('process-text');
        const text = textInput.value.trim();
        const pin = document.getElementById('process-pin').checked;
        
        if (!text) return;
        
        submitBtn.disabled = true;
        const origHtml = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Processing...';
        
        try {
            await apiCall('/process', {
                method: 'POST',
                body: JSON.stringify({ text, pin })
            });
            showToast("Memory successfully encoded into Cortex.", "success");
            textInput.value = '';
            document.getElementById('process-pin').checked = false;
        } catch(e) {
            showToast(`Encoding failed: ${e.message}`, "error");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = origHtml;
        }
    });

    batchBtn.addEventListener('click', async () => {
        const textInput = document.getElementById('batch-text');
        const texts = textInput.value.split('\n').map(t => t.trim()).filter(t => t.length > 0);
        
        if (texts.length === 0) return;
        
        batchBtn.disabled = true;
        const origHtml = batchBtn.innerHTML;
        batchBtn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Batch Ingesting...';
        
        try {
            const res = await apiCall('/ingest-batch', {
                method: 'POST',
                body: JSON.stringify({ texts, pinned: false })
            });
            showToast(`Successfully batch encoded ${res.ingested} memories.`, "success");
            textInput.value = '';
        } catch(e) {
            showToast(`Batch Encoding failed: ${e.message}`, "error");
        } finally {
            batchBtn.disabled = false;
            batchBtn.innerHTML = origHtml;
        }
    });
}

// --- 5. Edit Modal ---
function initEditModal() {
    const modal = document.getElementById('edit-modal');
    const closeBtns = document.querySelectorAll('.close-modal, .close-modal-btn');
    const saveBtn = document.getElementById('save-edit-btn');
    
    closeBtns.forEach(b => b.addEventListener('click', () => {
        modal.classList.remove('visible');
    }));
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('visible');
    });
    
    saveBtn.addEventListener('click', async () => {
        const id = document.getElementById('edit-id').value;
        const text = document.getElementById('edit-text').value.trim();
        
        if (!text) return;
        
        saveBtn.disabled = true;
        saveBtn.innerHTML = 'Saving...';
        
        try {
            await apiCall('/update', {
                method: 'POST',
                body: JSON.stringify({ id, text })
            });
            
            showToast("Memory successfully updated.", "success");
            modal.classList.remove('visible');
            loadMemories(); // Refresh table
        } catch(e) {
            showToast(`Update failed: ${e.message}`, "error");
        } finally {
            saveBtn.disabled = false;
            saveBtn.innerHTML = 'Save Changes';
        }
    });
}

function openEditModal(id, currentText) {
    document.getElementById('edit-id').value = id;
    document.getElementById('edit-text').value = currentText;
    document.getElementById('edit-modal').classList.add('visible');
}
