/**
 * DataPilot — Main Application
 * Orchestrates the UI: tab navigation, dataset management,
 * query submission, result display, and metrics dashboard.
 */

(function () {
    'use strict';

    // ── DOM References ────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        // Nav
        tabs: $$('.nav-tab'),
        pages: $$('.page'),
        connectionDot: $('#connection-status'),
        connectionText: $('#connection-text'),

        // Analysis
        datasetSelect: $('#dataset-select'),
        uploadBtn: $('#upload-btn'),
        fileInput: $('#file-input'),
        datasetInfo: $('#dataset-info'),
        datasetChips: $('#dataset-chips'),
        questionInput: $('#question-input'),
        submitBtn: $('#submit-btn'),
        pipelineBar: $('#pipeline-bar'),
        pipelineTrack: $('#pipeline-track'),
        progressCard: $('#progress-card'),
        progressAgentName: $('#progress-agent-name'),
        progressMsg: $('#progress-msg'),
        progressLineFill: $('#progress-line-fill'),
        metadataBar: $('#metadata-bar'),
        resultsPlaceholder: $('#results-placeholder'),
        resultsContent: $('#results-content'),
        answerText: $('#answer-text'),
        chartContainer: $('#chart-container'),
        chartImage: $('#chart-image'),
        quickMetaPill: $('#quick-meta-pill'),

        // Metadata
        modelValue: $('#model-value'),
        cacheBadge: $('#cache-badge'),
        latencyValue: $('#latency-value'),
        verifiedBadge: $('#verified-badge'),

        // Metrics
        refreshMetricsBtn: $('#refresh-metrics-btn'),
        recentRequestsBody: $('#recent-requests-body'),
    };

    // ── State ─────────────────────────────────────────────────────
    let datasets = [];
    let selectedDataset = null;
    let isLoading = false;
    let socket = null;
    let progressTimer = null;
    let recordedSteps = [];

    const STAGES = [
        { key: 'planner', label: 'Planner Agent', msg: 'Formulating execution plan & schema strategy...', pct: 20 },
        { key: 'data', label: 'Data Agent', msg: 'Retrieving dataset records via deterministic MCP tools...', pct: 45 },
        { key: 'analysis', label: 'Analysis Agent', msg: 'Performing deep analytical calculations & metric synthesis...', pct: 70 },
        { key: 'visualization', label: 'Visualization Agent', msg: 'Evaluating visual needs & rendering chart...', pct: 88 },
        { key: 'verifier', label: 'Verifier Agent', msg: 'Verifying mathematical accuracy against raw data...', pct: 96 },
    ];

    // ── Helper: Resolve Active Dataset ID ─────────────────────────
    function getSelectedDatasetId() {
        if (selectedDataset && selectedDataset.dataset_id) {
            return selectedDataset.dataset_id;
        }
        if (els.datasetSelect && els.datasetSelect.value) {
            return els.datasetSelect.value;
        }
        if (datasets && datasets.length > 0) {
            return datasets[0].dataset_id;
        }
        return null;
    }

    // ── Init ──────────────────────────────────────────────────────
    async function init() {
        // Render initial idle pipeline bar
        if (els.pipelineTrack) {
            els.pipelineTrack.innerHTML = components.buildPipeline([], null);
        }

        setupEventListeners();
        setupWebSocket();
        await loadDatasets();
    }

    // ── Tab Navigation & Event Listeners ──────────────────────────
    function setupEventListeners() {
        // Tab switching
        els.tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const target = tab.dataset.tab;
                els.tabs.forEach(t => t.classList.remove('active'));
                els.pages.forEach(p => p.classList.remove('active'));
                tab.classList.add('active');
                $(`#page-${target}`).classList.add('active');

                if (target === 'metrics') refreshMetrics();
            });
        });

        // Auto-pastable Example Queries
        $$('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const queryText = btn.dataset.q || btn.textContent.trim();
                
                // Ensure a dataset is active
                let datasetId = getSelectedDatasetId();
                if (!datasetId && datasets.length > 0) {
                    datasetId = datasets[0].dataset_id;
                    els.datasetSelect.value = datasetId;
                    onDatasetChange();
                } else if (!selectedDataset && datasetId) {
                    selectedDataset = datasets.find(d => d.dataset_id === datasetId) || { dataset_id: datasetId };
                }

                // Paste query text into textarea
                els.questionInput.value = queryText;
                updateSubmitButton();
                els.questionInput.focus();

                // Interactive highlight animation
                els.questionInput.classList.add('flash-input');
                setTimeout(() => els.questionInput.classList.remove('flash-input'), 600);
            });
        });

        // Dataset selection
        els.datasetSelect.addEventListener('change', onDatasetChange);

        // Upload
        els.uploadBtn.addEventListener('click', () => els.fileInput.click());
        els.fileInput.addEventListener('change', onFileUpload);

        // Query submission
        els.submitBtn.addEventListener('click', (e) => {
            e.preventDefault();
            onSubmitQuery();
        });
        els.questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                onSubmitQuery();
            }
        });

        // Listen for all input events to dynamically update button state
        ['input', 'change', 'paste', 'keyup', 'focus', 'blur'].forEach(evt => {
            els.questionInput.addEventListener(evt, updateSubmitButton);
        });

        // Metrics refresh
        els.refreshMetricsBtn.addEventListener('click', refreshMetrics);
    }

    // ── WebSocket ─────────────────────────────────────────────────
    function setupWebSocket() {
        socket = new PipelineSocket();
        socket.onMessage((msg) => {
            switch (msg.type) {
                case 'connected':
                    updateConnectionStatus('connected');
                    break;
                case 'disconnected':
                    updateConnectionStatus('disconnected');
                    break;
                case 'error':
                    updateConnectionStatus('error');
                    break;
                case 'agent_start':
                    setActiveProgressStage(msg.agent, msg.summary);
                    break;
                case 'agent_complete':
                    markStageCompleted(msg.agent, msg.summary);
                    break;
                case 'agent_failed':
                    markStageFailed(msg.agent, msg.summary);
                    break;
                case 'pipeline_error':
                    showError(msg.error || 'Pipeline execution failed');
                    break;
            }
        });
        socket.connect();
    }

    function updateConnectionStatus(status) {
        els.connectionDot.className = 'status-dot';
        switch (status) {
            case 'connected':
                els.connectionDot.classList.add('connected');
                els.connectionText.textContent = 'Connected';
                break;
            case 'disconnected':
                els.connectionText.textContent = 'Reconnecting...';
                break;
            case 'error':
                els.connectionDot.classList.add('error');
                els.connectionText.textContent = 'Backend Offline';
                break;
            default:
                els.connectionText.textContent = 'Connecting...';
        }
    }

    // ── Dataset Management ────────────────────────────────────────
    async function loadDatasets() {
        try {
            datasets = await api.getDatasets();
            renderDatasetOptions();
            updateConnectionStatus('connected');
        } catch (e) {
            console.warn('Could not load datasets:', e);
            updateConnectionStatus('error');
        }
    }

    function renderDatasetOptions() {
        els.datasetSelect.innerHTML = '<option value="">Select a dataset...</option>';
        datasets.forEach(ds => {
            const opt = document.createElement('option');
            opt.value = ds.dataset_id;
            opt.textContent = `${ds.filename} (${ds.rows} rows × ${ds.columns} cols)`;
            els.datasetSelect.appendChild(opt);
        });

        // Auto-select first dataset by default if available
        if (datasets.length > 0) {
            const defaultId = datasets[0].dataset_id;
            els.datasetSelect.value = defaultId;
            selectedDataset = datasets[0];
            els.datasetChips.innerHTML = components.datasetChips(selectedDataset);
            els.datasetInfo.style.display = 'block';
        }
        updateSubmitButton();
    }

    function onDatasetChange() {
        const id = els.datasetSelect.value;
        if (!id) {
            selectedDataset = null;
            els.datasetInfo.style.display = 'none';
            updateSubmitButton();
            return;
        }

        selectedDataset = datasets.find(d => d.dataset_id === id) || { dataset_id: id };
        if (selectedDataset && selectedDataset.column_names) {
            els.datasetChips.innerHTML = components.datasetChips(selectedDataset);
            els.datasetInfo.style.display = 'block';
        }
        updateSubmitButton();
    }

    async function onFileUpload() {
        const file = els.fileInput.files[0];
        if (!file) return;

        els.uploadBtn.disabled = true;
        els.uploadBtn.textContent = 'Uploading...';

        try {
            const result = await api.uploadDataset(file);
            await loadDatasets();
            // Auto-select uploaded dataset
            els.datasetSelect.value = result.dataset_id;
            onDatasetChange();
        } catch (e) {
            alert(`Upload failed: ${e.message}`);
        } finally {
            els.uploadBtn.disabled = false;
            els.uploadBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                Upload
            `;
            els.fileInput.value = '';
        }
    }

    // ── Query Submission ──────────────────────────────────────────
    function updateSubmitButton() {
        const questionText = (els.questionInput.value || '').trim();
        const datasetId = getSelectedDatasetId();
        const canSubmit = questionText.length > 0 && !!datasetId && !isLoading;
        els.submitBtn.disabled = !canSubmit;
    }

    async function onSubmitQuery() {
        if (isLoading) return;

        const question = (els.questionInput.value || '').trim();
        if (!question) {
            els.questionInput.focus();
            return;
        }

        let datasetId = getSelectedDatasetId();
        if (!datasetId && datasets.length > 0) {
            datasetId = datasets[0].dataset_id;
            els.datasetSelect.value = datasetId;
            onDatasetChange();
        }
        if (!datasetId) {
            await loadDatasets();
            datasetId = getSelectedDatasetId();
        }
        if (!datasetId) {
            alert('Please select or upload a dataset before analyzing.');
            return;
        }

        isLoading = true;
        updateSubmitButton();
        startProgressTracking();

        try {
            const response = await api.query(question, datasetId);
            showResults(response);
        } catch (e) {
            showError(e.message || 'Analysis failed. Please check backend connection.');
        } finally {
            isLoading = false;
            stopProgressTracking();
            updateSubmitButton();
        }
    }

    // ── Live Horizontal Progress Tracking ─────────────────────────
    function startProgressTracking() {
        recordedSteps = [];
        if (progressTimer) clearInterval(progressTimer);

        els.resultsPlaceholder.style.display = 'none';
        els.resultsContent.style.display = 'none';
        els.progressCard.style.display = 'flex';
        els.pipelineBar.style.display = 'block';

        let currentStageIdx = 0;
        const applyStage = (idx) => {
            const stage = STAGES[idx] || STAGES[0];
            els.progressAgentName.textContent = stage.label;
            els.progressMsg.textContent = stage.msg;
            els.progressLineFill.style.width = `${stage.pct}%`;
            els.pipelineTrack.innerHTML = components.buildPipeline(recordedSteps, stage.key);
        };

        applyStage(0);

        // Smooth progressive timer fallback while awaiting backend response
        progressTimer = setInterval(() => {
            if (currentStageIdx < STAGES.length - 1) {
                currentStageIdx++;
                applyStage(currentStageIdx);
            }
        }, 2200);
    }

    function stopProgressTracking() {
        if (progressTimer) {
            clearInterval(progressTimer);
            progressTimer = null;
        }
        els.progressCard.style.display = 'none';
    }

    function setActiveProgressStage(agentName, summary) {
        const key = (agentName || '').toLowerCase();
        const stage = STAGES.find(s => s.key === key);
        if (stage) {
            els.progressAgentName.textContent = stage.label;
            if (summary) els.progressMsg.textContent = summary;
            els.progressLineFill.style.width = `${stage.pct}%`;
        }
        els.pipelineTrack.innerHTML = components.buildPipeline(recordedSteps, key);
    }

    function markStageCompleted(agentName, summary) {
        const key = (agentName || '').toLowerCase();
        recordedSteps.push({ agent: key, status: 'completed', summary: summary || 'Completed' });
        els.pipelineTrack.innerHTML = components.buildPipeline(recordedSteps);
    }

    function markStageFailed(agentName, summary) {
        const key = (agentName || '').toLowerCase();
        recordedSteps.push({ agent: key, status: 'failed', summary: summary || 'Failed' });
        els.pipelineTrack.innerHTML = components.buildPipeline(recordedSteps);
    }

    // ── Results Display ───────────────────────────────────────────
    function showResults(response) {
        stopProgressTracking();
        els.resultsPlaceholder.style.display = 'none';
        els.resultsContent.style.display = 'flex';

        // Horizontal pipeline
        els.pipelineBar.style.display = 'block';
        if (response.agent_steps && response.agent_steps.length > 0) {
            els.pipelineTrack.innerHTML = components.buildPipeline(response.agent_steps);
        }

        // Answer
        els.answerText.textContent = response.answer;
        if (els.quickMetaPill) {
            const isHit = (response.semantic_cache_status === 'HIT' || response.cache_status === 'HIT');
            els.quickMetaPill.innerHTML = `
                <span class="pill-badge pill-model">${components.modelBadge(response.model_used)}</span>
                <span class="pill-badge ${isHit ? 'pill-hit' : 'pill-miss'}">${isHit ? '⚡ CACHE HIT' : '🔄 COMPUTE'}</span>
            `;
        }

        // Chart
        if (response.visualization) {
            els.chartContainer.style.display = 'block';
            els.chartImage.src = `data:image/png;base64,${response.visualization}`;
        } else {
            els.chartContainer.style.display = 'none';
        }

        // Metadata
        els.metadataBar.style.display = 'flex';
        els.modelValue.textContent = components.modelBadge(response.model_used);
        els.cacheBadge.innerHTML = components.cacheBadge(response.semantic_cache_status || response.cache_status);
        els.latencyValue.textContent = components.formatLatency(response.total_latency_ms);
        els.verifiedBadge.innerHTML = components.verifiedBadge(response.verified);
    }

    function showError(message) {
        stopProgressTracking();
        els.resultsPlaceholder.style.display = 'none';
        els.resultsContent.style.display = 'flex';
        els.answerText.textContent = `Error: ${message}`;
        els.chartContainer.style.display = 'none';
    }

    // ── Metrics Dashboard ─────────────────────────────────────────
    async function refreshMetrics() {
        try {
            const [summary, recent] = await Promise.all([
                api.getMetrics(),
                api.getRecentMetrics(),
            ]);

            // Update metric cards
            $('#metric-total-requests').textContent = components.formatNumber(summary.total_requests);
            $('#metric-avg-latency').textContent = components.formatLatency(summary.avg_latency_ms);
            $('#metric-p95-latency').textContent = components.formatLatency(summary.p95_latency_ms);
            $('#metric-cache-rate').textContent = `${(summary.cache_hit_rate * 100).toFixed(1)}%`;
            $('#metric-total-tokens').textContent = components.formatNumber(summary.total_tokens);
            $('#metric-total-tools').textContent = components.formatNumber(summary.total_tool_calls);
            $('#metric-total-llm').textContent = components.formatNumber(summary.total_llm_calls);
            $('#metric-failures').textContent = summary.total_failures;

            // Update charts
            if (summary.model_usage && Object.keys(summary.model_usage).length > 0) {
                charts.updateModelUsage(summary.model_usage);
            }

            if (recent.records && recent.records.length > 0) {
                charts.updateLatencyChart(recent.records);

                // Update table
                els.recentRequestsBody.innerHTML = recent.records
                    .reverse()
                    .map(r => components.requestRow(r))
                    .join('');
            }

        } catch (e) {
            console.warn('Metrics refresh failed:', e);
        }
    }

    // ── Start ─────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', init);
})();
