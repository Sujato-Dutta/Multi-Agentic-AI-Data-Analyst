const components = {
    /**
     * Agent metadata with icons and labels.
     */
    agents: [
        { key: 'planner', label: 'Planner', activeLabel: 'Planning...', icon: '📋' },
        { key: 'data', label: 'Data', activeLabel: 'Fetching Data...', icon: '🗄️' },
        { key: 'analysis', label: 'Analysis', activeLabel: 'Analyzing...', icon: '📊' },
        { key: 'visualization', label: 'Viz', activeLabel: 'Visualizing...', icon: '🎨' },
        { key: 'verifier', label: 'Verifier', activeLabel: 'Verifying...', icon: '✅' },
    ],

    /**
     * Build horizontal agent pipeline HTML from agent steps array or active key.
     * Only shows red if a step fails; keeps it ultra-compact and horizontal.
     */
    buildPipeline(steps, activeKey = null) {
        // Map recorded steps by agent key
        const stepMap = {};
        (steps || []).forEach(s => {
            const k = (typeof s.agent === 'string' ? s.agent : (s.agent?.value || s.agent || '')).toLowerCase();
            stepMap[k] = s;
        });

        let foundActive = false;

        return this.agents.map((agent, i) => {
            const recorded = stepMap[agent.key];
            let status = 'pending';
            let displayLabel = agent.label;

            if (recorded) {
                if (recorded.status === 'failed') {
                    status = 'failed';
                } else if (recorded.status === 'skipped') {
                    status = 'skipped';
                } else {
                    status = 'completed';
                }
            } else if (activeKey && agent.key === activeKey) {
                status = 'active';
                displayLabel = agent.activeLabel;
                foundActive = true;
            } else {
                status = 'pending';
                displayLabel = agent.label;
            }

            const isLast = i === this.agents.length - 1;

            return `
                <div class="pipeline-node ${status}" id="node-${agent.key}">
                    <span class="node-icon">${agent.icon}</span>
                    <span class="node-label">${displayLabel}</span>
                    ${status === 'completed' ? `<span class="node-check">✓</span>` : ''}
                    ${status === 'failed' ? `<span class="node-fail-tag">Error</span>` : ''}
                </div>
                ${!isLast ? `<div class="pipeline-connector ${status}"></div>` : ''}
            `;
        }).join('');
    },

    /**
     * Build loading horizontal pipeline with initial active planner node.
     */
    buildLoadingPipeline(activeKey = 'planner') {
        return this.buildPipeline([], activeKey);
    },

    /**
     * Create a cache status badge.
     */
    cacheBadge(status) {
        const isHit = status === 'HIT';
        return `<span class="meta-badge ${isHit ? 'hit' : 'miss'}">${status}</span>`;
    },

    /**
     * Create a verified badge.
     */
    verifiedBadge(verified) {
        return `<span class="meta-badge ${verified ? 'verified' : 'unverified'}">${verified ? '✓ Yes' : '✗ No'}</span>`;
    },

    /**
     * Create a model indicator badge.
     */
    modelBadge(modelName) {
        const shortName = modelName
            .replace('gemini-', '')
            .replace('gemma-', 'gemma ')
            .replace('-', ' ');
        return shortName || '—';
    },

    /**
     * Format latency value.
     */
    formatLatency(ms) {
        if (ms < 1000) return `${Math.round(ms)}ms`;
        return `${(ms / 1000).toFixed(1)}s`;
    },

    /**
     * Format large numbers.
     */
    formatNumber(n) {
        if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
        if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
        return String(n);
    },

    /**
     * Create dataset info chips.
     */
    datasetChips(info) {
        const chips = [
            `${info.rows} rows`,
            `${info.columns} columns`,
            ...info.column_names.slice(0, 6).map(c => c),
        ];
        if (info.column_names.length > 6) {
            chips.push(`+${info.column_names.length - 6} more`);
        }
        return chips.map(c => `<span class="info-chip">${c}</span>`).join('');
    },

    /**
     * Build table row for recent requests.
     */
    requestRow(record) {
        const model = this.modelBadge(record.model_selected || '—');
        const cacheClass = record.cache_hit ? 'badge-success' : 'badge-warning';
        const cacheText = record.cache_hit ? 'HIT' : 'MISS';
        const statusClass = record.success ? 'badge-success' : 'badge-error';
        const statusText = record.success ? 'OK' : 'FAIL';

        return `
            <tr>
                <td title="${record.question}">${record.question.slice(0, 40)}${record.question.length > 40 ? '...' : ''}</td>
                <td><span class="badge badge-info">${model}</span></td>
                <td><span class="badge badge-info">${record.complexity || '—'}</span></td>
                <td><span class="badge ${cacheClass}">${cacheText}</span></td>
                <td>${this.formatLatency(record.total_latency_ms)}</td>
                <td>${record.tool_calls_count || 0}</td>
                <td><span class="badge ${statusClass}">${statusText}</span></td>
            </tr>
        `;
    },
};

window.components = components;
