/**
 * Determine the backend API URL dynamically.
 */
function getApiBase() {
    const proto = window.location.protocol;
    const host = window.location.hostname;
    const port = window.location.port;

    // If port is 8000 (served directly from FastAPI)
    if (port === '8000') {
        return '';
    }

    // Direct file opening (file://) or missing hostname
    if (proto === 'file:' || !host) {
        return 'http://localhost:8000';
    }

    // If served via Live Server (5500), Vite (5173), Next.js (3000), etc.
    if (port && port !== '8000') {
        const httpProto = proto === 'https:' ? 'https:' : 'http:';
        return `${httpProto}//${host}:8000`;
    }

    return ''; // Same-origin in production (e.g. Render / Docker behind reverse proxy)
}

const API_BASE = getApiBase();

const api = {
    /**
     * Submit an analytical query.
     */
    async query(question, datasetId, useCache = true) {
        const url = `${API_BASE}/api/query`;
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                dataset_id: datasetId,
                use_cache: useCache,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Query failed');
        }
        return res.json();
    },

    /**
     * Upload a CSV dataset.
     */
    async uploadDataset(file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Upload failed');
        }
        return res.json();
    },

    /**
     * List available datasets.
     */
    async getDatasets() {
        const res = await fetch(`${API_BASE}/api/datasets`);
        if (!res.ok) throw new Error('Failed to fetch datasets');
        return res.json();
    },

    /**
     * Get metrics summary.
     */
    async getMetrics() {
        const res = await fetch(`${API_BASE}/api/metrics`);
        if (!res.ok) throw new Error('Failed to fetch metrics');
        return res.json();
    },

    /**
     * Get recent request records with detailed metrics.
     */
    async getRecentMetrics() {
        const res = await fetch(`${API_BASE}/api/metrics/recent`);
        if (!res.ok) throw new Error('Failed to fetch recent metrics');
        return res.json();
    },

    /**
     * Health check.
     */
    async health() {
        const res = await fetch(`${API_BASE}/api/health`);
        if (!res.ok) throw new Error('Backend unavailable');
        return res.json();
    },

    /**
     * Clear all caches.
     */
    async clearCache() {
        const res = await fetch(`${API_BASE}/api/cache/clear`, { method: 'POST' });
        return res.json();
    },
};

/**
 * WebSocket connection for live pipeline updates.
 */
class PipelineSocket {
    constructor() {
        this.ws = null;
        this.listeners = [];
        this.reconnectDelay = 2000;
        this.maxReconnectDelay = 30000;
    }

    connect() {
        let url;
        if (API_BASE.startsWith('http')) {
            url = API_BASE.replace(/^http/, 'ws') + '/ws/query';
        } else {
            const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            url = `${proto}//${window.location.host}/ws/query`;
        }

        try {
            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                this.reconnectDelay = 2000;
                this._emit({ type: 'connected' });
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this._emit(data);
                } catch (e) {
                    console.warn('Invalid WS message:', event.data);
                }
            };

            this.ws.onclose = () => {
                this._emit({ type: 'disconnected' });
                this._reconnect();
            };

            this.ws.onerror = () => {
                this._emit({ type: 'error' });
            };
        } catch (e) {
            console.warn('WebSocket connection failed:', e);
            this._reconnect();
        }
    }

    onMessage(callback) {
        this.listeners.push(callback);
    }

    _emit(data) {
        this.listeners.forEach(cb => cb(data));
    }

    _reconnect() {
        setTimeout(() => {
            this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
            this.connect();
        }, this.reconnectDelay);
    }
}

// Export
window.api = api;
window.PipelineSocket = PipelineSocket;
