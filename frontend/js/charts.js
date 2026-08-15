/**
 * DataPilot — Chart Management
 * Handles Chart.js instances for the metrics dashboard.
 */

const charts = {
    modelUsageChart: null,
    latencyChart: null,

    /**
     * Initialize or update the model usage doughnut chart.
     */
    updateModelUsage(modelUsage) {
        const canvas = document.getElementById('model-usage-chart');
        if (!canvas) return;

        const labels = Object.keys(modelUsage).map(m =>
            m.replace('gemini-', '').replace('gemma-', 'gemma ')
        );
        const data = Object.values(modelUsage);

        const colors = ['#6366f1', '#818cf8', '#a78bfa', '#c4b5fd', '#e879f9'];

        if (this.modelUsageChart) {
            this.modelUsageChart.data.labels = labels;
            this.modelUsageChart.data.datasets[0].data = data;
            this.modelUsageChart.update();
            return;
        }

        this.modelUsageChart = new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: colors.slice(0, labels.length),
                    borderColor: '#0a0a1a',
                    borderWidth: 3,
                    hoverOffset: 8,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#9898b8',
                            font: { family: 'Inter', size: 11 },
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 10,
                        },
                    },
                },
            },
        });
    },

    /**
     * Initialize or update the latency line chart.
     */
    updateLatencyChart(records) {
        const canvas = document.getElementById('latency-chart');
        if (!canvas) return;

        const labels = records.map((_, i) => `#${i + 1}`);
        const data = records.map(r => r.total_latency_ms);

        if (this.latencyChart) {
            this.latencyChart.data.labels = labels;
            this.latencyChart.data.datasets[0].data = data;
            this.latencyChart.update();
            return;
        }

        this.latencyChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Latency (ms)',
                    data,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#6366f1',
                    pointBorderColor: '#0a0a1a',
                    pointBorderWidth: 2,
                    pointHoverRadius: 6,
                    borderWidth: 2.5,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    x: {
                        ticks: { color: '#6868a0', font: { family: 'Inter', size: 10 } },
                        grid: { color: 'rgba(99, 102, 241, 0.06)' },
                    },
                    y: {
                        ticks: {
                            color: '#6868a0',
                            font: { family: 'Inter', size: 10 },
                            callback: v => `${v}ms`,
                        },
                        grid: { color: 'rgba(99, 102, 241, 0.06)' },
                    },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1a1a3e',
                        titleColor: '#e8e8f0',
                        bodyColor: '#9898b8',
                        borderColor: 'rgba(99, 102, 241, 0.3)',
                        borderWidth: 1,
                        callbacks: {
                            label: ctx => `${Math.round(ctx.parsed.y)}ms`,
                        },
                    },
                },
            },
        });
    },

    /**
     * Destroy all chart instances.
     */
    destroy() {
        if (this.modelUsageChart) { this.modelUsageChart.destroy(); this.modelUsageChart = null; }
        if (this.latencyChart) { this.latencyChart.destroy(); this.latencyChart = null; }
    },
};

window.charts = charts;
