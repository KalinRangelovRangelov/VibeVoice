/**
 * VibeVoice TTS Web Application
 * Frontend JavaScript for text-to-speech interface
 */

class VibeVoiceApp {
    constructor() {
        // DOM Elements
        this.textInput = document.getElementById('textInput');
        this.charCount = document.getElementById('charCount');
        this.clearBtn = document.getElementById('clearBtn');
        this.synthesizeBtn = document.getElementById('synthesizeBtn');
        this.sampleBtns = document.querySelectorAll('.sample-btn');

        this.statusBadge = document.getElementById('statusBadge');
        this.voiceSelect = document.getElementById('voiceSelect');
        this.outputSection = document.getElementById('outputSection');
        this.loadingState = document.getElementById('loadingState');
        this.audioPlayer = document.getElementById('audioPlayer');

        this.playBtn = document.getElementById('playBtn');
        this.progressBar = document.getElementById('progressBar');
        this.progressFill = document.getElementById('progressFill');
        this.progressHandle = document.getElementById('progressHandle');
        this.currentTimeEl = document.getElementById('currentTime');
        this.durationEl = document.getElementById('duration');
        this.downloadBtn = document.getElementById('downloadBtn');

        this.waveformCanvas = document.getElementById('waveformCanvas');
        this.waveformCtx = this.waveformCanvas.getContext('2d');

        this.historySection = document.getElementById('history');
        this.historyList = document.getElementById('historyList');

        this.audioElement = document.getElementById('audioElement');

        // State
        this.currentAudioUrl = null;
        this.history = [];
        this.isPlaying = false;
        this.statusCheckInterval = null;

        // Initialize
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkStatus();
        this.loadVoices();
        this.resizeCanvas();

        // Check status frequently while loading, less often when ready
        this.startStatusPolling();

        // Handle window resize
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    async loadVoices() {
        try {
            const response = await fetch('/api/voices');
            const voices = await response.json();

            this.voiceSelect.innerHTML = voices.map(voice =>
                `<option value="${voice.id}">${voice.name} (${voice.gender}, ${voice.accent})</option>`
            ).join('');
        } catch (error) {
            console.error('Failed to load voices:', error);
        }
    }

    startStatusPolling() {
        // Clear existing interval
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }
        // Poll every 2 seconds while loading
        this.statusCheckInterval = setInterval(() => this.checkStatus(), 2000);
    }

    bindEvents() {
        // Text input
        this.textInput.addEventListener('input', () => this.updateCharCount());
        this.clearBtn.addEventListener('click', () => this.clearText());

        // Sample texts
        this.sampleBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.textInput.value = btn.dataset.text;
                this.updateCharCount();
            });
        });

        // Synthesize
        this.synthesizeBtn.addEventListener('click', () => this.synthesize());

        // Keyboard shortcut
        this.textInput.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                this.synthesize();
            }
        });

        // Audio controls
        this.playBtn.addEventListener('click', () => this.togglePlay());
        this.progressBar.addEventListener('click', (e) => this.seek(e));
        this.downloadBtn.addEventListener('click', () => this.download());

        // Audio element events
        this.audioElement.addEventListener('timeupdate', () => this.updateProgress());
        this.audioElement.addEventListener('ended', () => this.onAudioEnded());
        this.audioElement.addEventListener('loadedmetadata', () => this.onAudioLoaded());
    }

    async checkStatus() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingMessage = document.getElementById('loadingMessage');

        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            this.statusBadge.classList.remove('ready', 'error');

            if (data.model_loaded) {
                this.statusBadge.classList.add('ready');
                this.statusBadge.querySelector('.status-text').textContent =
                    `Ready (${data.device.toUpperCase()})`;
                this.synthesizeBtn.disabled = false;

                // Hide loading overlay
                if (loadingOverlay) {
                    loadingOverlay.classList.add('hidden');
                }

                // Slow down polling when ready
                if (this.statusCheckInterval) {
                    clearInterval(this.statusCheckInterval);
                    this.statusCheckInterval = setInterval(() => this.checkStatus(), 30000);
                }
            } else {
                // Show loading status
                this.statusBadge.querySelector('.status-text').textContent = 'Loading...';
                this.synthesizeBtn.disabled = true;

                // Show loading overlay with message
                if (loadingOverlay) {
                    loadingOverlay.classList.remove('hidden');
                }
                if (loadingMessage) {
                    loadingMessage.textContent = data.loading_message || 'Loading model...';
                }
            }
        } catch (error) {
            this.statusBadge.classList.add('error');
            this.statusBadge.querySelector('.status-text').textContent = 'Disconnected';
            this.synthesizeBtn.disabled = true;

            // Show error in overlay
            if (loadingOverlay) {
                loadingOverlay.classList.remove('hidden');
            }
            if (loadingMessage) {
                loadingMessage.textContent = 'Connecting to server...';
            }
        }
    }

    updateCharCount() {
        const count = this.textInput.value.length;
        this.charCount.textContent = count;

        if (count > 4500) {
            this.charCount.style.color = 'var(--warning)';
        } else if (count >= 5000) {
            this.charCount.style.color = 'var(--error)';
        } else {
            this.charCount.style.color = '';
        }
    }

    clearText() {
        this.textInput.value = '';
        this.updateCharCount();
        this.textInput.focus();
    }

    async synthesize() {
        const text = this.textInput.value.trim();

        if (!text) {
            this.textInput.focus();
            return;
        }

        // Show output section and loading state
        this.outputSection.classList.add('visible');
        this.loadingState.classList.add('active');
        this.audioPlayer.classList.remove('active');
        this.synthesizeBtn.disabled = true;

        try {
            const voice = this.voiceSelect.value;

            const response = await fetch('/api/synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, voice })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Synthesis failed');
            }

            const data = await response.json();

            // Add to history
            this.addToHistory(text, data.audio_url, data.audio_id, data.duration);

            // Load and play audio
            this.loadAudio(data.audio_url, text);

        } catch (error) {
            console.error('Synthesis error:', error);
            alert(`Error: ${error.message}`);
        } finally {
            this.loadingState.classList.remove('active');
            this.synthesizeBtn.disabled = false;
        }
    }

    loadAudio(url, text) {
        this.currentAudioUrl = url;
        this.audioElement.src = url;
        this.audioPlayer.classList.add('active');

        // Auto-play after loading
        this.audioElement.addEventListener('canplaythrough', () => {
            this.audioElement.play();
            this.isPlaying = true;
            this.playBtn.classList.add('playing');
        }, { once: true });
    }

    togglePlay() {
        if (this.isPlaying) {
            this.audioElement.pause();
            this.isPlaying = false;
            this.playBtn.classList.remove('playing');
        } else {
            this.audioElement.play();
            this.isPlaying = true;
            this.playBtn.classList.add('playing');
        }
    }

    seek(e) {
        const rect = this.progressBar.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        this.audioElement.currentTime = percent * this.audioElement.duration;
    }

    updateProgress() {
        const current = this.audioElement.currentTime;
        const duration = this.audioElement.duration || 0;
        const percent = duration ? (current / duration) * 100 : 0;

        this.progressFill.style.width = `${percent}%`;
        this.progressHandle.style.left = `${percent}%`;
        this.currentTimeEl.textContent = this.formatTime(current);

        // Update waveform visualization
        this.drawWaveform(percent);
    }

    onAudioLoaded() {
        this.durationEl.textContent = this.formatTime(this.audioElement.duration);
        this.drawWaveform(0);
    }

    onAudioEnded() {
        this.isPlaying = false;
        this.playBtn.classList.remove('playing');
        this.progressFill.style.width = '0%';
        this.progressHandle.style.left = '0%';
        this.audioElement.currentTime = 0;
    }

    formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    download() {
        if (!this.currentAudioUrl) return;

        const a = document.createElement('a');
        a.href = this.currentAudioUrl;
        a.download = 'vibevoice-audio.wav';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    addToHistory(text, audioUrl, audioId, duration) {
        const item = { text, audioUrl, audioId, duration, timestamp: Date.now() };
        this.history.unshift(item);

        // Keep only last 10 items
        if (this.history.length > 10) {
            const removed = this.history.pop();
            // Delete old file from server
            if (removed.audioId) {
                this.deleteAudio(removed.audioId, false);
            }
        }

        this.renderHistory();
    }

    async deleteAudio(audioId, removeFromHistory = true) {
        try {
            if (audioId && audioId !== 'undefined' && audioId !== '') {
                await fetch(`/api/audio/${audioId}.wav`, { method: 'DELETE' });
            }

            if (removeFromHistory) {
                // Find index by audioId or remove first item if no audioId
                if (audioId && audioId !== 'undefined' && audioId !== '') {
                    this.history = this.history.filter(item => item.audioId !== audioId);
                }
                this.renderHistory();

                // Hide section if no more items
                if (this.history.length === 0) {
                    this.historySection.classList.remove('visible');
                }
            }
        } catch (error) {
            console.error('Failed to delete audio:', error);
        }
    }

    renderHistory() {
        if (this.history.length === 0) {
            this.historySection.classList.remove('visible');
            return;
        }

        this.historySection.classList.add('visible');
        this.historyList.innerHTML = this.history.map((item, index) => `
            <div class="history-item" data-index="${index}" data-audio-id="${item.audioId || ''}">
                <div class="history-item-play">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                </div>
                <div class="history-item-text">${this.escapeHtml(item.text)}</div>
                <div class="history-item-duration">${this.formatTime(item.duration)}</div>
                <button type="button" class="history-item-delete" title="Delete" onclick="event.stopPropagation(); window.app.deleteAudio('${item.audioId || ''}', true);">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                    </svg>
                </button>
            </div>
        `).join('');

        // Bind play click events
        this.historyList.querySelectorAll('.history-item').forEach(el => {
            el.addEventListener('click', (e) => {
                // Don't trigger if clicking delete button
                if (e.target.closest('.history-item-delete')) return;

                const index = parseInt(el.dataset.index);
                const item = this.history[index];
                this.loadAudio(item.audioUrl, item.text);
            });
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    resizeCanvas() {
        const rect = this.waveformCanvas.parentElement.getBoundingClientRect();
        this.waveformCanvas.width = rect.width * window.devicePixelRatio;
        this.waveformCanvas.height = rect.height * window.devicePixelRatio;
        this.waveformCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.drawWaveform(0);
    }

    drawWaveform(progressPercent) {
        const canvas = this.waveformCanvas;
        const ctx = this.waveformCtx;
        const width = canvas.width / window.devicePixelRatio;
        const height = canvas.height / window.devicePixelRatio;

        ctx.clearRect(0, 0, width, height);

        // Generate pseudo-random waveform based on current audio
        const bars = 80;
        const barWidth = width / bars;
        const gap = 2;

        for (let i = 0; i < bars; i++) {
            // Create varied bar heights
            const seed = Math.sin(i * 0.5) * 0.5 + Math.cos(i * 0.3) * 0.3 + 0.5;
            const barHeight = seed * (height * 0.8);

            const x = i * barWidth;
            const y = (height - barHeight) / 2;

            // Color based on progress
            const barProgress = (i / bars) * 100;

            if (barProgress < progressPercent) {
                const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
                gradient.addColorStop(0, '#6366f1');
                gradient.addColorStop(1, '#a855f7');
                ctx.fillStyle = gradient;
            } else {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
            }

            ctx.beginPath();
            ctx.roundRect(x + gap/2, y, barWidth - gap, barHeight, 2);
            ctx.fill();
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new VibeVoiceApp();
});
