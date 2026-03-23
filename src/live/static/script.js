/* =====================================================
   FAV-ASTCL Dashboard — Client Script
   ===================================================== */

const API_URL = "/api/predictions";
const REFRESH_INTERVAL = 30; // seconds

let countdown = REFRESH_INTERVAL;
let countdownTimer = null;

// =====================================================
// CLOCK
// =====================================================

function updateClock() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, "0");
    const m = String(now.getMinutes()).padStart(2, "0");
    const s = String(now.getSeconds()).padStart(2, "0");
    document.getElementById("clock").textContent = `${h}:${m}:${s}`;
}

setInterval(updateClock, 1000);
updateClock();

// =====================================================
// COUNTDOWN
// =====================================================

function resetCountdown() {
    countdown = REFRESH_INTERVAL;
    if (countdownTimer) clearInterval(countdownTimer);
    countdownTimer = setInterval(() => {
        countdown--;
        const el = document.getElementById("countdown");
        if (el) el.textContent = countdown;
        if (countdown <= 0) {
            countdown = REFRESH_INTERVAL;
            fetchData();
        }
    }, 1000);
}

// =====================================================
// STATUS HELPERS
// =====================================================

function getStatusClass(status) {
    return status === "CONGESTED" ? "congested" : "normal";
}

function getStatusEmoji(status) {
    return status === "CONGESTED" ? "🚨" : "🟢";
}

function getRatioColor(ratio) {
    if (ratio < 0.5) return "red";
    if (ratio < 0.7) return "amber";
    return "green";
}

function getSpeedColor(status) {
    return status === "CONGESTED" ? "var(--red)" : "var(--green)";
}

function getTrend(forecast, current) {
    const diff = forecast - current;
    if (diff > 2)  return { arrow: "▲", cls: "trend-up" };
    if (diff < -2) return { arrow: "▼", cls: "trend-down" };
    return { arrow: "—", cls: "trend-flat" };
}

// =====================================================
// RENDER CONTEXT BAR
// =====================================================

function renderContext(data) {
    document.getElementById("ctx-temp").textContent = `${data.weather.temperature}°C`;
    document.getElementById("ctx-humidity").textContent = `${data.weather.humidity}%`;
    document.getElementById("ctx-wind").textContent = `${data.weather.wind_speed} km/h`;
    document.getElementById("ctx-hour").textContent = `${String(data.time_features.hour).padStart(2, "0")}:00`;
    document.getElementById("ctx-day").textContent = data.time_features.weekday_name;
    document.getElementById("ctx-weekend").textContent = data.time_features.is_weekend ? "Yes" : "No";
}

// =====================================================
// RENDER SENSOR CARDS
// =====================================================

function renderSensors(sensors) {
    const grid = document.getElementById("sensor-grid");
    const loading = document.getElementById("loading-state");

    if (loading) loading.classList.add("hidden");

    // Preserve existing cards for smooth updates
    const existingCards = {};
    grid.querySelectorAll(".sensor-card").forEach(card => {
        existingCards[card.dataset.sensorId] = card;
    });

    sensors.forEach((s, i) => {
        const statusCls = getStatusClass(s.congestion_status);
        const ratioColor = getRatioColor(s.congestion_ratio);
        const ratioPercent = Math.min(s.congestion_ratio * 100, 100);

        const t5  = getTrend(s.forecast["5_min"], s.current_speed);
        const t10 = getTrend(s.forecast["10_min"], s.current_speed);
        const t15 = getTrend(s.forecast["15_min"], s.current_speed);

        const html = `
            <div class="card-header">
                <div>
                    <div class="card-id">${s.sensor_id}</div>
                    <div class="card-name">📍 ${s.name}</div>
                    <div class="card-coords">${s.latitude.toFixed(4)}, ${s.longitude.toFixed(4)}</div>
                </div>
                <span class="status-badge ${statusCls}">
                    ${getStatusEmoji(s.congestion_status)} ${s.congestion_status}
                </span>
            </div>

            <div class="speed-section">
                <span class="speed-value" style="color:${getSpeedColor(s.congestion_status)}">${s.current_speed}</span>
                <span class="speed-unit">km/h</span>
                <span class="free-flow">Free flow: ${s.free_flow_speed} km/h</span>
            </div>

            <div class="ratio-bar-container">
                <div class="ratio-label">
                    <span>Congestion Ratio</span>
                    <span>${(s.congestion_ratio * 100).toFixed(1)}%</span>
                </div>
                <div class="ratio-bar">
                    <div class="ratio-fill ${ratioColor}" style="width:${ratioPercent}%"></div>
                </div>
            </div>

            <hr class="card-divider">

            <div class="forecast-title">Speed Forecast</div>
            <div class="forecast-grid">
                <div class="forecast-item">
                    <div class="forecast-horizon">+5 min</div>
                    <div class="forecast-speed">${s.forecast["5_min"]}</div>
                    <div class="forecast-trend ${t5.cls}">${t5.arrow}</div>
                </div>
                <div class="forecast-item">
                    <div class="forecast-horizon">+10 min</div>
                    <div class="forecast-speed">${s.forecast["10_min"]}</div>
                    <div class="forecast-trend ${t10.cls}">${t10.arrow}</div>
                </div>
                <div class="forecast-item">
                    <div class="forecast-horizon">+15 min</div>
                    <div class="forecast-speed">${s.forecast["15_min"]}</div>
                    <div class="forecast-trend ${t15.cls}">${t15.arrow}</div>
                </div>
            </div>
        `;

        if (existingCards[s.sensor_id]) {
            // Update existing card
            const card = existingCards[s.sensor_id];
            card.className = `sensor-card status-${statusCls}`;
            card.innerHTML = html;
            // flash animation
            card.querySelectorAll(".speed-value, .forecast-speed").forEach(el => {
                el.classList.add("flash-update");
                setTimeout(() => el.classList.remove("flash-update"), 500);
            });
            delete existingCards[s.sensor_id];
        } else {
            // Create new card
            const card = document.createElement("div");
            card.className = `sensor-card status-${statusCls}`;
            card.dataset.sensorId = s.sensor_id;
            card.style.animationDelay = `${i * 0.08}s`;
            card.innerHTML = html;
            grid.appendChild(card);
        }
    });

    // Remove any stale cards
    Object.values(existingCards).forEach(card => card.remove());
}

// =====================================================
// FETCH DATA
// =====================================================

async function fetchData() {
    try {
        const res = await fetch(API_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        renderContext(data);
        renderSensors(data.sensors);

        // Update footer
        const ts = new Date(data.timestamp);
        document.getElementById("last-updated").textContent = ts.toLocaleTimeString();

        resetCountdown();
    } catch (err) {
        console.error("Fetch error:", err);
        // Don't clear existing data on error — just retry
    }
}

// =====================================================
// INIT
// =====================================================

fetchData();
