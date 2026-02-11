/**
 * ChargeMate PWA - Frontend Application
 * 충전 요청 UI 및 로봇 상태 표시
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_BASE = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws`;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let ws = null;
let wsReconnectInterval = null;
let currentRequest = null;
let userLocation = null;
let map = null;
let robotMarker = null;
let userMarker = null;

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const elements = {
  connectionStatus: document.getElementById('connection-status'),
  robotState: document.getElementById('robot-state'),
  robotIcon: document.getElementById('robot-icon'),
  robotSoc: document.getElementById('robot-soc'),
  outputSoc: document.getElementById('output-soc'),
  robotBatteryBar: document.getElementById('robot-battery-bar'),
  outputBatteryBar: document.getElementById('output-battery-bar'),
  currentRequestId: document.getElementById('current-request-id'),
  stateDuration: document.getElementById('state-duration'),
  queuePending: document.getElementById('queue-pending'),
  queueCompleted: document.getElementById('queue-completed'),
  requestList: document.getElementById('request-list'),
  requestBtn: document.getElementById('request-btn'),
  emergencyBtn: document.getElementById('emergency-btn'),
  userName: document.getElementById('user-name'),
  deviceType: document.getElementById('device-type'),
  locationText: document.getElementById('location-text'),
  toast: document.getElementById('toast'),
  modal: document.getElementById('modal'),
  myRequestCard: document.getElementById('my-request-card'),
};

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------
function showToast(message, type = 'info') {
  elements.toast.textContent = message;
  elements.toast.className = `toast show ${type}`;
  setTimeout(() => {
    elements.toast.classList.remove('show');
  }, 3000);
}

function formatDuration(seconds) {
  if (!seconds) return '0s';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}s`;
}

function getStateIcon(state) {
  const icons = {
    'IDLE': '🔋',
    'DISPATCHING': '🚀',
    'NAVIGATING': '🚗',
    'APPROACHING_USER': '👋',
    'DOCKING': '🔌',
    'CHARGING': '⚡',
    'UNDOCKING': '🔓',
    'RETURNING': '🏠',
    'EMERGENCY_STOP': '🛑',
  };
  return icons[state] || '🤖';
}

function getStateText(state) {
  const texts = {
    'IDLE': '대기 중',
    'DISPATCHING': '출발 준비',
    'NAVIGATING': '이동 중',
    'APPROACHING_USER': '사용자 접근 중',
    'DOCKING': '연결 대기',
    'CHARGING': '충전 중',
    'UNDOCKING': '분리 대기',
    'RETURNING': '복귀 중',
    'EMERGENCY_STOP': '비상 정지',
  };
  return texts[state] || state;
}

// ---------------------------------------------------------------------------
// WebSocket Connection
// ---------------------------------------------------------------------------
function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    return;
  }

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('WebSocket connected');
    updateConnectionStatus(true);
    clearInterval(wsReconnectInterval);
    wsReconnectInterval = null;

    // Start heartbeat
    setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 10000);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleWSMessage(data);
    } catch (e) {
      console.error('Failed to parse WS message:', e);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected');
    updateConnectionStatus(false);
    scheduleReconnect();
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    updateConnectionStatus(false);
  };
}

function scheduleReconnect() {
  if (!wsReconnectInterval) {
    wsReconnectInterval = setInterval(() => {
      console.log('Attempting WebSocket reconnect...');
      connectWebSocket();
    }, 5000);
  }
}

function updateConnectionStatus(connected) {
  if (elements.connectionStatus) {
    elements.connectionStatus.className = `status-badge ${connected ? 'connected' : 'disconnected'}`;
    elements.connectionStatus.innerHTML = `
      <span class="status-dot"></span>
      ${connected ? '연결됨' : '연결 끊김'}
    `;
  }
}

function handleWSMessage(data) {
  switch (data.type) {
    case 'robot_update':
      updateRobotStatus(data.data);
      break;
    case 'request_update':
      updateRequestStatus(data.data);
      break;
  }
}

// ---------------------------------------------------------------------------
// Robot Status Updates
// ---------------------------------------------------------------------------
function updateRobotStatus(status) {
  const state = status.status || 'IDLE';

  // State display
  if (elements.robotState) {
    elements.robotState.textContent = getStateText(state);
    elements.robotState.className = `robot-state ${state.toLowerCase()}`;
  }

  if (elements.robotIcon) {
    elements.robotIcon.textContent = getStateIcon(state);
  }

  // Battery displays
  if (elements.robotSoc) {
    const soc = status.robot_soc || 0;
    elements.robotSoc.textContent = `${Math.round(soc)}%`;
    elements.robotSoc.className = soc < 20 ? 'info-value low' : 'info-value';
  }

  if (elements.robotBatteryBar) {
    const soc = status.robot_soc || 0;
    elements.robotBatteryBar.style.width = `${soc}%`;
    elements.robotBatteryBar.className = `battery-fill ${soc < 10 ? 'critical' : soc < 20 ? 'low' : ''}`;
  }

  if (elements.outputSoc) {
    const soc = status.output_soc || 0;
    elements.outputSoc.textContent = `${Math.round(soc)}%`;
  }

  if (elements.outputBatteryBar) {
    const soc = status.output_soc || 0;
    elements.outputBatteryBar.style.width = `${soc}%`;
  }

  // Update request button state
  if (elements.requestBtn) {
    const canRequest = state === 'IDLE';
    elements.requestBtn.disabled = !canRequest;
    elements.requestBtn.textContent = canRequest ? '충전 요청하기' : '로봇 사용 중...';
  }

  // Update map marker
  if (robotMarker && status.latitude && status.longitude) {
    robotMarker.setLatLng([status.latitude, status.longitude]);
  }
}

// ---------------------------------------------------------------------------
// Request Management
// ---------------------------------------------------------------------------
async function createChargeRequest() {
  const name = elements.userName?.value?.trim() || '익명';
  const deviceType = elements.deviceType?.value || 'phone';

  if (!userLocation) {
    showToast('위치 정보를 가져오는 중...', 'info');
    await getUserLocation();
    if (!userLocation) {
      showToast('위치를 가져올 수 없습니다', 'error');
      return;
    }
  }

  try {
    const response = await fetch(`${API_BASE}/api/request`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: userLocation.latitude,
        longitude: userLocation.longitude,
        user_name: name,
        device_type: deviceType,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || '요청 실패');
    }

    const data = await response.json();
    currentRequest = data;
    saveCurrentRequest(data);
    showToast('충전 요청이 접수되었습니다!', 'success');
    updateMyRequestCard(data);
    closeModal();
    fetchQueueStatus();
  } catch (error) {
    console.error('Request failed:', error);
    showToast(error.message, 'error');
  }
}

async function cancelRequest(requestId) {
  try {
    const response = await fetch(`${API_BASE}/api/request/${requestId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error('취소 실패');
    }

    currentRequest = null;
    localStorage.removeItem('chargemate_request');
    showToast('요청이 취소되었습니다', 'info');
    updateMyRequestCard(null);
    fetchQueueStatus();
  } catch (error) {
    console.error('Cancel failed:', error);
    showToast(error.message, 'error');
  }
}

function updateRequestStatus(data) {
  if (currentRequest && currentRequest.request_id === data.request_id) {
    currentRequest.status = data.status;
    currentRequest.queue_position = data.queue_position;
    updateMyRequestCard(currentRequest);

    if (data.status === 'COMPLETED') {
      showToast('충전이 완료되었습니다!', 'success');
      currentRequest = null;
      localStorage.removeItem('chargemate_request');
    } else if (data.status === 'CANCELLED') {
      currentRequest = null;
      localStorage.removeItem('chargemate_request');
    }
  }
}

function updateMyRequestCard(request) {
  if (!elements.myRequestCard) return;

  if (!request) {
    elements.myRequestCard.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📱</div>
        <p>활성 요청이 없습니다</p>
      </div>
    `;
    return;
  }

  const statusClass = request.status?.toLowerCase() || 'pending';
  const statusText = {
    'pending': '대기 중',
    'accepted': '수락됨',
    'in_progress': '진행 중',
    'completed': '완료',
    'cancelled': '취소됨',
  }[statusClass] || request.status;

  elements.myRequestCard.innerHTML = `
    <div class="request-card ${statusClass}">
      <div class="request-header">
        <span class="request-id">#${request.request_id}</span>
        <span class="request-status ${statusClass}">${statusText}</span>
      </div>
      <div class="request-info">
        ${request.queue_position ? `대기 순서: ${request.queue_position}번째` : ''}
        ${request.estimated_wait_minutes ? ` (약 ${Math.round(request.estimated_wait_minutes)}분)` : ''}
      </div>
      ${statusClass !== 'completed' && statusClass !== 'cancelled' ? `
        <div class="request-actions">
          <button class="btn btn-secondary btn-block" onclick="cancelRequest('${request.request_id}')">
            요청 취소
          </button>
        </div>
      ` : ''}
    </div>
  `;
}

function saveCurrentRequest(request) {
  localStorage.setItem('chargemate_request', JSON.stringify(request));
}

function loadSavedRequest() {
  try {
    const saved = localStorage.getItem('chargemate_request');
    if (saved) {
      currentRequest = JSON.parse(saved);
      updateMyRequestCard(currentRequest);
      // Verify request still exists
      checkRequestStatus(currentRequest.request_id);
    }
  } catch (e) {
    console.error('Failed to load saved request:', e);
  }
}

async function checkRequestStatus(requestId) {
  try {
    const response = await fetch(`${API_BASE}/api/request/${requestId}`);
    if (response.ok) {
      const data = await response.json();
      currentRequest = data;
      updateMyRequestCard(data);
    } else {
      // Request no longer exists
      currentRequest = null;
      localStorage.removeItem('chargemate_request');
      updateMyRequestCard(null);
    }
  } catch (e) {
    console.error('Failed to check request status:', e);
  }
}

// ---------------------------------------------------------------------------
// Queue Status
// ---------------------------------------------------------------------------
async function fetchQueueStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/queue`);
    if (!response.ok) throw new Error('Failed to fetch queue');

    const data = await response.json();
    updateQueueDisplay(data);
  } catch (error) {
    console.error('Queue fetch failed:', error);
  }
}

function updateQueueDisplay(queue) {
  if (elements.queuePending) {
    elements.queuePending.textContent = queue.total_pending || 0;
  }

  if (elements.queueCompleted) {
    elements.queueCompleted.textContent = queue.total_completed_today || 0;
  }

  if (elements.requestList) {
    if (!queue.requests || queue.requests.length === 0) {
      elements.requestList.innerHTML = `
        <div class="empty-state">
          <p>대기 중인 요청이 없습니다</p>
        </div>
      `;
      return;
    }

    elements.requestList.innerHTML = queue.requests
      .slice(0, 5)
      .map(req => {
        const statusClass = req.status?.toLowerCase() || 'pending';
        return `
          <div class="request-card ${statusClass}">
            <div class="request-header">
              <span class="request-id">#${req.request_id?.slice(0, 8)}...</span>
              <span class="request-status ${statusClass}">${req.status}</span>
            </div>
            <div class="request-info">
              ${req.user_name || '익명'} - ${req.device_type || '기기'}
            </div>
          </div>
        `;
      })
      .join('');
  }
}

// ---------------------------------------------------------------------------
// Robot Status
// ---------------------------------------------------------------------------
async function fetchRobotStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/robot/status`);
    if (!response.ok) throw new Error('Failed to fetch robot status');

    const data = await response.json();
    updateRobotStatus(data);
  } catch (error) {
    console.error('Robot status fetch failed:', error);
  }
}

async function emergencyStop() {
  if (!confirm('비상 정지를 활성화하시겠습니까?')) {
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/api/robot/emergency-stop`, {
      method: 'POST',
    });

    if (!response.ok) throw new Error('Emergency stop failed');

    showToast('비상 정지 활성화됨', 'error');
  } catch (error) {
    console.error('Emergency stop failed:', error);
    showToast('비상 정지 실패', 'error');
  }
}

// ---------------------------------------------------------------------------
// Location Services
// ---------------------------------------------------------------------------
async function getUserLocation() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) {
      showToast('이 브라우저는 위치 서비스를 지원하지 않습니다', 'error');
      resolve(null);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        userLocation = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        };
        updateLocationDisplay();
        resolve(userLocation);
      },
      (error) => {
        console.error('Geolocation error:', error);
        showToast('위치를 가져올 수 없습니다', 'error');
        resolve(null);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 60000,
      }
    );
  });
}

function updateLocationDisplay() {
  if (elements.locationText && userLocation) {
    elements.locationText.textContent = 
      `${userLocation.latitude.toFixed(4)}, ${userLocation.longitude.toFixed(4)}`;
  }

  if (userMarker && userLocation) {
    userMarker.setLatLng([userLocation.latitude, userLocation.longitude]);
  }
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------
function openModal() {
  if (elements.modal) {
    elements.modal.classList.add('show');
    getUserLocation();
  }
}

function closeModal() {
  if (elements.modal) {
    elements.modal.classList.remove('show');
  }
}

// ---------------------------------------------------------------------------
// Map (Leaflet)
// ---------------------------------------------------------------------------
function initMap() {
  const mapContainer = document.getElementById('map');
  if (!mapContainer || typeof L === 'undefined') {
    console.warn('Map container or Leaflet not available');
    return;
  }

  // Default center (Seoul)
  map = L.map('map').setView([37.5665, 126.9780], 15);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap',
    maxZoom: 19,
  }).addTo(map);

  // Robot marker
  robotMarker = L.marker([37.5665, 126.9780], {
    icon: L.divIcon({
      className: 'robot-map-marker',
      html: '🤖',
      iconSize: [30, 30],
    }),
  }).addTo(map);

  // User marker (hidden initially)
  userMarker = L.marker([0, 0], {
    icon: L.divIcon({
      className: 'user-map-marker',
      html: '📍',
      iconSize: [30, 30],
    }),
  });
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
function init() {
  // Connect WebSocket
  connectWebSocket();

  // Initial data fetch
  fetchRobotStatus();
  fetchQueueStatus();

  // Load saved request
  loadSavedRequest();

  // Get user location
  getUserLocation();

  // Initialize map if available
  initMap();

  // Set up periodic refresh
  setInterval(fetchRobotStatus, 5000);
  setInterval(fetchQueueStatus, 10000);

  // Event listeners
  if (elements.requestBtn) {
    elements.requestBtn.addEventListener('click', openModal);
  }

  if (elements.emergencyBtn) {
    elements.emergencyBtn.addEventListener('click', emergencyStop);
  }

  // Modal close on overlay click
  if (elements.modal) {
    elements.modal.addEventListener('click', (e) => {
      if (e.target === elements.modal) {
        closeModal();
      }
    });
  }

  // Form submit
  const submitBtn = document.getElementById('submit-request');
  if (submitBtn) {
    submitBtn.addEventListener('click', createChargeRequest);
  }

  const closeBtn = document.getElementById('modal-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', closeModal);
  }

  const locationBtn = document.getElementById('location-btn');
  if (locationBtn) {
    locationBtn.addEventListener('click', () => {
      getUserLocation().then(() => {
        if (userLocation && map) {
          map.setView([userLocation.latitude, userLocation.longitude], 17);
          userMarker.setLatLng([userLocation.latitude, userLocation.longitude]);
          userMarker.addTo(map);
        }
      });
    });
  }

  console.log('ChargeMate App initialized');
}

// Start app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// Expose functions for inline handlers
window.cancelRequest = cancelRequest;
window.openModal = openModal;
window.closeModal = closeModal;
window.createChargeRequest = createChargeRequest;
