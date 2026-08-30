export class WebSocketClient {
  constructor(url) {
    this.url = url || `ws://${window.location.host}/ws`;
    this.ws = null;
    this.listeners = new Map();
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.reconnectTimer = null;
    this.intentionalDisconnect = false;
  }

  connect() {
    // Clear intentional disconnect flag when explicitly connecting
    this.intentionalDisconnect = false;

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectDelay = 1000;
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const handlers = this.listeners.get(data.type) || [];
          handlers.forEach((handler) => handler(data));
        } catch {
          // ignore parse errors
        }
      };

      this.ws.onclose = () => {
        // Only reconnect if this was NOT an intentional disconnect
        if (!this.intentionalDisconnect) {
          this.reconnectTimer = setTimeout(() => this.connect(), this.reconnectDelay);
          this.reconnectDelay = Math.min(
            this.reconnectDelay * 2,
            this.maxReconnectDelay
          );
        }
      };

      this.ws.onerror = (err) => reject(err);
    });
  }

  on(type, handler) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, []);
    }
    this.listeners.get(type).push(handler);
    return () => {
      const handlers = this.listeners.get(type);
      const idx = handlers.indexOf(handler);
      if (idx >= 0) handlers.splice(idx, 1);
    };
  }

  send(data) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  sendApprovalResponse(approvalId, approved, denyEverything = false) {
    this.send({
      type: "approval_response",
      approvalId,
      approved: Boolean(approved),
      denyEverything: Boolean(denyEverything),
    });
  }

  disconnect() {
    // Set flag to prevent automatic reconnect
    this.intentionalDisconnect = true;

    // Cancel any pending reconnect timer
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    // Close the WebSocket connection
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
