// The WebSocket endpoint on the host the page came from. A page served over HTTPS has to use wss://:
// the browser blocks a plain ws:// connection from a secure page as mixed content.
function defaultUrl() {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/ws`;
}

export class WebSocketClient {
  constructor(url) {
    this.url = url || defaultUrl();
    this.ws = null;
    this.listeners = new Map();
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.reconnectTimer = null;
    this.intentionalDisconnect = false;
    this.hasConnected = false;
  }

  isConnected() {
    return (
      this.ws !== null &&
      (this.ws.readyState === 1 ||
        this.ws.readyState === (typeof WebSocket !== "undefined" ? WebSocket.OPEN : 1))
    );
  }

  connect() {
    // Clear intentional disconnect flag when explicitly connecting
    this.intentionalDisconnect = false;

    // Clear any pending reconnect timer so multiple connects don't race
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        const isReconnect = this.hasConnected;
        this.hasConnected = true;
        this.reconnectDelay = 1000;

        const openHandlers = this.listeners.get("open") || [];
        openHandlers.forEach((handler) => handler({ isReconnect }));

        if (isReconnect) {
          const reconnectHandlers = this.listeners.get("reconnect") || [];
          reconnectHandlers.forEach((handler) => handler());
        }

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
        // Notify listeners so the application can clean up in-flight state
        const closeHandlers = this.listeners.get("close") || [];
        closeHandlers.forEach((handler) => handler());

        // Only reconnect if this was NOT an intentional disconnect
        if (!this.intentionalDisconnect) {
          this.reconnectTimer = setTimeout(() => {
            this.connect().catch(() => {});
          }, this.reconnectDelay);
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
    if (this.isConnected()) {
      this.ws.send(JSON.stringify(data));
      return true;
    }
    return false;
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
    this.hasConnected = false;

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
