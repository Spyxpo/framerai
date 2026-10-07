/**
 * The WebSocket URL the client opens by default (Issue #435).
 *
 * The website and the backend share one origin: the Vite dev server and the nginx in the website
 * image both proxy /ws to the backend, so the client connects to the host the page came from. It
 * used to say ws:// whatever the page itself was served over, and a page served over HTTPS may not
 * open a plain ws:// connection: the browser blocks it as mixed content. useChat treats a failed
 * connection as non-fatal and falls back to REST, so an HTTPS deployment lost streaming and
 * command approval without any error on screen.
 *
 * A page served over https: now opens wss://, and every other page keeps ws://. The host, the
 * port and the /ws path are what they were, and a URL passed in is used as it was given.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { WebSocketClient } from "../services/websocket.js";

// Every socket the client asked the browser to open, in order.
let sockets;

class RecordingWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    sockets.push(this);
  }

  close() {}

  send() {}
}
Object.assign(RecordingWebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });

// Runs the client in a page served from pageUrl. A URL has the two fields the client reads from
// window.location, protocol and host, and the same rules for them (no default port in the host).
function servePageFrom(pageUrl) {
  vi.stubGlobal("location", new URL(pageUrl));
}

beforeEach(() => {
  sockets = [];
  vi.stubGlobal("WebSocket", RecordingWebSocket);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("the protocol follows the page", () => {
  it("a page served over HTTPS connects with wss://", () => {
    servePageFrom("https://chat.example.com/");

    const client = new WebSocketClient();
    client.connect();

    expect(client.url).toBe("wss://chat.example.com/ws");
    expect(sockets.map((socket) => socket.url)).toEqual(["wss://chat.example.com/ws"]);
  });

  it("a page served over HTTP connects with ws://", () => {
    servePageFrom("http://chat.example.com/");

    const client = new WebSocketClient();
    client.connect();

    expect(client.url).toBe("ws://chat.example.com/ws");
    expect(sockets.map((socket) => socket.url)).toEqual(["ws://chat.example.com/ws"]);
  });

  it("a page that is neither HTTP nor HTTPS keeps ws://", () => {
    servePageFrom("capacitor://localhost/");

    expect(new WebSocketClient().url).toBe("ws://localhost/ws");
  });

  it("is decided for each client from the page it is created in", () => {
    servePageFrom("https://chat.example.com/");
    expect(new WebSocketClient().url).toBe("wss://chat.example.com/ws");

    servePageFrom("http://chat.example.com/");
    expect(new WebSocketClient().url).toBe("ws://chat.example.com/ws");

    servePageFrom("https://chat.example.com/");
    expect(new WebSocketClient().url).toBe("wss://chat.example.com/ws");
  });
});

describe("the rest of the default URL is unchanged", () => {
  it.each([
    // development: the Vite dev server and the nginx in the compose file, both over HTTP
    ["http://localhost:5173/", "ws://localhost:5173/ws"],
    ["http://localhost:8080/", "ws://localhost:8080/ws"],
    ["http://192.168.1.20:8080/app?x=1", "ws://192.168.1.20:8080/ws"],
    ["http://[::1]:5173/", "ws://[::1]:5173/ws"],
    // the same hosts behind TLS
    ["https://chat.example.com/", "wss://chat.example.com/ws"],
    ["https://chat.example.com:8443/", "wss://chat.example.com:8443/ws"],
    ["https://chat.example.com:443/", "wss://chat.example.com/ws"], // the default port is not part of the host
    ["https://[::1]:5173/", "wss://[::1]:5173/ws"],
    // the page the user is on does not leak into the endpoint
    ["https://chat.example.com/some/deep/page?conversation=3#top", "wss://chat.example.com/ws"],
    ["http://chat.example.com/some/deep/page?conversation=3#top", "ws://chat.example.com/ws"],
  ])("a page at %s opens %s", (pageUrl, expected) => {
    servePageFrom(pageUrl);

    expect(new WebSocketClient().url).toBe(expected);
  });

  it("keeps ws://localhost:3000/ws on jsdom's own page, which is a real Location over HTTP", () => {
    vi.unstubAllGlobals();

    expect(window.location.protocol).toBe("http:");
    expect(new WebSocketClient().url).toBe("ws://localhost:3000/ws");
  });
});

describe("a URL passed in is used as it was given", () => {
  it.each([
    ["https://chat.example.com/", "ws://backend.internal:3001/ws?token=abc"],
    ["http://localhost:5173/", "ws://backend.internal:3001/ws?token=abc"],
    ["https://chat.example.com/", "wss://api.example.com/socket?room=1&mode=x"],
    ["http://localhost:5173/", "wss://api.example.com/socket?room=1&mode=x"],
  ])("on a page at %s, %s is neither upgraded nor rewritten", (pageUrl, given) => {
    servePageFrom(pageUrl);

    const client = new WebSocketClient(given);
    client.connect();

    expect(client.url).toBe(given);
    expect(sockets.map((socket) => socket.url)).toEqual([given]);
  });
});

describe("reconnecting", () => {
  it("opens the same URL again, after the same delays", async () => {
    servePageFrom("https://chat.example.com/");
    const client = new WebSocketClient();

    const connected = client.connect();
    sockets[0].readyState = RecordingWebSocket.OPEN;
    sockets[0].onopen();
    await connected;
    expect(sockets).toHaveLength(1);

    // The network drops the connection: the first retry comes after a second.
    sockets[0].onclose();
    vi.advanceTimersByTime(999);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(2);

    // It never comes up, and the next retry waits twice as long.
    sockets[1].onclose();
    vi.advanceTimersByTime(1999);
    expect(sockets).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(3);

    expect(sockets.map((socket) => socket.url)).toEqual(Array(3).fill("wss://chat.example.com/ws"));
  });

  it("keeps retrying a connection that never came up, after the same delays", () => {
    servePageFrom("https://chat.example.com/");
    const client = new WebSocketClient();

    // The first attempt fails before it opens, as a blocked connection does.
    client.connect().catch(() => {});
    sockets[0].onerror(new Event("error"));
    sockets[0].onclose();
    vi.advanceTimersByTime(999);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(2);

    sockets[1].onclose();
    vi.advanceTimersByTime(1999);
    expect(sockets).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(3);

    expect(sockets.map((socket) => socket.url)).toEqual(Array(3).fill("wss://chat.example.com/ws"));
  });

  it("does not reconnect after the client disconnects itself", async () => {
    servePageFrom("https://chat.example.com/");
    const client = new WebSocketClient();

    const connected = client.connect();
    sockets[0].readyState = RecordingWebSocket.OPEN;
    sockets[0].onopen();
    await connected;

    const onclose = sockets[0].onclose;
    client.disconnect();
    onclose();
    vi.advanceTimersByTime(60000);

    expect(sockets).toHaveLength(1);
  });
});
