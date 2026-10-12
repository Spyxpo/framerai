/**
 * The upload routes added for voice, editing and video share the generation
 * rate limit, which is applied where the router is mounted (app.js) rather
 * than on each route. Static analysis that looks at a route on its own cannot
 * see that, so this pins it: once the budget is spent, every one of them is
 * refused before its handler or upload parser runs.
 */

process.env.RATE_LIMIT_WINDOW_MS = "60000";
process.env.RATE_LIMIT_MAX = "100";
process.env.GENERATE_RATE_LIMIT_MAX = "2";

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

mockModel();
const app = loadApp();

const ROUTES = ["voice", "verify-audio", "image-edit", "animate", "understand-video"];

test("every upload route draws on the generation budget", async () => {
  // Spend the budget. These carry no file, so the handlers refuse them, but
  // the limiter has already counted them by then.
  for (let i = 0; i < 2; i++) {
    const res = await request(app).post("/api/generate/voice");
    assert.equal(res.status, 400);
  }

  for (const route of ROUTES) {
    const res = await request(app).post(`/api/generate/${route}`);
    assert.equal(res.status, 429, `/api/generate/${route} must be rate limited`);
    assert.equal(res.body.code, "RATE_LIMITED");
  }
});
