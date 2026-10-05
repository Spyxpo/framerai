const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const fs = require("node:fs");
const path = require("node:path");

const { mockModel, loadApp } = require("./helpers");
const { generateOpenApiSpec, getOpenApiSpecJson } = require("../src/openapi");

mockModel();
const app = loadApp();

test("GET /api/openapi.json returns 200 OK and valid JSON spec", async () => {
  const res = await request(app).get("/api/openapi.json");

  assert.equal(res.status, 200);
  assert.equal(res.headers["content-type"].includes("application/json"), true);
  assert.equal(res.body.openapi, "3.1.0");
  assert.equal(res.body.info.title, "FramerAI REST API");
  assert.equal(res.body.info.version, "1.0.0");
});

// The committed file is stored in git with LF line endings, and the generator writes LF. A checkout
// with core.autocrlf=true, the Git for Windows default, leaves CRLF on disk instead, while macOS,
// Linux and CI check it out with LF. Whether the file matches the schema must not depend on how a
// clone was configured, so a copy read from disk is compared as git stores it, with LF line endings,
// and the line endings themselves are never part of the comparison (Issue #402).
function assertMatchesGenerated(onDisk, message) {
  assert.equal(onDisk.replace(/\r\n/g, "\n"), getOpenApiSpecJson(), message);
}

test("generated OpenAPI spec matches committed openapi.json file", () => {
  const committedPath = path.join(__dirname, "..", "openapi.json");
  assert.ok(fs.existsSync(committedPath), "committed openapi.json file must exist");

  const committedContent = fs.readFileSync(committedPath, "utf8");

  assertMatchesGenerated(committedContent, "committed openapi.json must match generator output");
});

test("a copy of the spec with CRLF line endings, as a Windows checkout leaves it, matches like one with LF", () => {
  // CI and macOS check the file out with LF. The CRLF form is made here, so a Windows clone
  // is covered on every platform.
  const generated = getOpenApiSpecJson();

  assertMatchesGenerated(generated, "LF line endings");
  assertMatchesGenerated(generated.replace(/\n/g, "\r\n"), "CRLF line endings");
});

test("ignoring line endings does not hide a real difference from the generated spec", () => {
  const generated = getOpenApiSpecJson();
  const retitled = generated.replace('"title": "FramerAI REST API"', '"title": "FramerAI REST API v2"');
  assert.notEqual(retitled, generated, "the text to change must exist in the spec");

  for (const [name, eol] of [["LF", "\n"], ["CRLF", "\r\n"]]) {
    const onDisk = (text) => text.replace(/\n/g, eol);

    assert.throws(() => assertMatchesGenerated(onDisk(retitled)), assert.AssertionError, `changed content, ${name}`);
    assert.throws(() => assertMatchesGenerated(onDisk(generated.trimEnd())), assert.AssertionError, `no final newline, ${name}`);
    assert.throws(() => assertMatchesGenerated(onDisk(generated + "\n")), assert.AssertionError, `extra final blank line, ${name}`);
  }
});

test("OpenAPI spec contains all required REST API routes", () => {
  const spec = generateOpenApiSpec();
  const paths = Object.keys(spec.paths);

  const expectedPaths = [
    "/health",
    "/openapi.json",
    "/chat/conversations",
    "/chat/conversations/{id}",
    "/chat/conversations/{id}/messages",
    "/generate/image",
    "/generate/video",
    "/generate/audio",
    "/generate/code",
    "/generate/understand",
    "/generate/document",
    "/generate/upload",
    "/generate/transcribe",
  ];

  for (const pathKey of expectedPaths) {
    assert.ok(paths.includes(pathKey), `OpenAPI spec missing path ${pathKey}`);
  }
});

test("OpenAPI spec includes component schemas and rate limit headers", () => {
  const spec = generateOpenApiSpec();
  const schemas = spec.components.schemas;
  const headers = spec.components.headers;

  assert.ok(schemas.ApiError, "missing ApiError schema");
  assert.ok(schemas.GenerationSettings, "missing GenerationSettings schema");
  assert.ok(schemas.ChatMessage, "missing ChatMessage schema");
  assert.ok(schemas.ConversationSummary, "missing ConversationSummary schema");
  assert.ok(schemas.Conversation, "missing Conversation schema");
  assert.ok(schemas.ImageGenerationResponse, "missing ImageGenerationResponse schema");
  assert.ok(schemas.VideoGenerationResponse, "missing VideoGenerationResponse schema");
  assert.ok(schemas.AudioGenerationResponse, "missing AudioGenerationResponse schema");
  assert.ok(schemas.CodeGenerationResponse, "missing CodeGenerationResponse schema");

  assert.ok(headers["RateLimit-Limit"], "missing RateLimit-Limit header");
  assert.ok(headers["RateLimit-Remaining"], "missing RateLimit-Remaining header");
  assert.ok(headers["RateLimit-Reset"], "missing RateLimit-Reset header");
  assert.ok(headers["Retry-After"], "missing Retry-After header");
});

test("OpenAPI spec derives validation constraints directly from route constants", () => {
  const spec = generateOpenApiSpec();
  const generateImageBody = spec.paths["/generate/image"].post.requestBody.content["application/json"].schema;
  const generateCodeBody = spec.paths["/generate/code"].post.requestBody.content["application/json"].schema;
  const chatMessageBody = spec.paths["/chat/conversations/{id}/messages"].post.requestBody.content["application/json"].schema;

  assert.ok(Array.isArray(generateImageBody.properties.aspect.enum), "aspect must be enum array");
  assert.ok(generateImageBody.properties.aspect.enum.includes("16:9"));
  assert.ok(Array.isArray(generateCodeBody.properties.language.enum), "language must be enum array");
  assert.ok(generateCodeBody.properties.language.enum.includes("python"));
  assert.equal(chatMessageBody.properties.content.maxLength, 8000);
});
