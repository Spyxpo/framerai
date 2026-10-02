const fs = require("node:fs");
const path = require("node:path");
const { getOpenApiSpecJson } = require("../src/openapi");

const committedPath = path.join(__dirname, "..", "openapi.json");

if (!fs.existsSync(committedPath)) {
  console.error(`::error::Committed OpenAPI specification file missing at ${committedPath}`);
  console.error("Run 'npm run openapi:generate' to generate it.");
  process.exit(1);
}

// Stored in git with LF line endings, like the generator's output, but a checkout with
// core.autocrlf=true (the Git for Windows default) leaves CRLF on disk. Line endings are not
// part of whether the file is in sync with the schema (Issue #402).
const committedContent = fs.readFileSync(committedPath, "utf8").replace(/\r\n/g, "\n");
const generatedContent = getOpenApiSpecJson();

if (committedContent !== generatedContent) {
  console.error(`::error::Committed OpenAPI specification (${committedPath}) is out of sync with the generated schema.`);
  console.error("Run 'npm run openapi:generate' in backend/ and commit the updated openapi.json file.");
  process.exit(1);
}

console.log("OpenAPI 3.1 specification is in sync with committed schema.");
