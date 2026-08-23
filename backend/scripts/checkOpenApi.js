const fs = require("node:fs");
const path = require("node:path");
const { getOpenApiSpecJson } = require("../src/openapi");

const committedPath = path.join(__dirname, "..", "openapi.json");

if (!fs.existsSync(committedPath)) {
  console.error(`::error::Committed OpenAPI specification file missing at ${committedPath}`);
  console.error("Run 'npm run openapi:generate' to generate it.");
  process.exit(1);
}

const committedContent = fs.readFileSync(committedPath, "utf8");
const generatedContent = getOpenApiSpecJson();

if (committedContent !== generatedContent) {
  console.error(`::error::Committed OpenAPI specification (${committedPath}) is out of sync with the generated schema.`);
  console.error("Run 'npm run openapi:generate' in backend/ and commit the updated openapi.json file.");
  process.exit(1);
}

console.log("OpenAPI 3.1 specification is in sync with committed schema.");
