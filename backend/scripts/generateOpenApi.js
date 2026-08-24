const fs = require("node:fs");
const path = require("node:path");
const { getOpenApiSpecJson } = require("../src/openapi");

const outputPath = path.join(__dirname, "..", "openapi.json");
const content = getOpenApiSpecJson();

fs.writeFileSync(outputPath, content, "utf8");
console.log(`Generated OpenAPI 3.1 specification at ${outputPath}`);
