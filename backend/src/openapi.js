/**
 * OpenAPI 3.1 Specification Generator.
 *
 * Dynamically constructs the OpenAPI 3.1.0 document for the FramerAI REST API
 * using validation rules and constants exported from the route definitions
 * and generation settings. This ensures documentation and runtime validation
 * cannot drift.
 */

const chatRoutes = require("./routes/chat");
const generateRoutes = require("./routes/generate");
const { LIMITS } = require("./generationSettings");
const config = require("./config");

const MAX_FILE_SIZE_MB = Math.floor(config.maxFileSize / (1024 * 1024));

function generateOpenApiSpec() {
  const rateLimitHeaderRefs = {
    "RateLimit-Limit": { $ref: "#/components/headers/RateLimit-Limit" },
    "RateLimit-Remaining": { $ref: "#/components/headers/RateLimit-Remaining" },
    "RateLimit-Reset": { $ref: "#/components/headers/RateLimit-Reset" },
  };

  const errorResponseRef = (code, description) => {
    const res = {
      description,
      content: {
        "application/json": {
          schema: { $ref: "#/components/schemas/ApiError" },
        },
      },
    };
    if (code === 429) {
      res.headers = {
        ...rateLimitHeaderRefs,
        "Retry-After": { $ref: "#/components/headers/Retry-After" },
      };
    }
    return res;
  };

  return {
    openapi: "3.1.0",
    info: {
      title: "FramerAI REST API",
      version: "1.0.0",
      description:
        "OpenAPI 3.1 specification for the FramerAI backend REST API covering chat, generation, health, upload endpoints, and rate limits.",
    },
    servers: [
      {
        url: "/api",
        description: "API server base path",
      },
    ],
    paths: {
      "/health": {
        get: {
          summary: "Health check and model capabilities",
          description: "Returns backend server status, loaded model name, version, and supported capabilities.",
          responses: {
            "200": {
              description: "Server health status and capabilities",
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["status", "model", "version", "capabilities", "timestamp"],
                    properties: {
                      status: { type: "string", example: "ok" },
                      model: { type: "string", example: "FramerAI" },
                      version: { type: "string", example: "1.0.0" },
                      capabilities: {
                        type: "array",
                        items: { type: "string" },
                        example: ["text", "code", "image", "video", "audio"],
                      },
                      timestamp: { type: "string", format: "date-time" },
                    },
                  },
                },
              },
            },
          },
        },
      },
      "/pricing": {
        get: {
          summary: "Derived rate card",
          description:
            "Returns per-model rates derived from GPU cost and measured throughput plus margin. Text and code are billed per million input, cached input and output tokens; media is billed per megapixel or per second. Chat and code responses carry metadata.usage with the tokens run and their cost.",
          responses: {
            "200": {
              description: "Rate card",
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["currency", "basis", "models"],
                    properties: {
                      currency: { type: "string", example: "USD" },
                      basis: {
                        type: "object",
                        required: ["gpu_hour_usd", "margin"],
                        properties: {
                          gpu_hour_usd: { type: "number" },
                          margin: { type: "number" },
                        },
                        additionalProperties: true,
                      },
                      models: {
                        type: "object",
                        additionalProperties: {
                          type: "object",
                          properties: {
                            input_per_million: { type: "number" },
                            cached_input_per_million: { type: "number" },
                            output_per_million: { type: "number" },
                            per_megapixel: { type: "number" },
                            per_output_second: { type: "number" },
                            per_input_second: { type: "number" },
                          },
                        },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
      "/openapi.json": {
        get: {
          summary: "Get OpenAPI schema specification",
          description: "Serves the machine-readable OpenAPI 3.1 JSON document for this API.",
          responses: {
            "200": {
              description: "OpenAPI 3.1 specification document",
              content: {
                "application/json": {
                  schema: { type: "object", additionalProperties: true },
                },
              },
            },
          },
        },
      },
      "/chat/conversations": {
        get: {
          summary: "List conversations",
          description: "Returns a list of conversation summaries sorted by creation date descending.",
          responses: {
            "200": {
              description: "List of conversation summaries",
              content: {
                "application/json": {
                  schema: {
                    type: "array",
                    items: { $ref: "#/components/schemas/ConversationSummary" },
                  },
                },
              },
            },
          },
        },
        post: {
          summary: "Create conversation",
          description: "Creates a new empty chat conversation.",
          responses: {
            "200": {
              description: "Newly created conversation",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/Conversation" },
                },
              },
            },
          },
        },
      },
      "/chat/conversations/{id}": {
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            description: "Conversation UUID",
            schema: { type: "string", format: "uuid" },
          },
        ],
        get: {
          summary: "Get conversation details",
          description: "Retrieves a conversation by UUID including all messages.",
          responses: {
            "200": {
              description: "Conversation details",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/Conversation" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "404": errorResponseRef(404, "Conversation not found"),
          },
        },
        patch: {
          summary: "Update conversation",
          description: "Updates conversation title with optional expectedVersion check.",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    title: { type: "string", minLength: 1, maxLength: 200 },
                    expectedVersion: {
                      type: "integer",
                      minimum: 1,
                      description: "Expected conversation version for optimistic concurrency control",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Updated conversation",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/Conversation" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "404": errorResponseRef(404, "Conversation not found"),
            "409": errorResponseRef(409, "Version conflict"),
          },
        },
        delete: {
          summary: "Delete conversation",
          description: "Deletes a conversation by UUID.",
          responses: {
            "200": {
              description: "Deletion confirmation",
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["success"],
                    properties: {
                      success: { type: "boolean", example: true },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "404": errorResponseRef(404, "Conversation not found"),
          },
        },
      },
      "/chat/conversations/{id}/messages": {
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            description: "Conversation UUID",
            schema: { type: "string", format: "uuid" },
          },
        ],
        post: {
          summary: "Send message",
          description: "Sends a message in a conversation and generates an assistant reply.",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["content"],
                  properties: {
                    content: {
                      type: "string",
                      minLength: 1,
                      maxLength: chatRoutes.MAX_MESSAGE_LENGTH,
                    },
                    type: {
                      type: "string",
                      enum: chatRoutes.MESSAGE_TYPES,
                      default: "text",
                    },
                    attachments: {
                      type: "array",
                      items: { type: "object", additionalProperties: true },
                      maxItems: 10,
                    },
                    expectedVersion: {
                      type: "integer",
                      minimum: 1,
                      description: "Expected conversation version for optimistic concurrency control",
                    },
                    settings: { $ref: "#/components/schemas/GenerationSettings" },
                    editMessageId: {
                      type: "string",
                      format: "uuid",
                      description: "ID of an earlier user message to edit in place",
                    },
                    regenerateMessageId: {
                      type: "string",
                      format: "uuid",
                      description: "ID of an assistant message to regenerate",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Assistant reply message, with the id the user's message was stored under",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/SendMessageResponse" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "404": errorResponseRef(404, "Conversation not found"),
            "409": errorResponseRef(409, "Version conflict"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/chat/conversations/{id}/branch": {
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            description: "Parent conversation UUID to branch from",
            schema: { type: "string", format: "uuid" },
          },
        ],
        post: {
          summary: "Branch conversation",
          description: "Creates a new branched conversation up to and including the specified message.",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["messageId"],
                  properties: {
                    messageId: {
                      type: "string",
                      format: "uuid",
                      description: "UUID of the message to branch from",
                    },
                    expectedVersion: {
                      type: "integer",
                      minimum: 1,
                      description: "Expected parent conversation version for optimistic concurrency control",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Newly created branched conversation",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/Conversation" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error or message not in conversation"),
            "404": errorResponseRef(404, "Conversation not found"),
            "409": errorResponseRef(409, "Version conflict"),
          },
        },
      },
      "/chat/conversations/{id}/sync": {
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            description: "Conversation UUID to synchronize",
            schema: { type: "string", format: "uuid" },
          },
        ],
        post: {
          summary: "Differential conversation synchronization",
          description: "Reconciles divergent client and server conversation states, applying differential changes and resolving conflicts.",
          requestBody: {
            required: false,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    clientVersion: {
                      type: "integer",
                      minimum: 1,
                      description: "Known client version for optimistic concurrency and diff tracking",
                    },
                    messages: {
                      type: "array",
                      description: "Current client messages for differential comparison",
                      items: { $ref: "#/components/schemas/ChatMessage" },
                    },
                    deletedMessageIds: {
                      type: "array",
                      description: "IDs of messages deleted on client",
                      items: { type: "string" },
                    },
                    title: {
                      type: "string",
                      maxLength: 200,
                      description: "Current client conversation title",
                    },
                    titleUpdatedAt: {
                      type: "number",
                      description: "Monotonic timestamp of local title modification",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Reconciled synchronization result",
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["status", "version", "conversation"],
                    properties: {
                      status: {
                        type: "string",
                        enum: ["synchronized", "up_to_date", "client_ahead", "server_ahead"],
                      },
                      version: { type: "integer", minimum: 1 },
                      serverVersion: { type: "integer", minimum: 1 },
                      clientVersion: { type: "integer" },
                      diff: { type: "object" },
                      conversation: { $ref: "#/components/schemas/Conversation" },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "404": errorResponseRef(404, "Conversation not found"),
          },
        },
      },
      "/chat/conversations/{id}/messages/{messageId}": {
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            description: "Conversation UUID",
            schema: { type: "string", format: "uuid" },
          },
          {
            name: "messageId",
            in: "path",
            required: true,
            description: "Message UUID to delete",
            schema: { type: "string", format: "uuid" },
          },
        ],
        delete: {
          summary: "Delete conversation message",
          description: "Deletes a single message from the conversation and advances its version.",
          responses: {
            "200": {
              description: "Message deleted successfully",
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["success", "version", "messageId"],
                    properties: {
                      success: { type: "boolean", example: true },
                      version: { type: "integer", minimum: 1 },
                      messageId: { type: "string", format: "uuid" },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "404": errorResponseRef(404, "Message or conversation not found"),
          },
        },
      },
      "/generate/image": {
        post: {
          summary: "Generate image from text",
          description: "Generates one or more images matching the prompt.",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["prompt"],
                  properties: {
                    prompt: {
                      type: "string",
                      minLength: 1,
                      maxLength: generateRoutes.MAX_PROMPT_LENGTH,
                    },
                    num_images: {
                      type: "integer",
                      minimum: 1,
                      maximum: 4,
                      default: 1,
                    },
                    width: {
                      type: "integer",
                      minimum: generateRoutes.MIN_DIMENSION,
                      maximum: generateRoutes.MAX_DIMENSION,
                    },
                    height: {
                      type: "integer",
                      minimum: generateRoutes.MIN_DIMENSION,
                      maximum: generateRoutes.MAX_DIMENSION,
                    },
                    aspect: {
                      type: "string",
                      enum: generateRoutes.ASPECT_RATIOS,
                    },
                    tier: {
                      type: "integer",
                      enum: generateRoutes.SIZE_TIERS,
                    },
                    seed: {
                      type: "integer",
                      minimum: 0,
                      maximum: 2147483647,
                    },
                    resolution: {
                      type: "integer",
                      enum: generateRoutes.RESOLUTIONS,
                      description: "Deprecated square-only resolution tier alias",
                    },
                    shift: {
                      type: "number",
                      minimum: 0.1,
                      maximum: 10,
                      description: "Timestep shift; above 1 spends more steps at high noise",
                    },
                    guidance_interval: {
                      type: "array",
                      items: { type: "number", minimum: 0, maximum: 1 },
                      minItems: 2,
                      maxItems: 2,
                      description: "Apply guidance only between these two times (0 noise, 1 data)",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Generated image result",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/ImageGenerationResponse" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/video": {
        post: {
          summary: "Generate video from text",
          description: "Generates a video clip matching the prompt with customizable frame count, dimensions, frame rate, and other parameters.",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["prompt"],
                  properties: {
                    prompt: {
                      type: "string",
                      minLength: 1,
                      maxLength: generateRoutes.MAX_PROMPT_LENGTH,
                    },
                    num_frames: {
                      type: "integer",
                      minimum: 1,
                      maximum: generateRoutes.MAX_FRAMES,
                      default: 16,
                      description:
                        "Duration in frames. Requests longer than one denoising window " +
                        "are generated as overlapped windows with latent carry-over.",
                    },
                    width: {
                      type: "integer",
                      minimum: generateRoutes.MIN_DIMENSION,
                      maximum: generateRoutes.MAX_DIMENSION,
                      description: "Required alongside height.",
                    },
                    height: {
                      type: "integer",
                      minimum: generateRoutes.MIN_DIMENSION,
                      maximum: generateRoutes.MAX_DIMENSION,
                      description: "Required alongside width.",
                    },
                    aspect: { type: "string", enum: generateRoutes.ASPECT_RATIOS },
                    tier: { type: "integer", enum: generateRoutes.SIZE_TIERS },
                    fps: {
                      type: "integer",
                      minimum: generateRoutes.MIN_FPS,
                      maximum: generateRoutes.MAX_FPS,
                      description: "Frame rate the clip is conditioned on and written at.",
                    },
                    seed: { type: "integer", minimum: 0, maximum: 2 ** 31 - 1 },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Generated video result",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/VideoGenerationResponse" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/audio": {
        post: {
          summary: "Generate audio/speech from text",
          description: "Generates audio or speech from text input.",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["prompt"],
                  properties: {
                    prompt: {
                      type: "string",
                      minLength: 1,
                      maxLength: generateRoutes.MAX_PROMPT_LENGTH,
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Generated audio result",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/AudioGenerationResponse" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/code": {
        post: {
          summary: "Generate code",
          description: "Generates code snippets in the requested programming language.",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["prompt"],
                  properties: {
                    prompt: {
                      type: "string",
                      minLength: 1,
                      maxLength: generateRoutes.MAX_PROMPT_LENGTH,
                    },
                    language: {
                      type: "string",
                      enum: generateRoutes.LANGUAGES,
                      default: "python",
                    },
                    settings: { $ref: "#/components/schemas/GenerationSettings" },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Generated code result",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/CodeGenerationResponse" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/upload": {
        post: {
          summary: "Store a file for a later chat turn to reference",
          description:
            "Stores an image or document and returns the path a chat message may attach. " +
            "Runs no model: attaching is not asking.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["file"],
                  properties: {
                    file: {
                      type: "string",
                      format: "binary",
                      description: `Image or document to attach (image/*, ${generateRoutes.DOCUMENT_MIME_TYPES.join(", ")}, max file size ${MAX_FILE_SIZE_MB}MB)`,
                    },
                  },
                },
              },
            },
          },
          responses: {
            "201": {
              description: "Stored attachment",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["path", "kind"],
                    properties: {
                      path: { type: "string" },
                      kind: { type: "string", enum: ["image", "document", "audio"] },
                      name: { type: "string" },
                      size: { type: "integer" },
                      mimetype: { type: "string" },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error or unattachable file type"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/document": {
        post: {
          summary: "Read an uploaded document",
          description:
            "Uploads a document and returns its text in reading order, with page markers. " +
            "Answers a prompt about the document when one is given. Pages that carry no " +
            "text layer are reported in scannedPages rather than silently returned empty.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["document"],
                  properties: {
                    document: {
                      type: "string",
                      format: "binary",
                      description: `Document to read (${generateRoutes.DOCUMENT_MIME_TYPES.join(", ")}, max file size ${MAX_FILE_SIZE_MB}MB)`,
                    },
                    prompt: {
                      type: "string",
                      maxLength: generateRoutes.MAX_PROMPT_LENGTH,
                      description: "Optional question about the document.",
                    },
                    max_pages: {
                      type: "integer",
                      minimum: 1,
                      maximum: generateRoutes.MAX_DOCUMENT_PAGES,
                      description: "Read at most this many pages.",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Extracted document text",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["text", "pages", "documentPath"],
                    properties: {
                      text: { type: "string" },
                      pages: { type: "integer" },
                      title: { type: "string" },
                      scannedPages: { type: "array", items: { type: "integer" } },
                      content: { type: "string" },
                      documentPath: { type: "string" },
                      metadata: { type: "object", additionalProperties: true },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error, invalid file type, or unreadable document"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/understand": {
        post: {
          summary: "Vision understanding from uploaded image",
          description: "Uploads an image file for vision understanding and description.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["image"],
                  properties: {
                    image: {
                      type: "string",
                      format: "binary",
                      description: `Image file to analyze (image/*, max file size ${MAX_FILE_SIZE_MB}MB)`,
                    },
                    prompt: {
                      type: "string",
                      maxLength: generateRoutes.MAX_PROMPT_LENGTH,
                      default: "Describe this image",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Vision understanding analysis",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["description", "imagePath"],
                    properties: {
                      description: { type: "string" },
                      imagePath: { type: "string" },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error or invalid file type"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/image-edit": {
        post: {
          summary: "Edit an image to a prompt",
          description:
            "Redraws an uploaded image toward `prompt` to the degree `strength` says (0 keeps it, 1 starts from noise), or with a `mask` (white where to generate) repaints inside the mask only and holds everything outside it. Output takes the image's size, rounded to what the generator produces. Requires image_gen_arch='latent_dit'.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["image", "prompt"],
                  properties: {
                    image: { type: "string", format: "binary", description: "Image to edit (image/*)" },
                    mask: { type: "string", format: "binary", description: "Optional mask, white where to generate" },
                    prompt: { type: "string", maxLength: generateRoutes.MAX_PROMPT_LENGTH },
                    strength: { type: "number", minimum: 0, maximum: 1, default: 0.6 },
                    seed: { type: "integer", minimum: 0, maximum: 2147483647 },
                    shift: { type: "number", minimum: 0.1, maximum: 10 },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Edited image result",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/ImageGenerationResponse" },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error or invalid file type"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/animate": {
        post: {
          summary: "Animate an image into a video",
          description:
            "Generates a clip that opens on the uploaded image and continues from it, following `prompt`. Requires video_gen_arch='spacetime_dit' and a clip that fits one denoising window.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["image", "prompt"],
                  properties: {
                    image: { type: "string", format: "binary", description: "First frame (image/*)" },
                    prompt: { type: "string", maxLength: generateRoutes.MAX_PROMPT_LENGTH },
                    num_frames: { type: "integer", minimum: 1, maximum: generateRoutes.MAX_FRAMES, default: 16 },
                    fps: { type: "integer", minimum: generateRoutes.MIN_FPS, maximum: generateRoutes.MAX_FPS },
                    seed: { type: "integer", minimum: 0, maximum: 2147483647 },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Generated video result",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: { type: "object", additionalProperties: true },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error or invalid file type"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/understand-video": {
        post: {
          summary: "Answer a question about a video",
          description:
            "Reads frames spread evenly across an uploaded video, encodes each with the vision tower in order, and answers `prompt` about the clip. Needs opencv-python in the worker.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["video"],
                  properties: {
                    video: { type: "string", format: "binary", description: "Video (video/mp4, video/webm, video/quicktime)" },
                    prompt: { type: "string", maxLength: generateRoutes.MAX_PROMPT_LENGTH, default: "Describe this video:" },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Video description",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["description", "videoPath"],
                    properties: {
                      description: { type: "string" },
                      frames: { type: "integer" },
                      videoPath: { type: "string" },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error or invalid file type"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/transcribe": {
        post: {
          summary: "Speech-to-text audio transcription",
          description: "Uploads an audio file for speech transcription.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["audio"],
                  properties: {
                    audio: {
                      type: "string",
                      format: "binary",
                      description: `Audio file to transcribe (audio/*, max file size ${MAX_FILE_SIZE_MB}MB)`,
                    },
                    prompt: {
                      type: "string",
                      maxLength: generateRoutes.MAX_PROMPT_LENGTH,
                      default: "Transcribe the audio:",
                    },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Transcription result",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["text", "audioPath"],
                    properties: {
                      text: { type: "string" },
                      audioPath: { type: "string" },
                      metadata: { type: "object", additionalProperties: true },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error or invalid file type"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/voice": {
        post: {
          summary: "Speak text in a cloned voice",
          description:
            "Generates speech for `prompt` in the voice of an uploaded reference clip. `consent` must be `true`, confirming the speaker has given permission. The output always carries a watermark that `/generate/verify-audio` detects. With `reference_text` (what the clip says), the clip's own acoustic tokens are continued as well, which carries the voice further. Requires a model built with speaker conditioning.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["reference", "prompt", "consent"],
                  properties: {
                    reference: {
                      type: "string",
                      format: "binary",
                      description: `Reference clip of the voice, 1 to 30 seconds (audio/*, max file size ${MAX_FILE_SIZE_MB}MB)`,
                    },
                    prompt: { type: "string", maxLength: generateRoutes.MAX_PROMPT_LENGTH },
                    consent: { type: "string", enum: ["true", "false"] },
                    reference_text: { type: "string", maxLength: generateRoutes.MAX_PROMPT_LENGTH },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Generated speech",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["id", "prompt", "audio"],
                    properties: {
                      id: { type: "string" },
                      prompt: { type: "string" },
                      audio: { type: "object", additionalProperties: true },
                      metadata: { type: "object", additionalProperties: true },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error, missing consent, or a model that cannot clone"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
          },
        },
      },
      "/generate/verify-audio": {
        post: {
          summary: "Check audio for the generated-speech watermark",
          description:
            "Correlates an uploaded recording with the watermark generated speech carries. A provenance check, not a guarantee: resampling, time stretching or heavy lossy re-encoding can remove the mark.",
          requestBody: {
            required: true,
            content: {
              "multipart/form-data": {
                schema: {
                  type: "object",
                  required: ["audio"],
                  properties: {
                    audio: { type: "string", format: "binary", description: "Recording to check (audio/*)" },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "Detection result",
              headers: rateLimitHeaderRefs,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    required: ["watermarked", "score", "threshold"],
                    properties: {
                      watermarked: { type: "boolean" },
                      score: { type: "number" },
                      threshold: { type: "number" },
                    },
                  },
                },
              },
            },
            "400": errorResponseRef(400, "Validation error or invalid file type"),
            "413": errorResponseRef(413, "Uploaded file too large"),
            "429": errorResponseRef(429, "Rate limit exceeded"),
            "503": errorResponseRef(503, "Model worker not running"),
          },
        },
      },
    },
    components: {
      schemas: {
        ApiError: {
          type: "object",
          required: ["error", "code"],
          properties: {
            error: {
              type: "string",
              description: "Human-readable error message",
            },
            code: {
              type: "string",
              enum: [
                "VALIDATION_ERROR",
                "NOT_FOUND",
                "INVALID_JSON",
                "PAYLOAD_TOO_LARGE",
                "UPLOAD_ERROR",
                "RATE_LIMITED",
                "INTERNAL_ERROR",
              ],
              description: "Machine-readable error code",
            },
            details: {
              type: "array",
              description: "Field validation error list (present for VALIDATION_ERROR or UPLOAD_ERROR)",
              items: {
                type: "object",
                required: ["field", "message"],
                properties: {
                  field: { type: "string" },
                  message: { type: "string" },
                },
              },
            },
          },
        },
        GenerationSettings: {
          type: "object",
          description: "Sampling and generation control parameters",
          properties: {
            temperature: {
              type: "number",
              minimum: LIMITS.temperature.min,
              maximum: LIMITS.temperature.max,
            },
            top_p: {
              type: "number",
              minimum: LIMITS.top_p.min,
              maximum: LIMITS.top_p.max,
            },
            top_k: {
              type: "integer",
              minimum: LIMITS.top_k.min,
              maximum: LIMITS.top_k.max,
            },
            max_new_tokens: {
              type: "integer",
              minimum: LIMITS.max_new_tokens.min,
              maximum: LIMITS.max_new_tokens.max,
            },
            resolution: {
              type: "integer",
              enum: LIMITS.resolution,
            },
            num_frames: {
              type: "integer",
              minimum: LIMITS.num_frames.min,
              maximum: LIMITS.num_frames.max,
            },
          },
        },
        ChatMessage: {
          type: "object",
          required: ["id", "role", "content", "type"],
          properties: {
            id: { type: "string", format: "uuid" },
            role: { type: "string", enum: ["user", "assistant"] },
            content: { type: "string" },
            type: { type: "string", enum: ["text", "code", "image", "video", "audio", "error"] },
            attachments: {
              type: "array",
              items: { type: "object", additionalProperties: true },
            },
            metadata: { type: "object", additionalProperties: true },
            timestamp: { type: "string", format: "date-time" },
          },
        },
        SendMessageResponse: {
          allOf: [
            { $ref: "#/components/schemas/ChatMessage" },
            {
              type: "object",
              required: ["userMessageId"],
              properties: {
                userMessageId: {
                  type: "string",
                  format: "uuid",
                  description:
                    "Id the user's message was stored under. The reply's own id is id. Both are the ids to use when referring to these messages again, for example to branch from one.",
                },
              },
            },
          ],
        },
        ConversationSummary: {
          type: "object",
          required: ["id", "title", "messageCount"],
          properties: {
            id: { type: "string", format: "uuid" },
            title: { type: "string" },
            version: { type: "integer", minimum: 1 },
            parentConversationId: { type: "string", format: "uuid" },
            parentVersion: { type: "integer", minimum: 1 },
            branchedFromMessageId: { type: "string", format: "uuid" },
            createdAt: { type: "string", format: "date-time" },
            updatedAt: { type: "string", format: "date-time" },
            messageCount: { type: "integer", minimum: 0 },
          },
        },
        Conversation: {
          type: "object",
          required: ["id", "title", "messages"],
          properties: {
            id: { type: "string", format: "uuid" },
            title: { type: "string" },
            version: { type: "integer", minimum: 1, default: 1 },
            parentConversationId: { type: "string", format: "uuid" },
            parentVersion: { type: "integer", minimum: 1 },
            branchedFromMessageId: { type: "string", format: "uuid" },
            messages: {
              type: "array",
              items: { $ref: "#/components/schemas/ChatMessage" },
            },
            createdAt: { type: "string", format: "date-time" },
            updatedAt: { type: "string", format: "date-time" },
          },
        },
        ImageGenerationResponse: {
          type: "object",
          required: ["id", "prompt", "images", "metadata"],
          properties: {
            id: { type: "string", format: "uuid" },
            prompt: { type: "string" },
            images: {
              type: "array",
              items: {
                type: "object",
                required: ["id", "placeholder"],
                properties: {
                  id: { type: "string", format: "uuid" },
                  url: { type: ["string", "null"] },
                  placeholder: { type: "boolean" },
                  message: { type: "string" },
                },
              },
            },
            metadata: { type: "object", additionalProperties: true },
          },
        },
        VideoGenerationResponse: {
          type: "object",
          required: ["id", "prompt", "video", "metadata"],
          properties: {
            id: { type: "string", format: "uuid" },
            prompt: { type: "string" },
            video: {
              type: "object",
              required: ["frames", "placeholder"],
              properties: {
                url: { type: ["string", "null"] },
                frames: { type: "integer" },
                fps: { type: "integer" },
                placeholder: { type: "boolean" },
                message: { type: "string" },
              },
            },
            metadata: {
              type: "object",
              properties: {
                frames: { type: "integer" },
                fps: { type: "integer" },
                width: { type: "integer" },
                height: { type: "integer" },
                model: { type: "string" },
              },
              additionalProperties: true
            },
          },
        },
        AudioGenerationResponse: {
          type: "object",
          required: ["id", "prompt", "audio", "metadata"],
          properties: {
            id: { type: "string", format: "uuid" },
            prompt: { type: "string" },
            audio: {
              type: "object",
              required: ["placeholder"],
              properties: {
                url: { type: ["string", "null"] },
                placeholder: { type: "boolean" },
                message: { type: "string" },
              },
            },
            metadata: { type: "object", additionalProperties: true },
          },
        },
        CodeGenerationResponse: {
          type: "object",
          required: ["id", "prompt", "code", "language", "metadata"],
          properties: {
            id: { type: "string", format: "uuid" },
            prompt: { type: "string" },
            code: { type: "string" },
            language: { type: "string" },
            metadata: { type: "object", additionalProperties: true },
          },
        },
      },
      headers: {
        "RateLimit-Limit": {
          description: "Maximum requests allowed in rate-limit window",
          schema: { type: "integer" },
        },
        "RateLimit-Remaining": {
          description: "Remaining requests in current window",
          schema: { type: "integer" },
        },
        "RateLimit-Reset": {
          description: "Seconds until rate limit window resets",
          schema: { type: "integer" },
        },
        "Retry-After": {
          description: "Seconds to wait before retrying when rate-limited",
          schema: { type: "integer" },
        },
      },
    },
  };
}

function getOpenApiSpecJson() {
  const spec = generateOpenApiSpec();
  return JSON.stringify(spec, null, 2) + "\n";
}

module.exports = { generateOpenApiSpec, getOpenApiSpecJson };
