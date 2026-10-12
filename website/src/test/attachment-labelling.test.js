import { describe, it, expect } from "vitest";
import { labelled } from "../services/api";

describe("labelled", () => {
  it("labels an untyped source file by its extension", () => {
    const file = labelled(new File(["fn main() {}"], "main.rs", { type: "" }));
    expect(file.type).toBe("text/x-rust");
    expect(file.name).toBe("main.rs");
  });

  it("labels a generic office upload", () => {
    const file = labelled(new File(["x"], "plan.docx", { type: "application/octet-stream" }));
    expect(file.type).toBe("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
  });

  it("relabels TypeScript that a browser called a video stream", () => {
    expect(labelled(new File(["let a = 1"], "a.ts", { type: "video/mp2t" })).type).toBe("application/typescript");
  });

  it("keeps a specific type the browser already gave", () => {
    const original = new File(["%PDF"], "r.pdf", { type: "application/pdf" });
    expect(labelled(original)).toBe(original);
  });

  it("leaves an unknown extension alone", () => {
    const original = new File(["x"], "blob.bin", { type: "" });
    expect(labelled(original)).toBe(original);
  });
});
