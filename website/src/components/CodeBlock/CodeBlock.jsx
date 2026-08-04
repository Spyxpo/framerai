import React, { useState } from "react";
import { Copy, Check } from "lucide-react";

export default function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="code-block">
      <div className="code-header">
        <span className="code-language">{language}</span>
        <button
          type="button"
          className="copy-btn"
          onClick={handleCopy}
          aria-label={copied ? `${language} code copied` : `Copy ${language} code`}
        >
          {copied ? (
            <>
              <Check size={14} aria-hidden="true" /> Copied
            </>
          ) : (
            <>
              <Copy size={14} aria-hidden="true" /> Copy
            </>
          )}
        </button>
      </div>
      <pre className="code-content" tabIndex={0}>
        <code>{code}</code>
      </pre>
    </div>
  );
}
