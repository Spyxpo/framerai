import { useState, useRef, useEffect, useCallback } from "react";
import { X, Download, Upload, FileText, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";

export default function BackupModal({
  open,
  conversations = [],
  activeConversationId,
  onClose,
  onExportConversation,
  onExportAll,
  onImportBackup,
}) {
  const [activeTab, setActiveTab] = useState("export"); // "export" | "import"
  const [importing, setImporting] = useState(false);
  const [statusMessage, setStatusMessage] = useState(null); // { type: "success" | "error", text: string }
  const fileInputRef = useRef(null);
  const closeBtnRef = useRef(null);
  const activeConv = conversations.find((c) => c.id === activeConversationId);

  const handleClose = useCallback(() => {
    setStatusMessage(null);
    setImporting(false);
    onClose?.();
  }, [onClose]);

  useEffect(() => {
    if (!open) return undefined;

    closeBtnRef.current?.focus();

    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        handleClose();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, handleClose]);

  const handleFileChange = useCallback(
    async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setImporting(true);
      setStatusMessage(null);

      try {
        const result = await onImportBackup?.(file);
        if (result && result.success) {
          setStatusMessage({
            type: "success",
            text: `Successfully restored ${result.count} conversation${result.count === 1 ? "" : "s"}.`,
          });
        } else {
          setStatusMessage({
            type: "error",
            text: result?.error || "Failed to restore backup.",
          });
        }
      } catch (err) {
        setStatusMessage({
          type: "error",
          text: err.message || "Failed to restore backup.",
        });
      } finally {
        setImporting(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [onImportBackup]
  );

  const handleDrop = useCallback(
    async (e) => {
      e.preventDefault();
      const file = e.dataTransfer?.files?.[0];
      if (!file) return;

      setImporting(true);
      setStatusMessage(null);

      try {
        const result = await onImportBackup?.(file);
        if (result && result.success) {
          setStatusMessage({
            type: "success",
            text: `Successfully restored ${result.count} conversation${result.count === 1 ? "" : "s"}.`,
          });
        } else {
          setStatusMessage({
            type: "error",
            text: result?.error || "Failed to restore backup.",
          });
        }
      } catch (err) {
        setStatusMessage({
          type: "error",
          text: err.message || "Failed to restore backup.",
        });
      } finally {
        setImporting(false);
      }
    },
    [onImportBackup]
  );

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  if (!open) return null;

  return (
    <div
      className="settings-overlay backup-overlay"
      onMouseDown={(e) => e.target === e.currentTarget && handleClose()}
      role="presentation"
    >
      <div
        className="settings-panel backup-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="backup-modal-title"
      >
        <div className="settings-header">
          <h2 id="backup-modal-title">Backup & Restore</h2>
          <button
            className="icon-btn"
            onClick={handleClose}
            aria-label="Close backup dialog"
            ref={closeBtnRef}
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        <div className="backup-tabs" role="tablist">
          <button
            role="tab"
            aria-selected={activeTab === "export"}
            className={`backup-tab ${activeTab === "export" ? "active" : ""}`}
            onClick={() => setActiveTab("export")}
          >
            <Download size={14} aria-hidden="true" />
            <span>Export</span>
          </button>
          <button
            role="tab"
            aria-selected={activeTab === "import"}
            className={`backup-tab ${activeTab === "import" ? "active" : ""}`}
            onClick={() => setActiveTab("import")}
          >
            <Upload size={14} aria-hidden="true" />
            <span>Import</span>
          </button>
        </div>

        <div className="settings-body backup-body">
          {statusMessage && (
            <div
              className={`backup-status ${statusMessage.type === "success" ? "status-success" : "status-error"}`}
              role="alert"
            >
              {statusMessage.type === "success" ? (
                <CheckCircle2 size={16} aria-hidden="true" />
              ) : (
                <AlertCircle size={16} aria-hidden="true" />
              )}
              <span>{statusMessage.text}</span>
            </div>
          )}

          {activeTab === "export" && (
            <div className="backup-section">
              <div className="backup-group">
                <h3>Active Conversation</h3>
                <p className="backup-description">
                  {activeConv
                    ? `Export "${activeConv.title || "New Chat"}" (${activeConv.messages?.length || 0} messages)`
                    : "No conversation currently selected"}
                </p>
                <div className="backup-actions">
                  <button
                    className="btn btn-secondary backup-action-btn"
                    onClick={() => {
                      onExportConversation?.(activeConversationId, "json");
                      setStatusMessage({
                        type: "success",
                        text: "Exported conversation as JSON backup.",
                      });
                    }}
                    disabled={!activeConv}
                  >
                    <Download size={15} aria-hidden="true" />
                    <span>JSON Backup</span>
                  </button>
                  <button
                    className="btn btn-secondary backup-action-btn"
                    onClick={() => {
                      onExportConversation?.(activeConversationId, "markdown");
                      setStatusMessage({
                        type: "success",
                        text: "Exported conversation as Markdown.",
                      });
                    }}
                    disabled={!activeConv}
                  >
                    <FileText size={15} aria-hidden="true" />
                    <span>Markdown</span>
                  </button>
                </div>
              </div>

              <div className="backup-group">
                <h3>All Conversations</h3>
                <p className="backup-description">
                  Download a full backup of all {conversations.length} conversation
                  {conversations.length === 1 ? "" : "s"} including branch histories, metadata, and messages.
                </p>
                <button
                  className="btn btn-primary backup-action-btn full-width"
                  onClick={() => {
                    onExportAll?.();
                    setStatusMessage({
                      type: "success",
                      text: `Exported full backup of ${conversations.length} conversations.`,
                    });
                  }}
                  disabled={conversations.length === 0}
                >
                  <Download size={15} aria-hidden="true" />
                  <span>Download Full Backup ({conversations.length})</span>
                </button>
              </div>
            </div>
          )}

          {activeTab === "import" && (
            <div className="backup-section">
              <div
                className={`backup-dropzone ${importing ? "busy" : ""}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => !importing && fileInputRef.current?.click()}
                tabIndex={0}
                role="button"
                aria-label="Click or drag and drop a backup JSON file to import"
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json,application/json"
                  style={{ display: "none" }}
                  onChange={handleFileChange}
                  data-testid="backup-modal-file-input"
                />

                {importing ? (
                  <div className="backup-dropzone-loading">
                    <Loader2 size={32} className="spin" aria-hidden="true" />
                    <p>Restoring conversations...</p>
                  </div>
                ) : (
                  <div className="backup-dropzone-idle">
                    <Upload size={32} className="dropzone-icon" aria-hidden="true" />
                    <p>Click to select or drag and drop a backup file</p>
                    <span>Supports .json backups created with FramerAI</span>
                  </div>
                )}
              </div>

              <div className="backup-notice">
                <p>
                  <strong>Safety guarantee:</strong> Imported conversations are restored atomically.
                  Existing conversation histories will not be overwritten.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
