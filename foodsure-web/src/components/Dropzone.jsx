import { useCallback, useEffect, useRef, useState } from "react";
import { formatBytes } from "../lib/format.js";

const MAX_MB = 15;

export default function Dropzone({ file, onFile, onClear, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const previewUrl = useObjectUrl(file);

  const handleFiles = useCallback(
    (files) => {
      const f = files?.[0];
      if (!f) return;
      if (!f.type.startsWith("image/")) {
        setError("That file is not an image. Please choose a JPG or PNG photo.");
        return;
      }
      if (f.size > MAX_MB * 1024 * 1024) {
        setError(`That image is larger than ${MAX_MB} MB. Try a smaller photo.`);
        return;
      }
      setError("");
      onFile(f);
    },
    [onFile]
  );

  const open = () => !disabled && inputRef.current?.click();

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {file ? (
        <div className="flex items-center gap-3 rounded-[3px] border border-ink p-2.5">
          {previewUrl && (
            <img src={previewUrl} alt="Your photo" className="h-16 w-16 flex-shrink-0 rounded-[2px] object-cover" />
          )}
          <div className="min-w-0 flex-1">
            <p className="truncate text-[14px] font-medium">{file.name}</p>
            <p className="text-[12.5px] text-ink-faint">{formatBytes(file.size)}</p>
          </div>
          <button
            type="button"
            onClick={onClear}
            disabled={disabled}
            className="flex-shrink-0 px-2 py-1 text-[13px] text-ink-soft underline underline-offset-2 hover:text-chilli disabled:opacity-40"
          >
            Remove
          </button>
        </div>
      ) : (
        <div
          role="button"
          tabIndex={0}
          aria-disabled={disabled}
          onClick={open}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              open();
            }
          }}
          onDragEnter={(e) => {
            e.preventDefault();
            if (!disabled) setDragging(true);
          }}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (!disabled) handleFiles(e.dataTransfer.files);
          }}
          className={`cursor-pointer rounded-[3px] border border-dashed px-4 py-7 text-center ${
            dragging ? "border-ink bg-tint" : "border-ink-faint hover:bg-tint"
          } ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
        >
          <p className="text-[14.5px] font-medium">Choose a photo, or drop one here</p>
          <p className="mt-1 text-[12.5px] text-ink-faint">
            JPG or PNG. Powder on white paper, shot from above, in good light.
          </p>
        </div>
      )}

      {error && <p className="mt-2 text-[13px] text-chilli">{error}</p>}
    </div>
  );
}

function useObjectUrl(file) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);
  return url;
}
