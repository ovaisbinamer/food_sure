import { useState } from "react";
import Dropzone from "../components/Dropzone.jsx";
import ResultPanel from "../components/ResultPanel.jsx";
import { predict } from "../lib/api.js";
import { SPICES, spiceName } from "../lib/format.js";
import { useHistory } from "../context/HistoryContext.jsx";

export default function Analyze() {
  const [spice, setSpice] = useState(null);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const { addEntry } = useHistory();

  const ready = spice && file && !busy;

  function pickSpice(id) {
    setSpice(id);
    setResult(null);
  }

  function pickFile(f) {
    setFile(f);
    setResult(null);
    setError("");
  }

  async function handleAnalyze() {
    if (!ready) return;
    setBusy(true);
    setError("");
    try {
      const data = await predict(spice, file);
      setResult(data);
      addEntry({
        spice,
        spiceLabel: spiceName(spice),
        result: data,
        thumbUrl: URL.createObjectURL(file),
        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      setError(err.message || "Unexpected error.");
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const buttonText = busy
    ? "Testing…"
    : !spice
    ? "Choose a spice first"
    : !file
    ? "Add a photo first"
    : `Test this ${spiceName(spice).toLowerCase()}`;

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <h1 className="text-[28px] font-semibold tracking-tight">Test a sample</h1>

      <div className="mt-7 grid gap-10 md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div>
          <Field n="1" label="Which spice is it?">
            <div className="grid grid-cols-3 gap-2">
              {SPICES.map((s) => {
                const active = spice === s.id;
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => pickSpice(s.id)}
                    disabled={busy}
                    aria-pressed={active}
                    className={`rounded-[3px] border p-2 text-left disabled:opacity-60 ${
                      active ? "border-ink bg-tint shadow-[inset_0_-3px_0_var(--color-turmeric)]" : "border-rule hover:border-ink-faint"
                    }`}
                  >
                    <img src={s.thumb} alt="" className="aspect-[4/3] w-full rounded-[2px] object-cover" />
                    <span className="mt-2 block text-[14px] font-medium leading-tight">{s.name}</span>
                    <span className="mt-0.5 block text-[12px] leading-snug text-ink-faint">
                      checks for {s.adulterants.join(", ")}
                    </span>
                  </button>
                );
              })}
            </div>
          </Field>

          <Field n="2" label="Add a photo of the powder" className="mt-8">
            <Dropzone
              file={file}
              onFile={pickFile}
              onClear={() => {
                setFile(null);
                setResult(null);
              }}
              disabled={busy}
            />
          </Field>

          <button
            type="button"
            disabled={!ready}
            onClick={handleAnalyze}
            className={`mt-8 flex w-full items-center justify-center gap-2.5 rounded-[3px] py-3.5 text-[15px] font-semibold ${
              ready ? "bg-turmeric text-ink hover:bg-turmeric-dark" : "cursor-not-allowed bg-tint text-ink-faint"
            }`}
          >
            {busy && <span className="spinner" />}
            {buttonText}
          </button>
          {busy && (
            <p className="mt-2 text-[12.5px] text-ink-faint">
              The first test of each spice takes a few seconds while its models load.
            </p>
          )}

          {error && (
            <div role="alert" className="mt-4 border-l-4 border-chilli bg-tint px-4 py-3">
              <p className="text-[14px] font-semibold text-chilli">The test could not be completed</p>
              <p className="mt-0.5 text-[13.5px] text-ink-soft">{error}</p>
            </div>
          )}
        </div>

        <div aria-live="polite">
          {result ? (
            <ResultPanel result={result} spice={spice} />
          ) : (
            <div className="grid h-full min-h-48 place-items-center rounded-[3px] border border-dashed border-rule px-6 py-10 text-center">
              <p className="max-w-xs text-[14px] leading-relaxed text-ink-faint">
                The report appears here: the verdict, the level, and how each of the two models voted.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({ n, label, className = "", children }) {
  return (
    <div className={className}>
      <p className="mb-3 flex gap-2.5 text-[15px] font-semibold">
        <span className="font-mono font-normal text-ink-faint">{n}</span>
        {label}
      </p>
      {children}
    </div>
  );
}
