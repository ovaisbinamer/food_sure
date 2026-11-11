import { Link } from "react-router-dom";
import { useHistory } from "../context/HistoryContext.jsx";
import { SEVERITY_META, levelText, nice } from "../lib/format.js";

export default function History() {
  const { entries, clear } = useHistory();

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h1 className="text-[28px] font-semibold tracking-tight">History</h1>
        {entries.length > 0 && (
          <button onClick={clear} className="text-[13.5px] text-ink-soft underline underline-offset-2 hover:text-chilli">
            Clear the list
          </button>
        )}
      </div>
      <p className="mt-2 text-[14px] text-ink-faint">
        Tests run since this page was opened. The list is not saved and is lost when you reload.
      </p>

      {entries.length === 0 ? (
        <p className="mt-8 border-t border-ink pt-6 text-[15px] text-ink-soft">
          Nothing tested yet.{" "}
          <Link to="/analyze" className="font-medium text-ink underline underline-offset-2">
            Test a sample
          </Link>
        </p>
      ) : (
        <div className="mt-6 overflow-x-auto">
          <table className="report-table min-w-[640px]">
            <thead>
              <tr>
                <th>Time</th>
                <th>Photo</th>
                <th>Spice</th>
                <th>Verdict</th>
                <th>Level</th>
                <th className="num">Vote</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => {
                const severity = (e.result.severity || "UNKNOWN").toUpperCase();
                const meta = SEVERITY_META[severity] || SEVERITY_META.UNKNOWN;
                return (
                  <tr key={e.id}>
                    <td className="font-mono text-[13px] text-ink-soft">
                      {new Date(e.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </td>
                    <td>
                      {e.thumbUrl && <img src={e.thumbUrl} alt="" className="h-10 w-10 rounded-[2px] object-cover" />}
                    </td>
                    <td>{e.spiceLabel}</td>
                    <td className="font-medium capitalize" style={{ color: `var(--color-${meta.status})` }}>
                      {severity === "PURE" ? "Pure" : nice(e.result.adulterant)}
                    </td>
                    <td className="font-mono text-[13px]">{levelText(e.result)}</td>
                    <td className="num">{Math.round((e.result.confidence || 0) * 100)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
