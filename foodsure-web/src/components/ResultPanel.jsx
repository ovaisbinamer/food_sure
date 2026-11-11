import {
  SEVERITY_META,
  LEVEL_BANDS,
  formatClassLabel,
  typeOfClass,
  levelText,
  spiceName,
  nice,
} from "../lib/format.js";

const whole = (p) => `${Math.round((p || 0) * 100)}%`;

export default function ResultPanel({ result, spice }) {
  const severity = (result.severity || "UNKNOWN").toUpperCase();
  const meta = SEVERITY_META[severity] || SEVERITY_META.UNKNOWN;
  const color = `var(--color-${meta.status})`;
  const pure = severity === "PURE";
  const voted = result.decision === "vote" && Array.isArray(result.type_votes) && result.type_votes.length > 0;
  const chosenType = typeOfClass(result.predicted_class);

  return (
    <article className="rounded-[3px] border border-ink">
      <header className="border-l-[6px] px-5 py-5" style={{ borderColor: color }}>
        <p className="text-[13px] text-ink-faint">{spiceName(spice || result.spice)} sample</p>
        <h2 className="mt-1 text-[30px] font-semibold leading-tight tracking-tight" style={{ color }}>
          {pure ? "Pure" : severity === "UNKNOWN" ? "Unclear" : `Adulterated with ${nice(result.adulterant)}`}
        </h2>
        <p className="mt-2 text-[15px] text-ink-soft">
          {pure
            ? "No filler was found in this photo."
            : `Estimated level: ${levelText(result)} by weight.`}
        </p>
        <p className="mt-2 text-[12.5px] text-ink-faint">
          Only valid if this really is {spiceName(spice || result.spice).toLowerCase()}. The models cannot tell
          when the wrong spice was picked, and will still give a confident answer.
        </p>
      </header>

      {result.warning && (
        <div className="border-t border-rule bg-tint px-5 py-3.5">
          <p className="text-[14px] font-semibold text-warning">The models are not sure about this one</p>
          <p className="mt-1 text-[13.5px] leading-relaxed text-ink-soft">{result.warning}</p>
        </div>
      )}

      <Block title="Level">
        <div className="grid grid-cols-3 gap-1">
          {LEVEL_BANDS.map((b) => {
            const on = b.severity === severity;
            return (
              <div key={b.severity}>
                <div className="h-2.5 rounded-[2px]" style={{ background: on ? color : "var(--color-rule)" }} />
                <p className={`mt-1.5 text-[12.5px] ${on ? "font-semibold text-ink" : "text-ink-faint"}`}>{b.name}</p>
                <p className={`font-mono text-[12px] ${on ? "text-ink" : "text-ink-faint"}`}>{b.range}</p>
              </div>
            );
          })}
        </div>
        {!pure && severity !== "UNKNOWN" && (
          <p className="mt-3 text-[13px] text-ink-faint">
            ResNet50 picked this band with {whole(result.level_confidence)} confidence. Bands are used because
            the models were trained at 5%, 10% and 15% only.
          </p>
        )}
      </Block>

      {voted ? (
        <Block title="How the two models voted">
          <table className="report-table">
            <thead>
              <tr>
                <th>Option</th>
                <th className="num">ResNet50</th>
                <th className="num">Colour and texture</th>
                <th className="num">Average</th>
              </tr>
            </thead>
            <tbody>
              {result.type_votes.map((v) => {
                const chosen = v.type === chosenType;
                return (
                  <tr key={v.type} className={chosen ? "font-semibold" : "text-ink-soft"}>
                    <td className="capitalize">
                      {nice(v.type)}
                      {chosen && <span className="ml-2 text-[12px] font-normal text-ink-faint">chosen</span>}
                    </td>
                    <td className="num">{whole(v.resnet50)}</td>
                    <td className="num">{whole(v.pipeline)}</td>
                    <td className="num">{whole(v.voted)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-3 text-[13px] leading-relaxed text-ink-faint">
            {result.models_agree
              ? "Both models picked the same option."
              : "The two models picked different options, so treat this result with care."}{" "}
            The average decides the verdict.
          </p>
        </Block>
      ) : (
        Array.isArray(result.top3) &&
        result.top3.length > 0 && (
          <Block title="ResNet50's top guesses">
            <table className="report-table">
              <tbody>
                {result.top3.map(([label, prob], i) => (
                  <tr key={label + i} className={i === 0 ? "font-semibold" : "text-ink-soft"}>
                    <td>{formatClassLabel(label)}</td>
                    <td className="num">{whole(prob)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-[13px] text-ink-faint">
              Only ResNet50 answered. The colour and texture model is not installed for this spice.
            </p>
          </Block>
        )
      )}
    </article>
  );
}

function Block({ title, children }) {
  return (
    <section className="border-t border-rule px-5 py-4">
      <h3 className="mb-3 text-[13.5px] font-semibold">{title}</h3>
      {children}
    </section>
  );
}
