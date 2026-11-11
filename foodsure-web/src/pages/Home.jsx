import { Link } from "react-router-dom";
import { SPICES, LEVEL_BANDS, pct } from "../lib/format.js";

const FIGURES = [
  { src: "samples/black_pepper_powder__15_pct_millet_thumb.jpg", caption: "Black pepper with 15% millet" },
  { src: "samples/red_chilli_powder__10_pct_sawdust_thumb.jpg", caption: "Red chilli with 10% sawdust" },
  { src: "samples/turmeric__5_pct_semolina_thumb.jpg", caption: "Turmeric with 5% semolina" },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6">
      <section className="grid gap-10 py-12 md:grid-cols-[1fr_1.05fr] md:items-end md:py-16">
        <div>
          <h1 className="text-[34px] font-semibold leading-[1.12] tracking-tight sm:text-[42px]">
            A photo test for adulterated spices
          </h1>
          <p className="mt-5 max-w-md text-[16.5px] leading-relaxed text-ink-soft">
            Food-Sure looks at a photo of black pepper, red chilli or turmeric powder. It reports whether
            the powder has been mixed with a cheaper filler, which filler, and roughly how much.
          </p>
          <Link
            to="/analyze"
            className="mt-7 inline-block rounded-[3px] bg-turmeric px-5 py-3 text-[15px] font-semibold text-ink hover:bg-turmeric-dark"
          >
            Test a sample
          </Link>
        </div>

        <figure>
          <div className="grid grid-cols-3 gap-2">
            {FIGURES.map((f) => (
              <div key={f.src}>
                <img src={f.src} alt={f.caption} className="aspect-square w-full rounded-[3px] object-cover" />
                <p className="mt-2 text-[12.5px] leading-snug text-ink-soft">{f.caption}</p>
              </div>
            ))}
          </div>
          <figcaption className="mt-3 border-t border-rule pt-2 text-[12.5px] text-ink-faint">
            Three photos from our test set, taken on white paper from above.
          </figcaption>
        </figure>
      </section>

      <Section title="What it can detect">
        <div className="overflow-x-auto">
          <table className="report-table">
            <thead>
              <tr>
                <th>Spice</th>
                <th>Fillers it knows</th>
                <th>Reported level</th>
              </tr>
            </thead>
            <tbody>
              {SPICES.map((s) => (
                <tr key={s.id}>
                  <td className="font-medium">{s.name}</td>
                  <td>{s.adulterants.join(", ")}</td>
                  <td className="text-ink-soft">{LEVEL_BANDS.map((b) => b.range).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Note>
          The models were trained on samples we mixed by weight at exactly 5%, 10% and 15%. A sample in
          between is reported as the nearest band, not as an exact number.
        </Note>
      </Section>

      <Section title="How well it works">
        <div className="overflow-x-auto">
          <table className="report-table">
            <thead>
              <tr>
                <th>Spice</th>
                <th className="num">Test photos</th>
                <th className="num">Pure or right filler</th>
                <th className="num">Right filler and level</th>
              </tr>
            </thead>
            <tbody>
              {SPICES.map((s) => (
                <tr key={s.id}>
                  <td className="font-medium">{s.name}</td>
                  <td className="num">{s.test.photos}</td>
                  <td className="num">
                    {pct(s.test.type, s.test.photos)}{" "}
                    <span className="text-ink-faint">
                      ({s.test.type}/{s.test.photos})
                    </span>
                  </td>
                  <td className="num">
                    {pct(s.test.exact, s.test.photos)}{" "}
                    <span className="text-ink-faint">
                      ({s.test.exact}/{s.test.photos})
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Note>
          Measured on photos that were kept aside and never used for training. Red chilli is the weakest
          of the three: its colour and grain already vary a lot when it is pure.
        </Note>
      </Section>

      <Section title="How a test works">
        <ol className="max-w-2xl space-y-4 text-[15px] leading-relaxed">
          <Step n="1">
            You photograph a small pile of the powder on white paper, from above, and pick which spice it is.
          </Step>
          <Step n="2">
            Two models look at the photo. ResNet50 is a deep network that learned what each mix looks like
            from our photos. The second model measures colour and grain texture directly.
          </Step>
          <Step n="3">
            Their two opinions are averaged to decide between pure and each of the two fillers.
          </Step>
          <Step n="4">If a filler is found, ResNet50 estimates the level, which is shown as a band.</Step>
        </ol>
      </Section>

      <Section title="Limits">
        <ul className="max-w-2xl list-disc space-y-2 pl-5 text-[15px] leading-relaxed text-ink-soft">
          <li>It knows three spices and six fillers. Any other filler will be reported as one of these, or as pure.</li>
          <li>It cannot see dyes or chemicals, such as artificial colour or lead chromate.</li>
          <li>It was trained on photos like the ones above. Other lighting or backgrounds can change the answer.</li>
          <li>It is a screening aid. A result here is a reason to test further, not proof.</li>
        </ul>
      </Section>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section className="grid gap-x-10 gap-y-4 border-t border-ink py-9 md:grid-cols-[200px_1fr]">
      <h2 className="text-[17px] font-semibold tracking-tight">{title}</h2>
      <div>{children}</div>
    </section>
  );
}

function Note({ children }) {
  return <p className="mt-4 max-w-2xl text-[13.5px] leading-relaxed text-ink-faint">{children}</p>;
}

function Step({ n, children }) {
  return (
    <li className="flex gap-4">
      <span className="font-mono text-[14px] text-ink-faint">{n}</span>
      <span>{children}</span>
    </li>
  );
}
