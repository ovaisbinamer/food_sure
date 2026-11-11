import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "About" },
  { to: "/analyze", label: "Test a sample" },
  { to: "/history", label: "History" },
];

export default function NavBar() {
  return (
    <header className="border-b border-ink">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-8 gap-y-2 px-4 py-3 sm:px-6">
        <NavLink to="/" className="flex items-center gap-2 text-[17px] font-semibold tracking-tight">
          <span className="grid h-5 w-5 place-items-center rounded-[3px] bg-ink">
            <span className="h-2.5 w-2.5 rounded-full bg-turmeric" />
          </span>
          Food-Sure
        </NavLink>

        <nav className="order-3 flex w-full gap-6 text-[14.5px] sm:order-none sm:w-auto">
          {LINKS.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end
              className={({ isActive }) =>
                `border-b-2 py-1 ${
                  isActive ? "border-turmeric font-medium text-ink" : "border-transparent text-ink-soft hover:text-ink"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
