import { HashRouter, Routes, Route } from "react-router-dom";
import NavBar from "./components/NavBar.jsx";
import Footer from "./components/Footer.jsx";
import Home from "./pages/Home.jsx";
import Analyze from "./pages/Analyze.jsx";
import History from "./pages/History.jsx";
import { HistoryProvider } from "./context/HistoryContext.jsx";

export default function App() {
  return (
    <HistoryProvider>
      <HashRouter>
        <div className="flex min-h-screen flex-col">
          <NavBar />
          <p role="note" className="border-b border-ink bg-turmeric px-4 py-2 text-center text-[14px] font-medium text-ink">
            Screening aid built for study. It does not replace a laboratory test.
          </p>
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/analyze" element={<Analyze />} />
              <Route path="/history" element={<History />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </HashRouter>
    </HistoryProvider>
  );
}
