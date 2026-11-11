import { createContext, useContext, useState, useCallback } from "react";

const HistoryContext = createContext(null);

let nextId = 1;

export function HistoryProvider({ children }) {
  const [entries, setEntries] = useState([]);

  const addEntry = useCallback((entry) => {
    setEntries((prev) => [{ id: nextId++, ...entry }, ...prev]);
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return (
    <HistoryContext.Provider value={{ entries, addEntry, clear }}>
      {children}
    </HistoryContext.Provider>
  );
}

export function useHistory() {
  const ctx = useContext(HistoryContext);
  if (!ctx) throw new Error("useHistory must be used within HistoryProvider");
  return ctx;
}
