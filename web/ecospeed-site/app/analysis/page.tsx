"use client";

import { useState } from "react";

export default function Analysis() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    setResult(null);

    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: { lat: 48.8566, lng: 2.3522 }, // Paris (exemple)
        end: { lat: 45.7640, lng: 4.8357 },   // Lyon (exemple)
      }),
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  }

  return (
    <main className="min-h-screen bg-black text-white p-10">
      <h1 className="text-4xl font-bold">EcoSpeed — Analysis</h1>

      <button
        onClick={run}
        className="mt-6 rounded-xl bg-white px-6 py-3 text-black font-semibold"
      >
        {loading ? "Running..." : "Run analysis"}
      </button>

      {result && (
        <pre className="mt-6 rounded-xl bg-white/5 p-4 text-sm overflow-auto">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}
