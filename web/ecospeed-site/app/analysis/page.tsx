export default function Analysis() {
  const streamlitUrl =
    "https://ev-app-main-ev9nvgzne8e4qhc5jxwac3.streamlit.app/";

  return (
    <main className="min-h-screen bg-black text-white flex flex-col items-center justify-center gap-6 px-6">
      <h1 className="text-4xl font-bold">EcoSpeed — Analysis</h1>

      <a
        href={streamlitUrl}
        target="_blank"
        rel="noreferrer"
        className="rounded-xl bg-white px-6 py-3 text-black font-semibold"
      >
        Open Streamlit App
      </a>

      <p className="text-white/60 text-sm">
        The full analysis runs on Streamlit.
      </p>
    </main>
  );
}
