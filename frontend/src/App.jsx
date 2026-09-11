import React, { useState } from "react";

const sections = ["Dashboard", "Live Recognition", "Add Person", "Batch Enrollment", "Manage Identities", "Cameras", "Recognition History", "Evaluation", "Settings"];

export default function App() {
  const [active, setActive] = useState("Dashboard");
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">FR</span><span>FaceWatch</span></div>
        <p className="eyebrow">CONTROL CENTER</p>
        <nav>{sections.map((section) => <button className={active === section ? "nav-item active" : "nav-item"} key={section} onClick={() => setActive(section)}>{section}</button>)}</nav>
        <div className="privacy-note">Biometric data is sensitive. Delete reference data when it is no longer required.</div>
      </aside>
      <main className="content">
        <header className="topbar"><div><p className="eyebrow">REAL-TIME OPERATIONS</p><h1>{active}</h1></div><span className="status"><i /> Backend ready</span></header>
        {active === "Dashboard" ? <Dashboard /> : <section className="panel empty"><h2>{active}</h2><p>This module is connected to the backend API and will be populated as its phase is completed.</p></section>}
      </main>
    </div>
  );
}

function Dashboard() {
  const cards = [["0", "Registered identities", "Active records"], ["0", "Active cameras", "USB + RTSP"], ["0", "Recognition events", "Since startup"], ["—", "Current FPS", "Waiting for camera"]];
  return <>
    <section className="hero"><div><p className="eyebrow">SYSTEM OVERVIEW</p><h2>Recognition, without retraining.</h2><p>Add or remove identities through the database and vector index while the fixed ArcFace model stays unchanged.</p></div><div className="hero-orb">◉</div></section>
    <section className="card-grid">{cards.map(([value, label, detail]) => <article className="stat-card" key={label}><span className="stat-value">{value}</span><strong>{label}</strong><small>{detail}</small></article>)}</section>
    <section className="lower-grid"><article className="panel"><div className="panel-heading"><h3>Recent recognition events</h3><span className="muted">LIVE</span></div><div className="empty-table">No recognition events yet. Start a camera to begin.</div></article><article className="panel"><div className="panel-heading"><h3>System pipeline</h3></div><div className="pipeline"><span>Camera</span><b>→</b><span>Detection</span><b>→</b><span>Embedding</span><b>→</b><span>Search</span><b>→</b><span>Known / Unknown</span></div></article></section>
  </>;
}

