"use client";

import React from "react";

const PHASES = [
  {
    number: "01",
    icon: "🕸️",
    title: "Extracción Web",
    subtitle: "Web Scraping Automatizado",
    gradient: "from-cyan-400 to-blue-500",
    glowColor: "rgba(6,182,212,0.3)",
    borderColor: "border-cyan-500/30",
    bgColor: "bg-cyan-500/5",
    description:
      "Scraping automatizado utilizando BeautifulSoup y Selenium para recolectar mallas curriculares desde portales universitarios.",
    details: [
      { label: "IES objetivo", value: "47 Instituciones de Educación Superior chilenas" },
      { label: "Estrategia", value: "Fallback automático Requests ↔ Selenium" },
      { label: "Parser", value: "BeautifulSoup + selectores CSS personalizados" },
      { label: "Output", value: "perfiles_egreso_raw.csv · 63 perfiles depurados" },
    ],
    tools: ["BeautifulSoup", "Selenium", "Requests", "Pandas"],
  },
  {
    number: "02",
    icon: "🧠",
    title: "NLP y Limpieza Avanzada",
    subtitle: "Procesamiento de Lenguaje Natural",
    gradient: "from-blue-500 to-violet-500",
    glowColor: "rgba(99,102,241,0.3)",
    borderColor: "border-violet-500/30",
    bgColor: "bg-violet-500/5",
    description:
      "Procesamiento de Lenguaje Natural con spaCy. Remoción de stopwords, lematización y control de confusores institucionales.",
    details: [
      { label: "Motor NLP", value: "spaCy es_core_news_sm — Español" },
      { label: "Técnica", value: "Lematización: 'optimizando' → 'optimizar'" },
      { label: "Stopwords", value: "Términos institucionales neutralizados" },
      { label: "Output", value: "perfiles_egreso_limpio_v1.csv" },
    ],
    tools: ["spaCy", "es_core_news_sm", "Regex"],
  },
  {
    number: "03",
    icon: "🧮",
    title: "Vectorización TF-IDF",
    subtitle: "Representación Matemática del Texto",
    gradient: "from-violet-500 to-purple-500",
    glowColor: "rgba(139,92,246,0.3)",
    borderColor: "border-purple-500/30",
    bgColor: "bg-purple-500/5",
    description:
      "Transformación del texto limpio en una matriz matemática densa de 1500 dimensiones para el análisis de frecuencia relativa de términos (Z-Score).",
    details: [
      { label: "Algoritmo", value: "TF-IDF (Term Frequency · Inverse Document Frequency)" },
      { label: "Dimensiones", value: "1500 características por perfil de egreso" },
      { label: "Análisis semántico", value: "Z-Score para identificar términos distintivos" },
      { label: "Output", value: "Matriz densa 63 × 1500 · top15_palabras_clave.csv" },
    ],
    tools: ["TF-IDF", "scikit-learn", "Z-Score", "NumPy"],
  },
  {
    number: "04",
    icon: "🤖",
    title: "Machine Learning Supervisado",
    subtitle: "Clasificación y Validación Estadística",
    gradient: "from-purple-500 to-emerald-400",
    glowColor: "rgba(16,185,129,0.3)",
    borderColor: "border-emerald-500/30",
    bgColor: "bg-emerald-500/5",
    description:
      "SMOTE balancea las clases minoritarias. Un benchmark de 11 clasificadores con Validación Cruzada Estratificada (5-Folds) determina al SVM con Kernel RBF como modelo ganador con 87.58% de Accuracy. PCA y LDA proyectan el espacio vectorial en 2D.",
    details: [
      { label: "Modelo Ganador", value: "SVM con Kernel RBF 🏆 · Accuracy 87.58%" },
      { label: "Benchmark", value: "11 clasificadores evaluados con Stratified K-Fold (k=5)" },
      { label: "Balanceo", value: "SMOTE para clases minoritarias" },
      { label: "Reducción dim.", value: "PCA (no supervisado) + LDA (supervisado) → 2D" },
    ],
    tools: ["SVM Kernel RBF", "SMOTE", "PCA", "LDA", "K-Fold CV", "scikit-learn"],
  },
];

const MODEL_METRICS = [
  { label: "Modelo Ganador", value: "SVM", sub: "Kernel RBF · Benchmark 11 algoritmos", icon: "🏆", color: "#8b5cf6" },
  { label: "Accuracy Final", value: "87.58%", sub: "Validación Cruzada Estratificada 5-Folds", icon: "📊", color: "#06b6d4" },
  { label: "Convergencia Civil ↔ Info", value: "76.8%", sub: "Similitud Coseno · 0.768 sobre centroides TF-IDF", icon: "🔬", color: "#10b981" },
  { label: "Perfiles depurados", value: "63", sub: "3 grados · 47 Instituciones de Educación Superior", icon: "📋", color: "#f59e0b" },
];

export default function ArchitectureSection() {
  const [activePhase, setActivePhase] = React.useState(0);
  const phase = PHASES[activePhase];

  return (
    <section id="futuro" className="relative py-24 px-6">
      <div className="max-w-7xl mx-auto space-y-16">

        {/* Header */}
        <div className="text-center space-y-4">
          <span className="badge badge-purple">Stack Tecnológico</span>
          <h2 className="text-3xl md:text-5xl font-black text-white mt-4">
            Arquitectura del{" "}
            <span className="gradient-text-purple">Pipeline de Análisis</span>
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            4 fases modulares construidas en Python puro que transforman texto web
            crudo en evidencia estadística reproducible y validada.
          </p>
        </div>

        {/* Phase tabs */}
        <div className="flex flex-wrap justify-center gap-3">
          {PHASES.map((p, i) => (
            <button
              key={i}
              id={`arch-tab-${i}`}
              onClick={() => setActivePhase(i)}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 border ${
                activePhase === i
                  ? `bg-gradient-to-r ${p.gradient} text-white border-transparent shadow-lg`
                  : "text-slate-400 border-slate-700 hover:border-slate-500 hover:text-slate-200 bg-transparent"
              }`}
            >
              <span>{p.icon}</span>
              <span className="hidden sm:inline">Fase {p.number}:</span>
              {p.title}
            </button>
          ))}
        </div>

        {/* Phase detail */}
        <div
          key={activePhase}
          className={`glass-card border ${phase.borderColor} ${phase.bgColor} p-8 md:p-10 animate-slide-up`}
        >
          <div className="grid md:grid-cols-2 gap-10">
            {/* Left */}
            <div className="space-y-6">
              <div className="flex items-start gap-4">
                <div
                  className={`w-14 h-14 rounded-xl bg-gradient-to-br ${phase.gradient} flex items-center justify-center text-2xl flex-shrink-0`}
                  style={{ boxShadow: `0 0 20px ${phase.glowColor}` }}
                >
                  {phase.icon}
                </div>
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-widest mb-1">
                    Fase {phase.number}
                  </p>
                  <h3 className="text-2xl font-black text-white">{phase.title}</h3>
                  <p
                    className={`text-sm font-semibold bg-gradient-to-r ${phase.gradient} bg-clip-text text-transparent`}
                  >
                    {phase.subtitle}
                  </p>
                </div>
              </div>

              <p className="text-slate-300 leading-relaxed">{phase.description}</p>

              {/* Tool badges */}
              <div className="flex flex-wrap gap-2">
                {phase.tools.map((tool) => (
                  <span
                    key={tool}
                    className="px-3 py-1 text-xs font-semibold rounded-full bg-slate-800 text-slate-300 border border-slate-700"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </div>

            {/* Right: step details */}
            <div className="space-y-3">
              <p className="text-xs text-slate-500 uppercase tracking-widest mb-4">
                Detalles técnicos
              </p>
              {phase.details.map((d, j) => (
                <div
                  key={j}
                  className="flex items-start gap-4 p-4 rounded-xl bg-slate-900/40 border border-slate-800/60 hover:border-slate-600/60 transition-colors duration-200"
                >
                  <div
                    className={`w-6 h-6 rounded-full bg-gradient-to-br ${phase.gradient} flex items-center justify-center text-xs font-black text-white flex-shrink-0 mt-0.5`}
                  >
                    {j + 1}
                  </div>
                  <div>
                    <p className="font-mono text-sm font-semibold text-slate-200">{d.label}</p>
                    <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{d.value}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Pipeline flow */}
        <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
          {PHASES.map((p, i) => (
            <React.Fragment key={i}>
              <button
                onClick={() => setActivePhase(i)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all duration-200 ${
                  activePhase === i
                    ? "border-cyan-500/50 text-cyan-400 bg-cyan-500/10"
                    : "border-slate-800 text-slate-500 hover:text-slate-300 hover:border-slate-600"
                }`}
              >
                <span>{p.icon}</span>
                <span className="font-medium">{p.title}</span>
              </button>
              {i < PHASES.length - 1 && (
                <svg
                  className="w-4 h-4 text-slate-600 flex-shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Model metrics summary */}
        <div className="glass-card border border-white/5 p-8">
          <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-3">
            <span className="w-1 h-6 rounded-full bg-gradient-to-b from-purple-400 to-cyan-500" />
            Métricas del Modelo Final
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {MODEL_METRICS.map((m, i) => (
              <div
                key={i}
                className="p-5 rounded-xl bg-slate-900/50 border border-slate-800/60 text-center hover:border-slate-700 transition-colors duration-200"
              >
                <span className="text-2xl block mb-2">{m.icon}</span>
                <p className="text-xs text-slate-500 uppercase tracking-widest">{m.label}</p>
                <p
                  className="text-2xl font-black mt-1 leading-tight"
                  style={{ color: m.color }}
                >
                  {m.value}
                </p>
                <p className="text-xs text-slate-600 mt-1 leading-tight">{m.sub}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Tech stack row */}
        <div className="grid md:grid-cols-3 gap-4">
          {[
            {
              label: "Lenguaje",
              value: "Python 3.13.5",
              desc: "Pipeline 100% reproducible · scripts modulares",
              icon: "🐍",
              color: "text-cyan-400",
            },
            {
              label: "NLP & ML",
              value: "scikit-learn + spaCy",
              desc: "TF-IDF · SVM Kernel RBF · SMOTE · PCA · LDA · K-Fold CV",
              icon: "🧠",
              color: "text-purple-400",
            },
            {
              label: "Visualización",
              value: "Matplotlib + Next.js",
              desc: "Gráficos estáticos · Landing Page interactiva",
              icon: "📊",
              color: "text-emerald-400",
            },
          ].map((item, i) => (
            <div
              key={i}
              className="glass-card p-5 border border-slate-800/60 text-center hover:border-slate-700/60 transition-colors duration-200"
            >
              <span className="text-2xl">{item.icon}</span>
              <p className="text-xs text-slate-500 mt-2 uppercase tracking-widest">
                {item.label}
              </p>
              <p className={`text-lg font-black mt-1 ${item.color}`}>{item.value}</p>
              <p className="text-xs text-slate-500 mt-1">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
