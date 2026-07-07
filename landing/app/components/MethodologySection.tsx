"use client";

import React from "react";

// ─── Las 4 fases reales del pipeline ────────────────────────────────────────
const PHASES = [
  {
    number: "01",
    title: "Recolección",
    subtitle: "Web Scraping Híbrido",
    gradient: "from-cyan-400 to-blue-500",
    glowColor: "rgba(6, 182, 212, 0.3)",
    borderColor: "border-cyan-500/30",
    bgColor: "bg-cyan-500/5",
    icon: "🌐",
    description:
      "Extracción automatizada de 64 perfiles de egreso depurados (Civil, Informática, Ejecución) de 47 Instituciones de Educación Superior configurada con selectores CSS y fallback automático.",
    steps: [
      { label: "config.py", desc: "64 perfiles objetivo de 47 IES chilenas configurados con selectores CSS por sitio" },
      { label: "scraper.py", desc: "Fallback automático Requests ↔ Selenium según tipo de sitio" },
      { label: "extractors.py", desc: "Parser BeautifulSoup con limpieza básica de HTML" },
      { label: "main.py", desc: "Orquestador + validación de calidad y guardado en CSV" },
    ],
    output: "perfiles_egreso_raw.csv",
    outputDesc: "64 perfiles de egreso en texto crudo sin procesar (de 47 IES)",
    tools: ["Python", "BeautifulSoup", "Selenium", "Pandas"],
  },
  {
    number: "02",
    title: "Procesamiento NLP",
    subtitle: "Limpieza Lingüística con spaCy",
    gradient: "from-blue-500 to-violet-500",
    glowColor: "rgba(59, 130, 246, 0.3)",
    borderColor: "border-blue-500/30",
    bgColor: "bg-blue-500/5",
    icon: "🧹",
    description:
      "Pipeline lingüístico que aplica tokenización, lematización y remoción estricta de Stopwords y confusores institucionales (términos ajenos a las competencias técnicas declaradas) usando la librería spaCy.",
    steps: [
      { label: "Tokenización", desc: "Segmentación del texto en unidades léxicas individuales" },
      { label: "Lematización", desc: "spaCy: reducción a raíz canónica — 'optimizando → optimizar'" },
      { label: "Stopwords custom", desc: "Eliminación de 'confusores institucionales': alumno, universidad, malla, egresado..." },
      { label: "Control de calidad", desc: "Perfiles con < 150 chars descartados automáticamente" },
    ],
    output: "perfiles_egreso_limpio_v1.csv",
    outputDesc: "Corpus normalizado listo para vectorización",
    tools: ["spaCy", "Regex"],
  },
  {
    number: "03",
    title: "Vectorización",
    subtitle: "TF-IDF y Diferenciación por Z-Score",
    gradient: "from-violet-500 to-purple-500",
    glowColor: "rgba(139, 92, 246, 0.3)",
    borderColor: "border-violet-500/30",
    bgColor: "bg-violet-500/5",
    icon: "🔢",
    description:
      "Se vectorizó el corpus mediante TF-IDF (1500 features). Además, se aplicó estadística Z-Score sobre frecuencias relativas para extraer el ADN léxico (Top 15 palabras) exclusivo de cada grado.",
    steps: [
      { label: "TfidfVectorizer", desc: "1500 features · Matriz densa 64 × 1500 dimensiones" },
      { label: "CountVectorizer", desc: "Frecuencias absolutas por grado para análisis Z-Score" },
      { label: "Z-Score (Keyness)", desc: "Identifica los términos estadísticamente distintivos por grado" },
      { label: "Top 15 por grado", desc: "ADN léxico exportado → top15_palabras_clave.csv" },
    ],
    output: "top15_palabras_clave.csv",
    outputDesc: "Términos distintivos por grado + matriz TF-IDF 63×1500",
    tools: ["Scikit-Learn (TfidfVectorizer, CountVectorizer)", "NumPy"],
  },
  {
    number: "04",
    title: "Análisis ML",
    subtitle: "Benchmark · SMOTE · Random Forest Ganador",
    gradient: "from-purple-500 to-emerald-400",
    glowColor: "rgba(16, 185, 129, 0.3)",
    borderColor: "border-emerald-500/30",
    bgColor: "bg-emerald-500/5",
    icon: "🎯",
    description:
      "SMOTE balancea las clases minoritarias. Un benchmark competitivo de 11 clasificadores (con Validación Cruzada Estratificada de 5-Folds) corona al Random Forest como modelo ganador con 92.79% de Accuracy. LDA se utiliza exclusivamente para la proyección espacial 2D; PCA permite la visualización no supervisada de la distribución léxica.",
    steps: [
      { label: "SMOTE", desc: "Balanceo sintético de clases minoritarias para evitar sesgo en el entrenamiento" },
      { label: "Benchmark 11 Alg.", desc: "Competencia de clasificadores: Random Forest, SVM, KNN, Gradient Boosting, AdaBoost y más" },
      { label: "Random Forest 🏆", desc: "Modelo ganador · 5-Fold Stratified CV · Accuracy 92.79%" },
      { label: "PCA + LDA → 2D", desc: "Reducción de dimensionalidad: PCA (no supervisado) vs LDA (supervisado) para visualizar separabilidad" },
    ],
    output: "resultados.json · visualizaciones .png",
    outputDesc: "Métricas del benchmark, proyecciones PCA/LDA, mapa de calor de similitud coseno y modelo_random_forest.joblib",
    tools: ["Scikit-Learn (RandomForestClassifier, PCA, LDA, SMOTE, StratifiedKFold)"],
  },
];

export default function MethodologySection() {
  const [activePhase, setActivePhase] = React.useState(0);
  const phase = PHASES[activePhase];

  return (
    <section id="metodologia" className="relative py-24 px-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="text-center mb-16 space-y-4">
          <span className="badge badge-purple">Arquitectura y Metodología</span>
          <h2 className="text-3xl md:text-5xl font-black text-white mt-4">
            El Pipeline de{" "}
            <span className="gradient-text-purple">Ciencia de Datos</span>
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            4 fases modulares en Python puro que transforman texto web crudo
            en evidencia estadística reproducible sobre la convergencia de mallas.
          </p>
        </div>

        {/* Phase selector tabs */}
        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {PHASES.map((p, i) => (
            <button
              key={i}
              id={`phase-tab-${i}`}
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

        {/* Phase detail card */}
        <div
          key={activePhase}
          className={`glass-card border ${phase.borderColor} ${phase.bgColor} p-8 md:p-10 animate-slide-up`}
        >
          <div className="grid md:grid-cols-2 gap-10">

            {/* Left: Info */}
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

              {/* Output */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-700/50">
                <p className="text-xs text-slate-500 uppercase tracking-widest mb-1">
                  Output generado
                </p>
                <p
                  className={`font-mono text-sm font-bold bg-gradient-to-r ${phase.gradient} bg-clip-text text-transparent`}
                >
                  📄 {phase.output}
                </p>
                <p className="text-xs text-slate-500 mt-1">{phase.outputDesc}</p>
              </div>
            </div>

            {/* Right: Steps */}
            <div className="space-y-3">
              <p className="text-xs text-slate-500 uppercase tracking-widest mb-4">
                Implementación técnica
              </p>
              {phase.steps.map((step, j) => (
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
                    <p className="font-mono text-sm font-semibold text-slate-200">
                      {step.label}
                    </p>
                    <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                      {step.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>

          </div>
        </div>

        {/* Pipeline flow summary */}
        <div className="mt-12 flex flex-wrap items-center justify-center gap-2 text-sm">
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

      </div>
    </section>
  );
}
