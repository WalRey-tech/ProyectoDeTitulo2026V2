"use client";

import React from "react";

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
      "Motor de scraping inteligente que adapta su estrategia según el sitio. Usa Requests para sitios estáticos y Selenium para páginas con JavaScript dinámico.",
    steps: [
      { label: "config.py", desc: "32+ universidades configuradas con selectores CSS" },
      { label: "scraper.py", desc: "Fallback automático Requests ↔ Selenium" },
      { label: "extractors.py", desc: "Parser BeautifulSoup con limpieza básica" },
      { label: "main.py", desc: "Orquestador + validación y guardado en CSV" },
    ],
    output: "perfiles_egreso_raw.csv",
    outputDesc: "Dataset bruto con texto sin procesar",
    tools: ["BeautifulSoup", "Selenium", "Requests", "Pandas"],
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
      "Pipeline de limpieza en dos capas. Primero se normaliza la estructura con Regex y luego spaCy aplica lematización, eliminando stopwords académicas personalizadas.",
    steps: [
      { label: "Capa Estructural", desc: "Regex: minúsculas, sin símbolos/números, género neutralizado" },
      { label: "Capa Lingüística", desc: "spaCy: lematización 'optimizando → optimizar'" },
      { label: "Stopwords Custom", desc: "47 palabras académicas filtradas (\"universidad\", \"egresado\"...)" },
      { label: "Control de Calidad", desc: "Perfiles < 150 chars descartados automáticamente" },
    ],
    output: "perfiles_limpios.csv",
    outputDesc: "Texto normalizado listo para vectorización",
    tools: ["spaCy", "es_core_news_sm", "Regex", "Pandas"],
  },
  {
    number: "02b",
    title: "Vectorización",
    subtitle: "Embeddings Semánticos Transformer",
    gradient: "from-violet-500 to-purple-500",
    glowColor: "rgba(139, 92, 246, 0.3)",
    borderColor: "border-violet-500/30",
    bgColor: "bg-violet-500/5",
    icon: "🔢",
    description:
      "Cada perfil de egreso se convierte en un vector numérico de 384 dimensiones usando un modelo Transformer multilingüe que captura el significado semántico profundo.",
    steps: [
      { label: "Modelo", desc: "paraphrase-multilingual-MiniLM-L12-v2 (Hugging Face)" },
      { label: "Dimensionalidad", desc: "384 coordenadas por perfil, normalized_embeddings=True" },
      { label: "Matriz resultante", desc: "32 perfiles × 384 dimensiones = espacio semántico" },
      { label: "Persistencia", desc: "Guardado en vectores_perfiles.npy (NumPy binary)" },
    ],
    output: "vectores_perfiles.npy",
    outputDesc: "Matriz numérica de alta dimensionalidad",
    tools: ["SentenceTransformers", "NumPy", "Hugging Face"],
  },
  {
    number: "03",
    title: "Análisis ML",
    subtitle: "K-Means + PCA + Validación",
    gradient: "from-purple-500 to-emerald-400",
    glowColor: "rgba(16, 185, 129, 0.3)",
    borderColor: "border-emerald-500/30",
    bgColor: "bg-emerald-500/5",
    icon: "🎯",
    description:
      "Algoritmo K-Means (k=4) agrupa los vectores por similitud semántica. PCA reduce las 384 dimensiones a 2D para visualización. El Silhouette Score valida la calidad del agrupamiento.",
    steps: [
      { label: "K-Means (k=4)", desc: "n_init=10, random_state=42 para reproducibilidad" },
      { label: "PCA 2D", desc: "Comprime 384 → 2 dimensiones preservando varianza máxima" },
      { label: "TF-IDF", desc: "Extrae las 8 palabras clave de cada cluster para su interpretación" },
      { label: "Silhouette Score", desc: "Métrica de cohesión y separación entre clusters" },
    ],
    output: "perfiles_con_clusters.csv",
    outputDesc: "Dataset final con etiquetas de cluster asignadas",
    tools: ["scikit-learn", "K-Means", "PCA", "TF-IDF", "Matplotlib", "Seaborn"],
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
            <span className="gradient-text-purple">Inteligencia Artificial</span>
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            4 fases modulares e independientes que transforman texto web crudo
            en conocimiento estructurado y accionable.
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
                  <p className={`text-sm font-semibold bg-gradient-to-r ${phase.gradient} bg-clip-text text-transparent`}>
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
                <p className="text-xs text-slate-500 uppercase tracking-widest mb-1">Output generado</p>
                <p className={`font-mono text-sm font-bold bg-gradient-to-r ${phase.gradient} bg-clip-text text-transparent`}>
                  📄 {phase.output}
                </p>
                <p className="text-xs text-slate-500 mt-1">{phase.outputDesc}</p>
              </div>
            </div>

            {/* Right: Steps */}
            <div className="space-y-3">
              <p className="text-xs text-slate-500 uppercase tracking-widest mb-4">Implementación técnica</p>
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
                    <p className="font-mono text-sm font-semibold text-slate-200">{step.label}</p>
                    <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{step.desc}</p>
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
                    ? `border-cyan-500/50 text-cyan-400 bg-cyan-500/10`
                    : "border-slate-800 text-slate-500 hover:text-slate-300 hover:border-slate-600"
                }`}
              >
                <span>{p.icon}</span>
                <span className="font-medium">{p.title}</span>
              </button>
              {i < PHASES.length - 1 && (
                <svg className="w-4 h-4 text-slate-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </section>
  );
}
