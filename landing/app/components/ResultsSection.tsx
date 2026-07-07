"use client";

import React from "react";
import Image from "next/image";

// ─── Data sincronizada con src/data/processed/resultados.json ────────────
const resultadosData = {
  n_por_grado: {
    Civil: 32,
    Informática: 25,
    Ejecución: 7,
    Total: 64,
  },
  metricas_supervisadas: {
    modelo: "Random Forest (Benchmark competitivo · 11 algoritmos · SMOTE · 5-Fold Stratified CV)",
    accuracy: 0.9279,
    benchmark_size: 11,
  },
  analisis_similitud: {
    similitud_civil_informatica: 0.7663,
    solapamiento_pct: 76.6,
    p_valor: 0.0,
    significancia: "Estadísticamente significativo (p < 0.05) · Ejecución presenta similitud < 50% con los otros grados",
  },
  terminos_distintivos: {
    Civil: [
      "ingeniería civil", "civil computación", "fundamento", "disciplina",
      "económico social", "tecnología información", "ingeniera",
      "identificar", "admisión", "comunicar él",
      "inglés", "institucional", "civil informático",
      "idioma inglés", "gestión proyecto",
    ],
    Ejecución: [
      "gestión", "institución", "eficiencia", "analizar",
      "sistema información", "adaptar", "visión cristiano", "tic",
      "adaptar él", "mundo",
      "cristiano", "comunidad", "ingeniero ejecución",
      "matemática", "distinguir",
    ],
    Informática: [
      "informática chile", "industrial", "formar profesional", "chile",
      "educación", "ciberseguridad", "sector productivo", "respuesta",
      "empresarial", "función",
      "in", "práctica industria", "necesidad organización",
      "phd", "continuo",
    ],
  },
} as const;

type GradoKey = keyof typeof resultadosData.terminos_distintivos;

// ─── Configuración visual por grado ─────────────────────────────────────
const GRADO_CONFIG: Record<
  GradoKey,
  { color: string; bg: string; border: string; icon: string; desc: string }
> = {
  Civil: {
    color: "#8b5cf6",
    bg: "bg-violet-500/10",
    border: "border-violet-500/30",
    icon: "🏛️",
    desc: "Foco gerencial y de liderazgo organizacional",
  },
  Informática: {
    color: "#06b6d4",
    bg: "bg-cyan-500/10",
    border: "border-cyan-500/30",
    icon: "💻",
    desc: "Lenguaje híbrido: tecnología + gestión empresarial",
  },
  Ejecución: {
    color: "#f59e0b",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    icon: "🔩",
    desc: "Vocabulario técnico aislado — el más diferenciado",
  },
};

// ─── Solo 2 gráficos finales ─────────────────────────────────────────
const CHARTS = [
  {
    id: "pca-lda",
    src: "/proyeccion_pca_vs_lda.png",
    alt: "Proyección 2D PCA vs LDA de Perfiles de Egreso",
    title: "Proyección 2D de Perfiles de Egreso (PCA vs LDA)",
    badge: "PCA · No Supervisado | LDA · Supervisado",
    badgeColor: "badge-cyan",
    stat: "92.79%",
    statLabel: "Accuracy Random Forest 🏆",
    statColor: "#8b5cf6",
    explanation:
      "En el panel izquierdo (PCA, no supervisado), los tres grados se proyectan formando un clúster central de alta densidad entre Civil e Informática, evidenciando una marcada homogeneidad léxica (76.6%). En el panel derecho (LDA, supervisado), el modelo es guiado para maximizar la separabilidad entre clases: Civil e Informática logran una separación parcial, mientras que Ejecución se posiciona de forma aislada. Este contraste explica el rendimiento excepcional del Random Forest (92.79%): la clasificación automática detecta las fronteras semánticas latentes que el análisis cualitativo humano no puede cuantificar.",
    insight: {
      label: "Interpretación clave",
      text: "La proyección PCA revela la alta homogeneidad curricular entre Civil e Informática. LDA demuestra que solo un enfoque supervisado —Random Forest, ganador de un benchmark de 11 algoritmos con SMOTE— puede identificar las fronteras semánticas latentes entre programas.",
      color: "text-violet-400",
      bgColor: "bg-violet-500/10",
      borderColor: "border-violet-500/20",
    },
  },
  {
    id: "similitud",
    src: "/similitud_centroides.png",
    alt: "Mapa de Calor de Similitud Coseno entre Grados",
    title: "Análisis de Similitud Semántica (Similitud Coseno)",
    badge: "Heatmap · Similitud Coseno",
    badgeColor: "badge-emerald",
    stat: "76.6%",
    statLabel: "Civil ↔ Informática (0.7663)",
    statColor: "#10b981",
    explanation:
      "Este mapa de calor revela el 76.6% (Similitud Coseno: 0.7663) de solapamiento semántico entre la Ingeniería Civil Informática y la Ingeniería en Informática, resultado estadísticamente significativo (p-valor = 0.0000, test de permutación). Por el contrario, la carrera de ‘Ejecución’ muestra una similitud inferior al 50% frente a las otras dos, confirmándose como el único grado con vocabulario verdaderamente aislado y una identidad semántica genuinamente diferenciada.",
    insight: {
      label: "Hallazgo central estadísticamente significativo",
      text: "Civil e Informática comparten un 76.6% de su vocabulario de competencias (p-valor = 0.0000). Solo la Ejecución mantiene un perfil léxico genuinamente diferenciado.",
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
      borderColor: "border-emerald-500/20",
    },
  },
];

// ─── Lightbox Modal ────────────────────────────────────────────────────────
function Lightbox({
  src,
  alt,
  title,
  onClose,
}: {
  src: string;
  alt: string;
  title: string;
  onClose: () => void;
}) {
  // Close on Escape key
  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", handler);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[999] flex items-center justify-center p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/90 backdrop-blur-sm" />

      {/* Container */}
      <div
        className="relative z-10 max-w-6xl w-full animate-slide-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header bar */}
        <div className="flex items-center justify-between mb-3 px-1">
          <p className="text-sm font-semibold text-slate-200">{title}</p>
          <button
            onClick={onClose}
            id="lightbox-close"
            className="w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-700 transition-all duration-200"
            aria-label="Cerrar imagen"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Image */}
        <div className="rounded-xl overflow-hidden border border-white/10 bg-slate-950/80 shadow-2xl">
          <Image
            src={src}
            alt={alt}
            width={1200}
            height={800}
            className="w-full h-auto object-contain"
            style={{ maxHeight: "80vh", objectFit: "contain" }}
            priority
          />
        </div>

        <p className="text-center text-xs text-slate-600 mt-3">
          Haz clic fuera de la imagen o presiona{" "}
          <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 font-mono text-slate-400">
            Esc
          </kbd>{" "}
          para cerrar
        </p>
      </div>
    </div>
  );
}

// ─── Term chip ────────────────────────────────────────────────────────────
function TermChip({ term, rank, color }: { term: string; rank: number; color: string }) {
  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 hover:scale-105 cursor-default"
      style={{
        background: `${color}18`,
        border: `1px solid ${color}35`,
      }}
    >
      <span className="font-black text-xs opacity-50" style={{ minWidth: "14px", color }}>
        {rank}
      </span>
      <span className="text-white/80">{term}</span>
    </div>
  );
}

// ─── Chart card with lightbox trigger ────────────────────────────────────
function ChartCard({
  chart,
  index,
  onOpenLightbox,
}: {
  chart: (typeof CHARTS)[number];
  index: number;
  onOpenLightbox: (src: string, alt: string, title: string) => void;
}) {
  return (
    <div className="glass-card border border-white/5 overflow-hidden group">
      {/* Header */}
      <div className="p-6 border-b border-white/5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-600 font-mono">
                Fig. {String(index + 1).padStart(2, "0")}
              </span>
              <span className={`badge text-xs ${chart.badgeColor}`}>{chart.badge}</span>
            </div>
            <h3 className="text-lg font-black text-white">{chart.title}</h3>
          </div>
          {/* Stat bubble */}
          <div
            className="text-center px-4 py-2 rounded-xl flex-shrink-0"
            style={{
              background: `${chart.statColor}15`,
              border: `1px solid ${chart.statColor}30`,
            }}
          >
            <p className="text-2xl font-black leading-none" style={{ color: chart.statColor }}>
              {chart.stat}
            </p>
            <p className="text-xs text-slate-500 mt-0.5">{chart.statLabel}</p>
          </div>
        </div>
      </div>

      {/* Image — clickable lightbox trigger */}
      <div
        className="relative bg-slate-950/40 overflow-hidden cursor-zoom-in"
        onClick={() => onOpenLightbox(chart.src, chart.alt, chart.title)}
        role="button"
        tabIndex={0}
        id={`chart-img-${chart.id}`}
        aria-label={`Ampliar: ${chart.title}`}
        onKeyDown={(e) => e.key === "Enter" && onOpenLightbox(chart.src, chart.alt, chart.title)}
      >
        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/30 to-transparent z-10 pointer-events-none" />
        {/* Zoom hint overlay */}
        <div className="absolute inset-0 z-20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
          <div className="bg-black/60 backdrop-blur-sm rounded-xl px-4 py-2 flex items-center gap-2 border border-white/10">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
            <span className="text-xs font-semibold text-white">Clic para ampliar</span>
          </div>
        </div>
        <Image
          src={chart.src}
          alt={chart.alt}
          width={900}
          height={600}
          className="w-full h-auto object-contain transition-transform duration-500 group-hover:scale-[1.02]"
          style={{ maxHeight: "380px", objectFit: "contain" }}
        />
      </div>

      {/* Explanation */}
      <div className="p-6 space-y-4">
        <p className="text-slate-300 text-sm leading-relaxed">{chart.explanation}</p>
        <div
          className={`p-4 rounded-xl ${chart.insight.bgColor} border ${chart.insight.borderColor}`}
        >
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${chart.insight.color}`}>
            🔍 {chart.insight.label}
          </p>
          <p className="text-sm text-slate-300">{chart.insight.text}</p>
        </div>
      </div>
    </div>
  );
}

// ─── Componente principal ─────────────────────────────────────────────────
export default function ResultsSection() {
  const [activeGrado, setActiveGrado] = React.useState<GradoKey>("Civil");
  const [lightbox, setLightbox] = React.useState<{
    src: string;
    alt: string;
    title: string;
  } | null>(null);

  const terminos = resultadosData.terminos_distintivos;
  const distribucion = resultadosData.n_por_grado;
  const total = distribucion.Total;
  const grados = Object.keys(terminos) as GradoKey[];

  const openLightbox = React.useCallback(
    (src: string, alt: string, title: string) => setLightbox({ src, alt, title }),
    []
  );
  const closeLightbox = React.useCallback(() => setLightbox(null), []);

  return (
    <section id="resultados" className="relative py-24 px-6">
      {/* Lightbox */}
      {lightbox && (
        <Lightbox
          src={lightbox.src}
          alt={lightbox.alt}
          title={lightbox.title}
          onClose={closeLightbox}
        />
      )}

      <div className="max-w-7xl mx-auto space-y-20">

        {/* ── HEADER ── */}
        <div className="text-center space-y-4">
          <span className="badge badge-cyan">Resultados Empíricos</span>
          <h2 className="text-3xl md:text-5xl font-black text-white mt-4">
            Evidencia{" "}
            <span className="gradient-text-cyan">Estadística</span>{" "}
            del Análisis
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            El modelo Random Forest —ganador de un benchmark competitivo de 11 algoritmos,
            optimizado con SMOTE— alcanza un <strong className="text-white">92.79% de Accuracy</strong> al
            detectar las sutiles diferencias semánticas que el análisis cualitativo humano
            no puede cuantificar: una convergencia léxica del <strong className="text-white">76.6%</strong>{" "}
            entre Civil e Informática, estadísticamente significativa (p-valor = 0.0000).
          </p>
        </div>

        {/* ── DISTRIBUTION STATS ── */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {(Object.entries(distribucion) as [string, number][])
            .filter(([k]) => k !== "Total")
            .map(([grado, n]) => {
              const cfg = GRADO_CONFIG[grado as GradoKey];
              const pct = ((n / total) * 100).toFixed(1);
              return (
                <button
                  key={grado}
                  id={`grado-stat-${grado.toLowerCase()}`}
                  onClick={() => setActiveGrado(grado as GradoKey)}
                  className={`glass-card p-5 text-left transition-all duration-300 border ${
                    activeGrado === grado
                      ? cfg.border
                      : "border-white/5 hover:border-white/10"
                  }`}
                  style={
                    activeGrado === grado
                      ? { boxShadow: `0 0 20px ${cfg.color}20` }
                      : {}
                  }
                >
                  <span className="text-2xl block mb-2">{cfg.icon}</span>
                  <p className="text-xs text-slate-500 font-medium">{grado}</p>
                  <p className="text-3xl font-black mt-1" style={{ color: cfg.color }}>
                    {n}
                  </p>
                  <p className="text-xs text-slate-600 mt-1">perfiles · {pct}%</p>
                  <div
                    className="h-1 rounded-full mt-3 w-full"
                    style={{ background: `${cfg.color}20` }}
                  >
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${pct}%`, background: cfg.color }}
                    />
                  </div>
                </button>
              );
            })}
        </div>

        {/* ── CHARTS ── */}
        <div className="space-y-8">
          <h3 className="text-xl font-bold text-slate-300 flex items-center gap-3">
            <span className="w-1 h-6 rounded-full bg-gradient-to-b from-cyan-400 to-purple-500" />
            Visualizaciones del Modelo
            <span className="text-xs font-normal text-slate-600 ml-1">
              — Clic en cada imagen para ampliar
            </span>
          </h3>
          {/* Ambos gráficos lado a lado en desktop, apilados en móvil */}
          <div className="grid md:grid-cols-2 gap-8">
            <ChartCard chart={CHARTS[0]} index={0} onOpenLightbox={openLightbox} />
            <ChartCard chart={CHARTS[1]} index={1} onOpenLightbox={openLightbox} />
          </div>
        </div>

        {/* ── ADN LÉXICO TOP 15 ── */}
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <h3 className="text-xl font-bold text-slate-300 flex items-center gap-3">
              <span className="w-1 h-6 rounded-full bg-gradient-to-b from-emerald-400 to-cyan-500" />
              ADN Léxico por Grado{" "}
              <span className="text-sm font-normal text-slate-500">
                (Top 15 términos TF-IDF distintivos)
              </span>
            </h3>
            {/* Tab selector */}
            <div className="flex flex-wrap gap-2">
              {grados.map((g) => {
                const cfg = GRADO_CONFIG[g];
                return (
                  <button
                    key={g}
                    id={`terms-tab-${g.toLowerCase()}`}
                    onClick={() => setActiveGrado(g)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 border ${
                      activeGrado === g
                        ? `${cfg.bg} ${cfg.border} text-white`
                        : "border-slate-800 text-slate-500 hover:border-slate-600 hover:text-slate-300"
                    }`}
                  >
                    <span>{cfg.icon}</span>
                    {g}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Terms panel */}
          {grados.map((g) => {
            const cfg = GRADO_CONFIG[g];
            if (g !== activeGrado) return null;
            return (
              <div
                key={g}
                className={`glass-card border ${cfg.border} ${cfg.bg} p-6 animate-slide-up`}
              >
                <div className="flex items-center gap-3 mb-5">
                  <span className="text-3xl">{cfg.icon}</span>
                  <div>
                    <h4 className="font-black text-white text-lg">
                      {g === "Civil"
                        ? "Ingeniería Civil Informática"
                        : g === "Ejecución"
                        ? "Ingeniería de Ejecución"
                        : "Ingeniería en Informática"}
                    </h4>
                    <p className="text-sm" style={{ color: cfg.color }}>
                      {cfg.desc}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {terminos[g].map((term, i) => (
                    <TermChip key={i} term={term} rank={i + 1} color={cfg.color} />
                  ))}
                </div>
                <p className="text-xs text-slate-600 mt-5">
                  Términos extraídos por Z-Score sobre TF-IDF · Modelo:{" "}
                  {resultadosData.metricas_supervisadas.modelo}
                </p>
              </div>
            );
          })}
        </div>

        {/* ── CONCLUSIÓN FINAL ── */}
        <div
          id="conclusion"
          className="relative overflow-hidden rounded-2xl border border-violet-500/20"
          style={{
            background:
              "linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(6,182,212,0.05) 50%, rgba(16,185,129,0.05) 100%)",
          }}
        >
          {/* Glows decorativos */}
          <div
            className="absolute -top-24 -left-24 w-96 h-96 rounded-full opacity-20 pointer-events-none"
            style={{ background: "radial-gradient(circle, rgba(139,92,246,0.4) 0%, transparent 70%)" }}
          />
          <div
            className="absolute -bottom-24 -right-24 w-96 h-96 rounded-full opacity-20 pointer-events-none"
            style={{ background: "radial-gradient(circle, rgba(6,182,212,0.4) 0%, transparent 70%)" }}
          />

          <div className="relative z-10 p-8 md:p-12 space-y-8">
            {/* Título */}
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <span className="badge badge-purple">Conclusión del Estudio</span>
                <span className="badge badge-cyan">Modelo Híbrido · Validación Estadística</span>
              </div>
              <h2 className="text-2xl md:text-4xl font-black text-white leading-tight">
                Convergencia Semántica{" "}
                <span className="gradient-text-purple">Cuantificada</span>
              </h2>
            </div>

            {/* Cuerpo */}
            <div className="space-y-5 text-slate-300 leading-relaxed max-w-4xl">
              <p className="text-base md:text-lg">
                El pipeline de Machine Learning demuestra científicamente una{" "}
                <span className="text-violet-400 font-semibold">
                  alta homogeneidad semántica en la educación superior TI en Chile
                </span>
                . Las Ingenierías Civil e Informática comparten un núcleo léxico casi idéntico
                (76.6%, Similitud Coseno: 0.7663), hallazgo que resulta{" "}
                <span className="text-cyan-400 font-semibold">
                  estadísticamente significativo (p-valor = 0.0000)
                </span>
                {" "}según el test de permutación aplicado, lo que descarta la posibilidad
                de que dicha convergencia sea producto del azar.
              </p>
              <p className="text-base md:text-lg">
                Por su parte, la{" "}
                <span className="text-amber-400 font-semibold">Ingeniería de Ejecución</span>{" "}
                se consolida como el único grado con una identidad semántica verdaderamente
                diferenciada (similitud &lt; 50% respecto a los otros grados).
                El modelo Random Forest, ganador de un benchmark competitivo de 11 algoritmos
                optimizado con SMOTE,{" "}
                <span className="text-emerald-400 font-semibold">
                  alcanza un 92.79% de Accuracy
                </span>
                , transformando la incertidumbre cualitativa en certeza cuantitativa y reproducible.
              </p>
            </div>

            {/* Métricas clave */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
              {[
                {
                  icon: "🏆",
                  label: "Modelo Ganador",
                  value: "Random Forest",
                  sub: "Benchmark 11 Alg. · SMOTE · 92.79% Accuracy (5-Fold Stratified CV)",
                  color: "#8b5cf6",
                },
                {
                  icon: "📀",
                  label: "Convergencia Léxica Civil ↔ Informática",
                  value: "76.6%",
                  sub: "Similitud Coseno 0.7663 · p-valor = 0.0000",
                  color: "#10b981",
                },
                {
                  icon: "🔬",
                  label: "Grado Diferenciado",
                  value: "Ejecución",
                  sub: "Similitud < 50% · Identidad semántica significativamente diferenciada",
                  color: "#f59e0b",
                },
              ].map((m, i) => (
                <div
                  key={i}
                  className="p-5 rounded-xl bg-slate-900/50 border border-white/5 text-center"
                >
                  <span className="text-2xl">{m.icon}</span>
                  <p className="text-xs text-slate-500 mt-2 uppercase tracking-widest">{m.label}</p>
                  <p className="text-3xl font-black mt-1" style={{ color: m.color }}>
                    {m.value}
                  </p>
                  <p className="text-xs text-slate-600 mt-1">{m.sub}</p>
                </div>
              ))}
            </div>

            {/* Cita oficial */}
            <blockquote className="border-l-4 border-violet-500/50 pl-6">
              <p className="text-slate-400 italic text-sm md:text-base">
                &ldquo;Este ecosistema analítico automatizado supera las auditorías humanas,
                pasando de la{" "}
                <span className="text-white not-italic font-semibold">incertidumbre cualitativa</span>{" "}
                a la{" "}
                <span className="text-cyan-400 not-italic font-semibold">certeza cuantitativa</span>.&rdquo;
              </p>
              <footer className="text-xs text-slate-600 mt-2">
                &mdash; Conclusión central, Proyecto de Título UDLA 2026
              </footer>
            </blockquote>
          </div>
        </div>

      </div>
    </section>
  );
}
