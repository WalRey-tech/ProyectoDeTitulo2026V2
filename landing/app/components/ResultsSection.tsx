"use client";

import React from "react";
import Image from "next/image";

// ─── Data real extraída de src/data/processed/resultados.json ──────────
// Se embebe directamente para Next.js App Router (no requiere fetch client-side)
const resultadosData = {
  n_por_grado: {
    Civil: 29,
    Informática: 23,
    Técnico: 15,
    Ejecución: 6,
    Total: 73,
  },
  metricas_supervisadas: {
    modelo: "Logistic Regression (Validación Cruzada 5-Folds)",
    accuracy: 0.5771,
    f1_macro: 0.5723,
  },
  analisis_homogeneidad: {
    similitud_civil_informatica: 0.6927,
    test_permutacion_p_valor: 0.0,
    significancia: "Estadísticamente significativo (p < 0.05)",
  },
  terminos_distintivos: {
    Civil: [
      "ingeniería civil", "acorde", "gerencial", "gerencia",
      "generación conocimiento", "forma colaborativo", "adquirido",
      "fundamento", "visión integral", "ámbito ciencia",
      "ámbito disciplina", "formativo desarrollar", "experiencial",
      "habilidad comunicación", "habilidad identificar",
    ],
    Ejecución: [
      "él facilidad", "área matemática", "diseño mantención",
      "eficiente resolver", "ambiente laboral", "efectivo eficiente",
      "principio sello", "alto capacidad", "diversidad capaz",
      "desempeñar institución", "programación recurso", "equipo respetar",
      "permanente tenacidad", "permitir desenvolver", "informática comprometido",
    ],
    Informática: [
      "doctorado ingeniería", "eficiencia operativo", "género diversidad",
      "egreso competencias", "entorno digital", "enmarcar",
      "especialización", "generar valor", "desarrollo implemento",
      "estándar industria", "tomar consideración", "género",
      "gestión tecnología", "informático eficiente", "industria organización",
    ],
    Técnico: [
      "web aplicación", "dato lan", "técnico informática",
      "existente", "estructura", "desarrollar sistema",
      "dato seguro", "escritorio", "metodología trabajo",
      "aplicación escritorio", "sistema web", "metodologíos",
      "mención ciberseguridad", "vida desarrollo", "operativo servidor",
    ],
  },
} as const;

// ─── Tipos derivados del objeto de datos ────────────────────────────────
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
  Técnico: {
    color: "#10b981",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/30",
    icon: "⚙️",
    desc: "Identidad operativa pura — el más diferenciado",
  },
  Ejecución: {
    color: "#f59e0b",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    icon: "🔩",
    desc: "Nicho puente entre técnico e ingeniería",
  },
};

// ─── Gráficos con metadata ────────────────────────────────────────────────
const CHARTS = [
  {
    id: "confusion",
    src: "/assets/matriz_confusion_LR.png",
    alt: "Matriz de Confusión — Logistic Regression",
    title: "Separabilidad vs. Convergencia de Grados",
    badge: "Regresión Logística · 5-Fold CV",
    badgeColor: "badge-purple",
    stat: `${(resultadosData.metricas_supervisadas.accuracy * 100).toFixed(1)}%`,
    statLabel: "Accuracy",
    statColor: "#8b5cf6",
    explanation:
      "Este modelo de Machine Learning (Accuracy 58%) demuestra que los grados 'Técnico' y 'Ejecución' poseen una identidad textual altamente separable y única. En contraste, existe una severa confusión entre 'Ingeniería Civil' e 'Ingeniería en Informática', revelando que las instituciones redactan sus competencias usando los mismos patrones semánticos.",
    insight: {
      label: "Hallazgo clave",
      text: "Civil ↔ Informática son los grados más confundidos por el modelo, confirmando la hipótesis de convergencia semántica.",
      color: "text-violet-400",
      bgColor: "bg-violet-500/10",
      borderColor: "border-violet-500/20",
    },
  },
  {
    id: "pca-lda",
    src: "/assets/proyeccion_pca_vs_lda.png",
    alt: "Proyección PCA vs LDA",
    title: "Proyección Vectorial 2D del Lenguaje Institucional",
    badge: "PCA + LDA · Reducción de Dimensionalidad",
    badgeColor: "badge-cyan",
    stat: "2D",
    statLabel: "Espacio proyectado",
    statColor: "#06b6d4",
    explanation:
      "Al forzar al algoritmo a encontrar diferencias (LDA), los perfiles Técnicos (rojo) se aíslan completamente debido a su lenguaje operativo. Sin embargo, Informática y Civil colapsan en el mismo espacio bidimensional, demostrando visualmente su convergencia estratégica.",
    insight: {
      label: "Evidencia visual",
      text: "Los vectores de Informática y Civil se superponen en el espacio LDA — son matemáticamente indistinguibles.",
      color: "text-cyan-400",
      bgColor: "bg-cyan-500/10",
      borderColor: "border-cyan-500/20",
    },
  },
  {
    id: "similitud",
    src: "/assets/similitud_centroides.png",
    alt: "Similitud de Centroides — Cosine Similarity",
    title: "Matriz de Homogeneidad",
    badge: "Similitud Coseno · Test de Permutación",
    badgeColor: "badge-emerald",
    stat: (resultadosData.analisis_homogeneidad.similitud_civil_informatica).toFixed(2),
    statLabel: "Civil ↔ Informática",
    statColor: "#10b981",
    explanation:
      "Validado con un Test de Permutación (p < 0.05), el análisis de similitud coseno confirma que la mayor superposición del mercado ocurre entre Civil e Informática (0.69), comprobando que comparten un núcleo de competencias casi idéntico.",
    insight: {
      label: "Significancia estadística",
      text: `p-valor = ${resultadosData.analisis_homogeneidad.test_permutacion_p_valor} — ${resultadosData.analisis_homogeneidad.significancia}`,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
      borderColor: "border-emerald-500/20",
    },
  },
];

// ─── Componente de chip de término ───────────────────────────────────────
function TermChip({
  term,
  rank,
  color,
}: {
  term: string;
  rank: number;
  color: string;
}) {
  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 hover:scale-105 cursor-default"
      style={{
        background: `${color}18`,
        border: `1px solid ${color}35`,
        color: color,
      }}
    >
      <span
        className="font-black text-xs opacity-60"
        style={{ minWidth: "14px" }}
      >
        {rank}
      </span>
      <span className="text-white/80">{term}</span>
    </div>
  );
}

// ─── Componente de tarjeta de gráfico ────────────────────────────────────
function ChartCard({
  chart,
  index,
}: {
  chart: (typeof CHARTS)[number];
  index: number;
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
              <span className={`badge text-xs ${chart.badgeColor}`}>
                {chart.badge}
              </span>
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
            <p
              className="text-2xl font-black leading-none"
              style={{ color: chart.statColor }}
            >
              {chart.stat}
            </p>
            <p className="text-xs text-slate-500 mt-0.5">{chart.statLabel}</p>
          </div>
        </div>
      </div>

      {/* Image */}
      <div className="relative bg-slate-950/40 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/30 to-transparent z-10 pointer-events-none" />
        <Image
          src={chart.src}
          alt={chart.alt}
          width={900}
          height={600}
          className="w-full h-auto object-contain transition-transform duration-500 group-hover:scale-[1.02]"
          style={{ maxHeight: "420px", objectFit: "contain" }}
        />
      </div>

      {/* Explanation */}
      <div className="p-6 space-y-4">
        <p className="text-slate-300 text-sm leading-relaxed">
          {chart.explanation}
        </p>
        {/* Insight box */}
        <div
          className={`p-4 rounded-xl ${chart.insight.bgColor} border ${chart.insight.borderColor}`}
        >
          <p
            className={`text-xs font-bold uppercase tracking-widest mb-1 ${chart.insight.color}`}
          >
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
  const terminos = resultadosData.terminos_distintivos;
  const distribucion = resultadosData.n_por_grado;
  const total = distribucion.Total;
  const grados = Object.keys(terminos) as GradoKey[];

  return (
    <section id="resultados" className="relative py-24 px-6">
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
            Tres métricas independientes convergen en la misma conclusión:
            la frontera semántica entre Civil e Informática se ha disuelto.
          </p>
        </div>

        {/* ── STATS ROW ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {(Object.entries(distribucion) as [string, number][])
            .filter(([k]) => k !== "Total")
            .map(([grado, n]) => {
              const cfg = GRADO_CONFIG[grado as GradoKey];
              const pct = ((n / total) * 100).toFixed(1);
              return (
                <button
                  key={grado}
                  onClick={() => setActiveGrado(grado as GradoKey)}
                  id={`grado-stat-${grado.toLowerCase()}`}
                  className={`glass-card p-5 text-left transition-all duration-300 border ${
                    activeGrado === grado
                      ? `${cfg.border}`
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
                  <p
                    className="text-3xl font-black mt-1"
                    style={{ color: cfg.color }}
                  >
                    {n}
                  </p>
                  <p className="text-xs text-slate-600 mt-1">
                    perfiles · {pct}%
                  </p>
                  <div
                    className="h-1 rounded-full mt-3 w-full"
                    style={{
                      background: `${cfg.color}20`,
                    }}
                  >
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${pct}%`,
                        background: cfg.color,
                      }}
                    />
                  </div>
                </button>
              );
            })}
        </div>

        {/* ── CHARTS GRID ── */}
        <div className="space-y-8">
          <h3 className="text-xl font-bold text-slate-300 flex items-center gap-3">
            <span className="w-1 h-6 rounded-full bg-gradient-to-b from-cyan-400 to-purple-500" />
            Visualizaciones del Modelo
          </h3>
          <div className="grid md:grid-cols-1 lg:grid-cols-1 gap-8">
            {/* First chart full width */}
            <ChartCard chart={CHARTS[0]} index={0} />
            {/* Next two side by side */}
            <div className="grid md:grid-cols-2 gap-8">
              <ChartCard chart={CHARTS[1]} index={1} />
              <ChartCard chart={CHARTS[2]} index={2} />
            </div>
          </div>
        </div>

        {/* ── TOP 15 PALABRAS — ADN LÉXICO ── */}
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
                      Ingeniería {g === "Técnico" ? "" : "en "}
                      {g}
                    </h4>
                    <p className="text-sm" style={{ color: cfg.color }}>
                      {cfg.desc}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {terminos[g].map((term, i) => (
                    <TermChip
                      key={i}
                      term={term}
                      rank={i + 1}
                      color={cfg.color}
                    />
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

        {/* ── CONCLUSIÓN FINAL — HIGH IMPACT ── */}
        <div
          id="conclusion"
          className="relative overflow-hidden rounded-2xl border border-violet-500/20"
          style={{
            background:
              "linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(6,182,212,0.05) 50%, rgba(16,185,129,0.05) 100%)",
          }}
        >
          {/* Decorative glow */}
          <div
            className="absolute -top-24 -left-24 w-96 h-96 rounded-full opacity-20 pointer-events-none"
            style={{
              background:
                "radial-gradient(circle, rgba(139,92,246,0.4) 0%, transparent 70%)",
            }}
          />
          <div
            className="absolute -bottom-24 -right-24 w-96 h-96 rounded-full opacity-20 pointer-events-none"
            style={{
              background:
                "radial-gradient(circle, rgba(6,182,212,0.4) 0%, transparent 70%)",
            }}
          />

          <div className="relative z-10 p-8 md:p-12 space-y-8">
            {/* Header */}
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <span className="badge badge-purple">Conclusión del Estudio</span>
                <span className="badge badge-cyan">p &lt; 0.05 — Validado estadísticamente</span>
              </div>
              <h2 className="text-2xl md:text-4xl font-black text-white leading-tight">
                La Ilusión de la Diferenciación:{" "}
                <span className="gradient-text-purple">
                  Crisis de Identidad en la Ingeniería Chilena
                </span>
              </h2>
            </div>

            {/* Body text */}
            <div className="space-y-5 text-slate-300 leading-relaxed max-w-4xl">
              <p className="text-base md:text-lg">
                Nuestra investigación concluye que el mercado de la educación
                superior en Chile sufre de una{" "}
                <span className="text-violet-400 font-semibold">
                  alta convergencia semántica impulsada por el marketing
                  institucional
                </span>
                . Al limpiar el ruido comercial y analizar puramente las
                competencias exigidas, descubrimos que la frontera entre la
                &lsquo;Ingeniería Civil&rsquo; y la &lsquo;Ingeniería en
                Informática&rsquo; prácticamente{" "}
                <span className="text-red-400 font-semibold">
                  ha desaparecido en el papel
                </span>
                . Las instituciones están inflando los perfiles informáticos con
                el mismo vocabulario gerencial, de liderazgo y estratégico que
                históricamente le pertenecía a la Ingeniería Civil, ofreciendo
                promesas formativas que, matemáticamente, son clones.
              </p>
              <p className="text-base md:text-lg">
                La verdadera diferenciación hoy solo sobrevive en los extremos:
                en el grado{" "}
                <span className="text-emerald-400 font-semibold">
                  &lsquo;Técnico&rsquo;
                </span>
                , que se mantiene fiel a su identidad puramente operativa y de
                implementación de herramientas, y en el casi extinto grado de{" "}
                <span className="text-amber-400 font-semibold">
                  &lsquo;Ejecución&rsquo;
                </span>
                , que conserva un nicho puente específico. Nuestra creencia es
                que el sistema requiere{" "}
                <span className="text-cyan-400 font-semibold">
                  urgente transparencia léxica
                </span>
                : los títulos están diferenciando por duración y costo, pero no
                por las competencias reales prometidas en sus perfiles de
                egreso.
              </p>
            </div>

            {/* Key metrics strip */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
              {[
                {
                  icon: "🎯",
                  label: "Accuracy del modelo",
                  value: `${(resultadosData.metricas_supervisadas.accuracy * 100).toFixed(1)}%`,
                  sub: "Limitado por la homogeneidad real",
                  color: "#8b5cf6",
                },
                {
                  icon: "📐",
                  label: "Similitud Civil ↔ Informática",
                  value: resultadosData.analisis_homogeneidad.similitud_civil_informatica.toFixed(2),
                  sub: "Cosine similarity sobre centroides",
                  color: "#06b6d4",
                },
                {
                  icon: "🧪",
                  label: "Validación estadística",
                  value: "p = 0.00",
                  sub: "Test de Permutación — Significativo",
                  color: "#10b981",
                },
              ].map((m, i) => (
                <div
                  key={i}
                  className="p-5 rounded-xl bg-slate-900/50 border border-white/5 text-center"
                >
                  <span className="text-2xl">{m.icon}</span>
                  <p className="text-xs text-slate-500 mt-2 uppercase tracking-widest">
                    {m.label}
                  </p>
                  <p
                    className="text-3xl font-black mt-1"
                    style={{ color: m.color }}
                  >
                    {m.value}
                  </p>
                  <p className="text-xs text-slate-600 mt-1">{m.sub}</p>
                </div>
              ))}
            </div>

            {/* Quote */}
            <blockquote className="border-l-4 border-violet-500/50 pl-6 mt-2">
              <p className="text-slate-400 italic text-sm md:text-base">
                &ldquo;Un sistema educativo que cobra precios distintos por
                competencias idénticas no está diferenciando formación —
                está diferenciando{" "}
                <span className="text-white not-italic font-semibold">
                  marketing
                </span>
                .&rdquo;
              </p>
              <footer className="text-xs text-slate-600 mt-2">
                — Hallazgo central, Tesis UDLA 2026
              </footer>
            </blockquote>
          </div>
        </div>

      </div>
    </section>
  );
}
