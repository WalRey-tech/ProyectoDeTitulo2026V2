"use client";

import React from "react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

const CLUSTERS = [
  {
    id: 0,
    name: "Ciencia de Datos y\nDesarrollo de Software",
    shortName: "Datos & Software",
    percentage: 34.4,
    count: 11,
    color: "#06b6d4",
    keywords: ["datos", "algoritmo", "software", "análisis", "sistema", "desarrollo", "programación", "base"],
    description:
      "El perfil más común. Combina habilidades de análisis de datos con desarrollo de software, reflejando la demanda del mercado por profesionales que dominan ambas disciplinas.",
    icon: "💻",
    institutions: ["INACAP", "Duoc UC", "AIEP", "CFT Santa Catalina"],
  },
  {
    id: 1,
    name: "Gestión Tecnológica",
    shortName: "Gestión TI",
    percentage: 31.2,
    count: 10,
    color: "#8b5cf6",
    keywords: ["gestión", "proyecto", "empresa", "negocio", "organización", "proceso", "recurso", "tecnología"],
    description:
      "Enfoque en la administración de proyectos tecnológicos y la alineación de TI con los objetivos de negocio. Perfil orientado a roles de liderazgo y gestión.",
    icon: "📊",
    institutions: ["UAI", "UDD", "UANDES", "UST"],
  },
  {
    id: 2,
    name: "Ingeniería de Software y\nComputación Aplicada",
    shortName: "Ing. Software",
    percentage: 28.1,
    count: 9,
    color: "#3b82f6",
    keywords: ["sistema", "red", "arquitectura", "infraestructura", "servidor", "computación", "código", "aplicación"],
    description:
      "Perfil con fuerte base matemática e ingenieril. Orientado al diseño de sistemas complejos, arquitecturas de software y computación de alto rendimiento.",
    icon: "⚙️",
    institutions: ["UCH", "USACH", "PUC", "UTFSM"],
  },
  {
    id: 3,
    name: "Ciberseguridad y\nGestión de Riesgos",
    shortName: "Ciberseguridad",
    percentage: 6.2,
    count: 2,
    color: "#10b981",
    keywords: ["seguridad", "riesgo", "auditoría", "norma", "cumplimiento", "protección", "amenaza", "vulnerabilidad"],
    description:
      "Nicho emergente y de alta demanda. Aunque representa el grupo más pequeño, su escasez indica una oportunidad de mercado y una brecha en la oferta académica chilena.",
    icon: "🔐",
    institutions: ["UDLA", "Universidad Andrés Bello"],
  },
];

const PIE_DATA = CLUSTERS.map((c) => ({
  name: c.shortName,
  value: c.percentage,
  count: c.count,
}));

const BAR_DATA = CLUSTERS.map((c) => ({
  name: c.shortName,
  porcentaje: c.percentage,
  programas: c.count,
}));

// Custom Tooltip for Pie
const PieTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ name: string; value: number; payload: { count: number } }> }) => {
  if (active && payload && payload.length) {
    const d = payload[0];
    const cluster = CLUSTERS.find((c) => c.shortName === d.name);
    return (
      <div className="bg-gray-900/95 border border-white/10 rounded-xl p-4 shadow-xl max-w-xs">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-lg">{cluster?.icon}</span>
          <span className="font-bold text-white text-sm">{d.name}</span>
        </div>
        <p className="text-2xl font-black" style={{ color: cluster?.color }}>
          {d.value}%
        </p>
        <p className="text-slate-400 text-xs mt-1">{d.payload.count} programas formativos</p>
      </div>
    );
  }
  return null;
};

// Custom Tooltip for Bar
const BarTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name: string }>; label?: string }) => {
  if (active && payload && payload.length) {
    const cluster = CLUSTERS.find((c) => c.shortName === label);
    return (
      <div className="bg-gray-900/95 border border-white/10 rounded-xl p-4 shadow-xl">
        <p className="font-semibold text-white text-sm mb-2">{label}</p>
        <p className="text-lg font-black" style={{ color: cluster?.color }}>
          {payload[0].value}%
        </p>
        <p className="text-slate-400 text-xs">del total de programas</p>
      </div>
    );
  }
  return null;
};

const CustomLegend = () => (
  <div className="flex flex-wrap justify-center gap-4 mt-4">
    {CLUSTERS.map((c) => (
      <div key={c.id} className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: c.color }} />
        <span className="text-xs text-slate-400">{c.shortName}</span>
      </div>
    ))}
  </div>
);

export default function ResultsSection() {
  const [activeCluster, setActiveCluster] = React.useState(0);
  const cluster = CLUSTERS[activeCluster];

  return (
    <section id="resultados" className="relative py-24 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16 space-y-4">
          <span className="badge badge-emerald">Resultados del Análisis</span>
          <h2 className="text-3xl md:text-5xl font-black text-white mt-4">
            Los{" "}
            <span className="gradient-text-cyan">4 Clústeres</span>{" "}
            Descubiertos
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            El algoritmo K-Means identificó 4 grupos semánticos claros y diferenciables
            en la oferta de Informática en Chile. Estos son los hallazgos empíricos del modelo.
          </p>
        </div>

        {/* Charts row */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {/* Pie Chart */}
          <div className="glass-card p-6 border border-white/5">
            <h3 className="text-base font-bold text-white mb-1">Distribución por Clúster</h3>
            <p className="text-xs text-slate-500 mb-6">Porcentaje de programas en cada grupo</p>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={PIE_DATA}
                  cx="50%"
                  cy="50%"
                  innerRadius={70}
                  outerRadius={110}
                  paddingAngle={4}
                  dataKey="value"
                  animationBegin={0}
                  animationDuration={800}
                >
                  {PIE_DATA.map((_, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={CLUSTERS[index].color}
                      opacity={activeCluster === index ? 1 : 0.6}
                      stroke={activeCluster === index ? CLUSTERS[index].color : "transparent"}
                      strokeWidth={activeCluster === index ? 2 : 0}
                      style={{ cursor: "pointer", filter: activeCluster === index ? `drop-shadow(0 0 8px ${CLUSTERS[index].color}80)` : "none" }}
                      onClick={() => setActiveCluster(index)}
                    />
                  ))}
                </Pie>
                <Tooltip content={<PieTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <CustomLegend />
          </div>

          {/* Bar Chart */}
          <div className="glass-card p-6 border border-white/5">
            <h3 className="text-base font-bold text-white mb-1">Comparativa de Tamaño</h3>
            <p className="text-xs text-slate-500 mb-6">Porcentaje de cada clúster respecto al total</p>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={BAR_DATA} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip content={<BarTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                <Bar dataKey="porcentaje" radius={[6, 6, 0, 0]} animationDuration={800}>
                  {BAR_DATA.map((_, index) => (
                    <Cell
                      key={`bar-${index}`}
                      fill={CLUSTERS[index].color}
                      opacity={activeCluster === index ? 1 : 0.5}
                      style={{ cursor: "pointer" }}
                      onClick={() => setActiveCluster(index)}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Cluster selector */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          {CLUSTERS.map((c, i) => (
            <button
              key={i}
              id={`cluster-btn-${i}`}
              onClick={() => setActiveCluster(i)}
              className={`p-4 rounded-xl border text-left transition-all duration-300 ${
                activeCluster === i
                  ? "border-current bg-current/10"
                  : "border-slate-800 bg-slate-900/30 hover:border-slate-600"
              }`}
              style={activeCluster === i ? { borderColor: c.color, color: c.color, backgroundColor: `${c.color}15` } : {}}
            >
              <span className="text-2xl block mb-2">{c.icon}</span>
              <p className={`text-xs font-bold leading-tight ${activeCluster === i ? "" : "text-slate-400"}`}>
                {c.shortName}
              </p>
              <p className={`text-xl font-black mt-1 ${activeCluster === i ? "" : "text-slate-300"}`}
                style={activeCluster === i ? { color: c.color } : {}}>
                {c.percentage}%
              </p>
              <p className="text-xs text-slate-500">{c.count} programas</p>
            </button>
          ))}
        </div>

        {/* Cluster detail */}
        <div
          key={activeCluster}
          className="glass-card border p-8 animate-slide-up"
          style={{ borderColor: `${cluster.color}30`, backgroundColor: `${cluster.color}08` }}
        >
          <div className="grid md:grid-cols-3 gap-8">
            {/* Main info */}
            <div className="md:col-span-2 space-y-5">
              <div className="flex items-center gap-4">
                <span className="text-4xl">{cluster.icon}</span>
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-widest">Clúster {cluster.id}</p>
                  <h3 className="text-xl font-black text-white whitespace-pre-line">{cluster.name}</h3>
                </div>
              </div>
              <p className="text-slate-300 leading-relaxed">{cluster.description}</p>

              {/* Keywords */}
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-widest mb-3">Top Keywords (TF-IDF)</p>
                <div className="flex flex-wrap gap-2">
                  {cluster.keywords.map((kw, i) => (
                    <span
                      key={i}
                      className="px-3 py-1.5 text-xs font-semibold rounded-lg"
                      style={{
                        background: `${cluster.color}20`,
                        border: `1px solid ${cluster.color}40`,
                        color: cluster.color,
                      }}
                    >
                      #{kw}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Stats sidebar */}
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/50 text-center">
                <p className="text-xs text-slate-500 mb-1">Participación</p>
                <p className="text-4xl font-black" style={{ color: cluster.color }}>
                  {cluster.percentage}%
                </p>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/50 text-center">
                <p className="text-xs text-slate-500 mb-1">Programas</p>
                <p className="text-4xl font-black text-white">{cluster.count}</p>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/50">
                <p className="text-xs text-slate-500 mb-2">Ejemplos de institución</p>
                <ul className="space-y-1">
                  {cluster.institutions.map((inst, i) => (
                    <li key={i} className="text-xs text-slate-400 flex items-center gap-2">
                      <span className="w-1 h-1 rounded-full flex-shrink-0" style={{ background: cluster.color }} />
                      {inst}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* CTA to dashboard */}
        <div className="mt-12 text-center">
          <div className="inline-block glass-card p-8 border border-cyan-500/20 bg-cyan-500/5">
            <p className="text-slate-400 text-sm mb-2">Exploración interactiva completa</p>
            <h3 className="text-xl font-bold text-white mb-4">
              ¿Quieres ver el mapa semántico 2D completo?
            </h3>
            <a
              id="dashboard-link"
              href="/dashboard_clusters.html"
              className="btn-primary inline-flex items-center gap-3"
              onClick={(e) => {
                e.preventDefault();
                alert("🚀 En producción, este botón te llevará al dashboard interactivo generado por Plotly con todos los clusters visualizados en 2D.\n\nArchivo: dashboard_clusters.html");
              }}
            >
              <span className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
                Abrir Dashboard Interactivo
              </span>
            </a>
            <p className="text-xs text-slate-600 mt-3">Powered by Plotly · dashboard_clusters.html</p>
          </div>
        </div>
      </div>
    </section>
  );
}
