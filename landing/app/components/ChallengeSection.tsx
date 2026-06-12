"use client";

import React from "react";

const PROBLEMS = [
  {
    icon: "🕐",
    title: "Proceso Manual Lento",
    description:
      "Un académico tardaría semanas en revisar, leer y comparar más de 32 perfiles de egreso de distintas instituciones para identificar tendencias.",
    color: "border-red-500/20 bg-red-500/5",
    tag: "Problema",
    tagColor: "text-red-400 bg-red-400/10 border-red-400/20",
  },
  {
    icon: "⚖️",
    title: "Alta Subjetividad",
    description:
      "Las categorizaciones manuales dependen del criterio de quien las realiza, generando inconsistencias entre evaluadores y falta de reproducibilidad científica.",
    color: "border-orange-500/20 bg-orange-500/5",
    tag: "Problema",
    tagColor: "text-orange-400 bg-orange-400/10 border-orange-400/20",
  },
  {
    icon: "📊",
    title: "Falta de Escala",
    description:
      "El método tradicional no escala. Con el crecimiento de la oferta académica en Chile, se necesita un sistema que procese cientos de registros en minutos.",
    color: "border-yellow-500/20 bg-yellow-500/5",
    tag: "Problema",
    tagColor: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20",
  },
];

const SOLUTIONS = [
  {
    icon: "🤖",
    title: "Web Scraping Automatizado",
    description: "Motor híbrido (Requests + Selenium) que extrae texto de portales institucionales respetando la dinámica de cada sitio web.",
    color: "border-cyan-500/20 bg-cyan-500/5",
  },
  {
    icon: "🧠",
    title: "NLP con spaCy",
    description: "Limpieza semántica en dos capas: Regex estructural + lematización lingüística, eliminando ruido y reduciendo el vocabulario a su esencia.",
    color: "border-blue-500/20 bg-blue-500/5",
  },
  {
    icon: "🔢",
    title: "Embeddings Transformers",
    description: "Modelo multilingüe paraphrase-multilingual-MiniLM-L12-v2 que convierte cada perfil en un vector de 384 dimensiones capturando su significado profundo.",
    color: "border-purple-500/20 bg-purple-500/5",
  },
  {
    icon: "🎯",
    title: "K-Means + PCA",
    description: "Algoritmo K-Means para agrupar vectores similares y PCA para proyectar el espacio de 384D a 2D, creando el mapa semántico visible.",
    color: "border-emerald-500/20 bg-emerald-500/5",
  },
];

export default function ChallengeSection() {
  return (
    <section id="desafio" className="relative py-24 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16 space-y-4">
          <span className="badge badge-cyan">El Desafío</span>
          <h2 className="text-3xl md:text-5xl font-black text-white mt-4">
            ¿Por qué la revisión manual{" "}
            <span className="gradient-text-cyan">ya no es suficiente?</span>
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            La educación superior en Chile ofrece decenas de variantes de Ingeniería en Informática.
            Comparar sus perfiles de egreso manualmente es inviable a escala. La IA lo resuelve.
          </p>
        </div>

        {/* Problems */}
        <div className="grid md:grid-cols-3 gap-6 mb-20">
          {PROBLEMS.map((p, i) => (
            <div
              key={i}
              className={`glass-card p-6 border ${p.color} group transition-all duration-300 hover:scale-[1.02]`}
            >
              <div className="flex items-start gap-4">
                <span className="text-3xl flex-shrink-0">{p.icon}</span>
                <div>
                  <div className={`badge text-xs mb-3 ${p.tagColor} border`}>{p.tag}</div>
                  <h3 className="text-lg font-bold text-white mb-2">{p.title}</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">{p.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Divider with arrow */}
        <div className="flex flex-col items-center gap-3 mb-16">
          <div className="w-px h-12 bg-gradient-to-b from-transparent via-cyan-500/50 to-transparent" />
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center text-xl font-black text-white glow-cyan">
            IA
          </div>
          <div className="w-px h-12 bg-gradient-to-b from-transparent via-cyan-500/50 to-transparent" />
          <p className="text-cyan-400 font-semibold tracking-wide uppercase text-sm">Nuestra Solución</p>
        </div>

        {/* Solutions */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
          {SOLUTIONS.map((s, i) => (
            <div
              key={i}
              className={`glass-card p-6 border ${s.color} group transition-all duration-300 hover:scale-[1.03] hover:-translate-y-1`}
            >
              <span className="text-3xl block mb-4">{s.icon}</span>
              <h3 className="text-base font-bold text-white mb-2">{s.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{s.description}</p>
            </div>
          ))}
        </div>

        {/* Impact quote */}
        <div className="mt-16 glass-card p-8 border border-cyan-500/20 bg-cyan-500/5 text-center">
          <blockquote className="text-xl md:text-2xl font-semibold text-white italic max-w-3xl mx-auto leading-relaxed">
            &ldquo;El sistema procesa en{" "}
            <span className="text-cyan-400 not-italic font-black">minutos</span> lo que un equipo
            académico tardaría{" "}
            <span className="text-orange-400 not-italic font-black">semanas</span> en analizar
            manualmente.&rdquo;
          </blockquote>
          <p className="text-slate-500 text-sm mt-4">— Hipótesis central del proyecto</p>
        </div>
      </div>
    </section>
  );
}
