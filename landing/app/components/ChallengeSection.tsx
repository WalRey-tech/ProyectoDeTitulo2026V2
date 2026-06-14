"use client";

import React from "react";

// ─── Problemas reales del proyecto ─────────────────────────────────────────
const PROBLEMS = [
  {
    icon: "🔄",
    title: "Convergencia de Mallas",
    description:
      "Las instituciones de educación superior ofrecen múltiples variantes (Técnico, Ejecución, Informática, Civil). Sin embargo, redactan las competencias de estos perfiles copiando atributos gerenciales, creando la ilusión de diferenciación.",
    color: "border-red-500/20 bg-red-500/5",
    tag: "Problema",
    tagColor: "text-red-400 bg-red-400/10 border-red-400/20",
  },
  {
    icon: "⚖️",
    title: "Alta Subjetividad Manual",
    description:
      "Comparar y clasificar cientos de competencias a mano para descubrir si se copian o no, depende del sesgo del evaluador, careciendo de reproducibilidad científica.",
    color: "border-orange-500/20 bg-orange-500/5",
    tag: "Problema",
    tagColor: "text-orange-400 bg-orange-400/10 border-orange-400/20",
  },
  {
    icon: "📢",
    title: "Falta de Transparencia Léxica",
    description:
      "Los títulos informáticos en Chile se están diferenciando comercialmente por costo y duración, pero semánticamente prometen las mismas competencias estratégicas, confundiendo al mercado laboral.",
    color: "border-yellow-500/20 bg-yellow-500/5",
    tag: "Problema",
    tagColor: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20",
  },
];

// ─── Los 4 pilares de la solución ──────────────────────────────────────────
const SOLUTIONS = [
  {
    icon: "🤖",
    title: "Web Scraping Híbrido",
    description:
      "Motor híbrido (Requests + Selenium) que extrae texto de portales institucionales, recolectando 73 perfiles de egreso validados.",
    color: "border-cyan-500/20 bg-cyan-500/5",
  },
  {
    icon: "🧠",
    title: "NLP con spaCy",
    description:
      "Limpieza semántica eliminando ruido web y aplicando control de 'confusores institucionales' (ej. alumno, universidad, malla) que alteraban la realidad.",
    color: "border-blue-500/20 bg-blue-500/5",
  },
  {
    icon: "🧮",
    title: "Vectorización TF-IDF",
    description:
      "Transformación de los textos limpios en una matriz matemática densa de 1500 dimensiones para análisis de frecuencia y cálculo de Keyness (Z-Score).",
    color: "border-violet-500/20 bg-violet-500/5",
  },
  {
    icon: "🎯",
    title: "Machine Learning Supervisado",
    description:
      "Uso de Logistic Regression y Análisis Discriminante Lineal (LDA) para medir matemáticamente la separabilidad y convergencia de las carreras en un plano 2D.",
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
            La Crisis de Identidad en la{" "}
            <span className="gradient-text-cyan">Ingeniería Chilena</span>
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            La educación superior en Chile ofrece 4 grados de Informática con nombres
            distintos. Nuestro análisis estadístico revela que sus competencias son,
            matemáticamente, casi idénticas.
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

        {/* Divider */}
        <div className="flex flex-col items-center gap-3 mb-16">
          <div className="w-px h-12 bg-gradient-to-b from-transparent via-cyan-500/50 to-transparent" />
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center text-xl font-black text-white glow-cyan">
            ML
          </div>
          <div className="w-px h-12 bg-gradient-to-b from-transparent via-cyan-500/50 to-transparent" />
          <p className="text-cyan-400 font-semibold tracking-wide uppercase text-sm">
            Nuestra Solución
          </p>
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
            &ldquo;Al eliminar el ruido comercial, descubrimos que Civil e Informática
            comparten un núcleo de competencias con una similitud coseno de{" "}
            <span className="text-cyan-400 not-italic font-black">0.69</span>,
            validado estadísticamente con{" "}
            <span className="text-violet-400 not-italic font-black">p = 0.00</span>.&rdquo;
          </blockquote>
          <p className="text-slate-500 text-sm mt-4">— Hallazgo central del análisis</p>
        </div>

      </div>
    </section>
  );
}
