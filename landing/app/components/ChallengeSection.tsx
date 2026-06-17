"use client";

import React from "react";

// ─── Problemas reales del proyecto ───────────────────────────────────────────
const PROBLEMS = [
  {
    icon: "🔄",
    title: "Alta Homogeneidad Curricular",
    description:
      "Las tres carreras informáticas analizadas (Ing. Civil, Ing. en Informática e Ing. de Ejecución) presentan una marcada convergencia léxica en sus perfiles de egreso, lo que dificulta una diferenciación objetiva entre sus competencias declaradas.",
    color: "border-red-500/20 bg-red-500/5",
    tag: "Problema",
    tagColor: "text-red-400 bg-red-400/10 border-red-400/20",
  },
  {
    icon: "⚖️",
    title: "Sesgo en la Revisión Manual",
    description:
      "La comparación cualitativa de perfiles de egreso carece de reproducibilidad científica: distintos evaluadores pueden llegar a conclusiones distintas sobre la similitud o diferencia entre programas académicos.",
    color: "border-orange-500/20 bg-orange-500/5",
    tag: "Problema",
    tagColor: "text-orange-400 bg-orange-400/10 border-orange-400/20",
  },
  {
    icon: "📊",
    title: "Ausencia de Métricas Cuantitativas",
    description:
      "Hasta ahora no existían métricas estandarizadas que permitieran medir objetivamente el grado de estandarización lingüística entre programas informáticos de distintas instituciones de educación superior en Chile.",
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
      "Motor híbrido (Requests + Selenium) que extrae texto de portales institucionales, recolectando y depurando 63 perfiles de egreso válidos de 47 IES chilenas.",
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
    title: "Benchmark + SVM Ganador",
    description:
      "SMOTE balancea las clases minoritarias. Un benchmark de 11 algoritmos determina al SVM (Kernel RBF) como el modelo ganador con un 87.58% de Accuracy en Validación Cruzada Estratificada.",
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
            Convergencia Semántica en la{" "}
            <span className="gradient-text-cyan">Educación Superior TI</span>
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            La educación superior en Chile ofrece 3 grados informáticos con denominaciones distintas.
            Este estudio cuantifica objetivamente el nivel de homogeneidad curricular
            entre sus perfiles de egreso mediante métodos computacionales reproducibles.
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
            &ldquo;El modelo híbrido demuestra científicamente que la Ingeniería Civil e Informática
            comparten un núcleo léxico con una convergencia de{" "}
            <span className="text-cyan-400 not-italic font-black">76.8% (0.768)</span>,
            mientras que la Ingeniería de Ejecución consolida una identidad semántica
            verdaderamente especializada y pragmática.&rdquo;
          </blockquote>
          <p className="text-slate-500 text-sm mt-4">— Hallazgo central del estudio, Proyecto de Título UDLA 2026</p>
        </div>

      </div>
    </section>
  );
}
