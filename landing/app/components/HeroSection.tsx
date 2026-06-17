"use client";

import React from "react";

const STATS = [
  { value: "63", suffix: "", label: "Perfiles Depurados · 47 IES", color: "from-cyan-400 to-blue-500" },
  { value: "87.58", suffix: "%", label: "Accuracy SVM (Modelo Ganador)", color: "from-violet-400 to-purple-500" },
  { value: "76.8", suffix: "%", label: "Solapamiento Civil ↔ Informática", color: "from-amber-400 to-orange-500" },
  { value: "11", suffix: "Alg.", label: "Benchmark de Clasificadores", color: "from-emerald-400 to-teal-500" },
];

const TECH_TAGS = [
  "Python", "spaCy", "TF-IDF", "SVM · Kernel RBF", "SMOTE",
  "PCA", "LDA", "BeautifulSoup", "Selenium", "Next.js"
];

export default function HeroSection() {
  return (
    <section
      id="hero"
      className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-24 pb-16 overflow-hidden"
    >
      {/* Decorative rings */}
      <div className="orbit-ring w-[600px] h-[600px] opacity-20 animate-spin-slow" />
      <div
        className="orbit-ring w-[850px] h-[850px] opacity-10"
        style={{ animation: "spin-slow 35s linear infinite reverse" }}
      />

      {/* Floating dots */}
      <div className="absolute top-1/4 left-1/6 w-2 h-2 rounded-full bg-cyan-400 opacity-60 animate-float" />
      <div className="absolute top-1/3 right-1/5 w-3 h-3 rounded-full bg-purple-400 opacity-40 animate-float" style={{ animationDelay: "1s" }} />
      <div className="absolute bottom-1/3 left-1/4 w-2 h-2 rounded-full bg-blue-400 opacity-50 animate-float" style={{ animationDelay: "2s" }} />

      <div className="relative z-10 max-w-5xl mx-auto text-center space-y-8">
        {/* Badge */}
        <div className="flex justify-center">
          <span className="badge badge-cyan animate-slide-up">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            Proyecto de Título — Universidad de las Américas · 2026
          </span>
        </div>

        {/* Main heading */}
        <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-black leading-tight tracking-tight animate-slide-up" style={{ animationDelay: "0.1s" }}>
          Descubrimiento de Patrones en Perfiles de{" "}
          <span className="gradient-text-cyan">Egreso de Informática</span>{" "}
          mediante Aprendizaje no Supervisado
        </h1>

        {/* Evolution note */}
        <p className="text-sm text-slate-500 italic animate-slide-up" style={{ animationDelay: "0.15s" }}>
          Evolucionando hacia un modelo híbrido de validación estadística y clasificación supervisada.
        </p>

        {/* Subtitle */}
        <p className="text-lg md:text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed animate-slide-up" style={{ animationDelay: "0.2s" }}>
          Un pipeline automatizado en Python que aplica{" "}
          <span className="text-cyan-400 font-semibold">NLP con spaCy</span>,{" "}
          <span className="text-purple-400 font-semibold">SMOTE y SVM (Kernel RBF)</span>, y reducción de
          dimensionalidad para cuantificar la{" "}
          <span className="text-violet-400 font-semibold">convergencia semántica</span>{" "}
          entre los grados de Informática en la educación superior chilena.
        </p>

        {/* Authors */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-slide-up" style={{ animationDelay: "0.3s" }}>
          <div className="flex -space-x-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 border-2 border-gray-950 flex items-center justify-center text-xs font-bold text-white">BP</div>
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-400 to-pink-500 border-2 border-gray-950 flex items-center justify-center text-xs font-bold text-white">WR</div>
          </div>
          <div className="text-slate-400 text-sm">
            <span className="text-slate-200 font-semibold">Brayan Pineda Poblete</span>
            {" · "}
            <span className="text-slate-200 font-semibold">Walter Reyes Silva</span>
          </div>
          <span className="badge badge-purple">Facultad de Ing. y Negocios</span>
        </div>

        {/* CTA buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-slide-up" style={{ animationDelay: "0.4s" }}>
          <a href="#resultados" className="btn-primary inline-flex items-center gap-2">
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Ver Resultados del Análisis
            </span>
          </a>
          <a
            href="#metodologia"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl border border-slate-700 text-slate-300 font-semibold hover:border-cyan-500/50 hover:text-cyan-400 transition-all duration-300"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            Explorar Metodología
          </a>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-8 animate-slide-up" style={{ animationDelay: "0.5s" }}>
          {STATS.map((stat, i) => (
            <div key={i} className="glass-card p-5 text-center group cursor-default">
              <div className={`text-3xl font-black bg-gradient-to-r ${stat.color} bg-clip-text text-transparent`}>
                {stat.value}
                <span className="text-xl">{stat.suffix}</span>
              </div>
              <p className="text-xs text-slate-500 mt-1 leading-tight">{stat.label}</p>
            </div>
          ))}
        </div>

        {/* Tech stack */}
        <div className="flex flex-wrap justify-center gap-2 pt-2 animate-slide-up" style={{ animationDelay: "0.6s" }}>
          {TECH_TAGS.map((tag) => (
            <span
              key={tag}
              className="px-3 py-1 text-xs font-medium text-slate-500 border border-slate-800 rounded-full hover:border-cyan-500/40 hover:text-cyan-400 transition-all duration-200"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-bounce opacity-50">
        <span className="text-xs text-slate-500 uppercase tracking-widest">Scroll</span>
        <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </section>
  );
}
