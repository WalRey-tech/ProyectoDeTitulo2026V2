"use client";

import React from "react";

export default function Footer() {
  return (
    <footer className="relative border-t border-white/5 py-12 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-10">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center text-sm font-black text-white">
                ML
              </div>
              <div>
                <p className="font-bold text-white text-sm">Tesis UDLA 2026</p>
                <p className="text-xs text-slate-500">Machine Learning Aplicado</p>
              </div>
            </div>
            <p className="text-sm text-slate-500 leading-relaxed">
              Investigación académica sobre el descubrimiento de patrones semánticos
              en perfiles de egreso de Informática mediante aprendizaje no supervisado.
            </p>
          </div>

          {/* Authors */}
          <div className="space-y-4">
            <h4 className="text-sm font-bold text-slate-300 uppercase tracking-widest">Autores</h4>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center text-xs font-bold text-white">
                  BP
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-200">Brayan Pineda Poblete</p>
                  <p className="text-xs text-slate-500">Ingeniería en Informática</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-400 to-pink-500 flex items-center justify-center text-xs font-bold text-white">
                  WR
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-200">Walter Reyes Silva</p>
                  <p className="text-xs text-slate-500">Ingeniería en Informática</p>
                </div>
              </div>
            </div>
          </div>

          {/* Institution */}
          <div className="space-y-4">
            <h4 className="text-sm font-bold text-slate-300 uppercase tracking-widest">Institución</h4>
            <div className="space-y-2">
              <p className="text-sm font-semibold text-slate-200">Universidad de las Américas</p>
              <p className="text-xs text-slate-500">Facultad de Ingeniería y Negocios</p>
              <p className="text-xs text-slate-500">Carrera: Ingeniería en Informática</p>
              <div className="flex flex-wrap gap-2 mt-3">
                <span className="badge badge-cyan text-xs">Python</span>
                <span className="badge badge-purple text-xs">ML/NLP</span>
                <span className="badge badge-emerald text-xs">Next.js</span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="border-t border-white/5 pt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-600">
            © 2026 Brayan Pineda & Walter Reyes · Universidad de las Américas · Todos los derechos reservados
          </p>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-slate-600">Tesis en evaluación · Junio 2026</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
