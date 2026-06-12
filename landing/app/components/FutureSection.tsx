"use client";

import React from "react";

const FUTURE_FEATURES = [
  {
    icon: "🗄️",
    title: "Base de Datos PostgreSQL",
    description:
      "Toda la metadata extraída (universidades, carreras, perfiles, clusters) se migra a tablas relacionales en PostgreSQL, permitiendo consultas SQL complejas y actualizaciones incrementales.",
    detail: "Tablas: instituciones, programas, perfiles_vectores, clusters_asignados",
    color: "border-blue-500/30 bg-blue-500/5",
    badge: "Capa de Datos",
    badgeColor: "badge-cyan",
  },
  {
    icon: "🔄",
    title: "Actualización Continua",
    description:
      "El pipeline de scraping puede programarse como un job periódico. Cuando una institución actualiza su malla curricular, el sistema detecta el cambio y re-vectoriza solo el perfil modificado.",
    detail: "Trigger: comparación hash SHA-256 del texto nuevo vs. almacenado",
    color: "border-purple-500/30 bg-purple-500/5",
    badge: "Automatización",
    badgeColor: "badge-purple",
  },
  {
    icon: "🚀",
    title: "API REST con Next.js",
    description:
      "El frontend Next.js expone rutas de API que sirven los datos desde PostgreSQL. Esto permite que terceros (otras universidades, CNED, MINEDUC) consuman los resultados via HTTP.",
    detail: "GET /api/clusters · GET /api/instituciones · GET /api/perfil/:id",
    color: "border-cyan-500/30 bg-cyan-500/5",
    badge: "Integración",
    badgeColor: "badge-emerald",
  },
  {
    icon: "📈",
    title: "Escalabilidad Nacional",
    description:
      "La arquitectura actual (32 perfiles) puede escalar a cientos de instituciones. PostgreSQL + indices sobre columnas de texto y vectores garantizan respuesta < 200ms.",
    detail: "Estimado: 500+ instituciones sin degradación del rendimiento",
    color: "border-emerald-500/30 bg-emerald-500/5",
    badge: "Escalabilidad",
    badgeColor: "badge-cyan",
  },
];

const DB_SCHEMA = [
  { table: "instituciones", columns: ["id", "nombre", "tipo", "region", "url"], color: "#06b6d4" },
  { table: "programas", columns: ["id", "inst_id", "nombre", "tipo_carrera", "url_perfil"], color: "#8b5cf6" },
  { table: "perfiles_vectores", columns: ["id", "prog_id", "texto_raw", "texto_limpio", "vector_384d"], color: "#3b82f6" },
  { table: "clusters_asignados", columns: ["id", "perfil_id", "cluster_num", "cluster_nombre", "pca_x", "pca_y"], color: "#10b981" },
];

export default function FutureSection() {
  return (
    <section id="futuro" className="relative py-24 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16 space-y-4">
          <span className="badge badge-cyan">Escalabilidad y Futuro</span>
          <h2 className="text-3xl md:text-5xl font-black text-white mt-4">
            De prototipo a{" "}
            <span className="gradient-text-cyan">sistema de producción</span>
          </h2>
          <div className="section-divider mt-6" />
          <p className="text-slate-400 max-w-2xl mx-auto text-lg mt-6">
            La integración con PostgreSQL transforma el proyecto de un análisis académico
            a una plataforma escalable de inteligencia educativa para Chile.
          </p>
        </div>

        {/* Features grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-16">
          {FUTURE_FEATURES.map((feat, i) => (
            <div
              key={i}
              className={`glass-card border ${feat.color} p-7 group transition-all duration-300 hover:scale-[1.02]`}
            >
              <div className="flex items-start gap-4">
                <span className="text-3xl flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                  {feat.icon}
                </span>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-bold text-white">{feat.title}</h3>
                    <span className={`badge text-xs ${feat.badgeColor}`}>{feat.badge}</span>
                  </div>
                  <p className="text-slate-400 text-sm leading-relaxed">{feat.description}</p>
                  <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/50">
                    <p className="font-mono text-xs text-slate-500">{feat.detail}</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* DB Schema visual */}
        <div className="glass-card border border-white/5 p-8">
          <div className="text-center mb-8">
            <h3 className="text-xl font-bold text-white">Esquema PostgreSQL Propuesto</h3>
            <p className="text-slate-500 text-sm mt-1">Estructura relacional para el sistema de producción</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {DB_SCHEMA.map((table, i) => (
              <div key={i} className="rounded-xl border border-slate-800/60 overflow-hidden">
                {/* Table header */}
                <div
                  className="px-4 py-3 flex items-center gap-2"
                  style={{ background: `${table.color}20`, borderBottom: `1px solid ${table.color}30` }}
                >
                  <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" style={{ color: table.color }}>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                  <span className="font-mono text-xs font-bold" style={{ color: table.color }}>
                    {table.table}
                  </span>
                </div>
                {/* Columns */}
                <div className="bg-slate-900/40 divide-y divide-slate-800/40">
                  {table.columns.map((col, j) => (
                    <div key={j} className="px-4 py-2 flex items-center gap-2">
                      {j === 0 ? (
                        <svg className="w-3 h-3 text-yellow-500 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                      ) : (
                        <div className="w-3 h-3 rounded-sm border border-slate-700 flex-shrink-0" />
                      )}
                      <span className="font-mono text-xs text-slate-400">{col}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Relationship arrows */}
          <div className="mt-6 text-center">
            <p className="text-xs text-slate-600">
              instituciones.id → programas.inst_id → perfiles_vectores.prog_id → clusters_asignados.perfil_id
            </p>
          </div>
        </div>

        {/* Architecture diagram text */}
        <div className="mt-8 grid md:grid-cols-3 gap-4">
          {[
            { label: "Frontend", value: "Next.js 15", desc: "React Server Components + API Routes", icon: "⚡", color: "text-cyan-400" },
            { label: "Base de Datos", value: "PostgreSQL 16", desc: "Railway Managed · Backups automáticos", icon: "🗄️", color: "text-blue-400" },
            { label: "Despliegue", value: "Railway", desc: "CI/CD automático desde GitHub · Gratis", icon: "🚂", color: "text-purple-400" },
          ].map((item, i) => (
            <div key={i} className="glass-card p-5 border border-slate-800/60 text-center">
              <span className="text-2xl">{item.icon}</span>
              <p className="text-xs text-slate-500 mt-2 uppercase tracking-widest">{item.label}</p>
              <p className={`text-lg font-black mt-1 ${item.color}`}>{item.value}</p>
              <p className="text-xs text-slate-500 mt-1">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
