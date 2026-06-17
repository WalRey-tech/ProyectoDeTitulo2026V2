import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Descubrimiento de Patrones en Perfiles de Egreso de Informática | Proyecto de Título UDLA 2026",
    description:
    "Proyecto de Título en Machine Learning: pipeline NLP + SVM (Kernel RBF) con 87.58% de Accuracy para cuantificar la convergencia semántica entre perfiles de egreso de Informática en Chile. Universidad de las Américas. Autores: Brayan Pineda Poblete y Walter Reyes Silva.",
  keywords: [
    "machine learning",
    "SVM",
    "NLP",
    "perfiles de egreso",
    "SMOTE",
    "PCA",
    "LDA",
    "spaCy",
    "proyecto de título",
    "UDLA",
    "convergencia semántica",
    "informática",
  ],
  authors: [
    { name: "Brayan Pineda Poblete" },
    { name: "Walter Reyes Silva" },
  ],
  openGraph: {
    title: "Separabilidad Semántica en Perfiles de Egreso de Informática",
    description:
      "Aprendizaje Supervisado y NLP para clasificar y medir convergencia en perfiles de egreso – Universidad de las Américas 2026",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" className={inter.variable}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body className="antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
