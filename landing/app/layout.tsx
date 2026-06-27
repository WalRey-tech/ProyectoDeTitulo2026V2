import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Descubrimiento de Patrones en Perfiles de Egreso de Informática mediante Machine Learning | UDLA 2026",
  description:
    "Proyecto de Título en Machine Learning: pipeline NLP + Random Forest (92.79% de Accuracy) con SMOTE para descubrir patrones y cuantificar la convergencia semántica (76.6%) entre perfiles de egreso de Informática en Chile. Universidad de las Américas. Autores: Brayan Pineda Poblete y Walter Reyes Silva.",
  keywords: [
    "machine learning",
    "random forest",
    "NLP",
    "perfiles de egreso",
    "SMOTE",
    "PCA",
    "LDA",
    "spaCy",
    "proyecto de título",
    "UDLA",
    "convergencia semántica",
    "similitud coseno",
    "informática",
    "descubrimiento de patrones",
  ],
  authors: [
    { name: "Brayan Pineda Poblete" },
    { name: "Walter Reyes Silva" },
  ],
  openGraph: {
    title: "Descubrimiento de Patrones en Perfiles de Egreso de Informática mediante Machine Learning",
    description:
      "Random Forest (92.79% Accuracy) + NLP para descubrir patrones y medir convergencia semántica (76.6%) en perfiles de egreso – Universidad de las Américas 2026",
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
