import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Descubrimiento de Patrones en Perfiles de Egreso | Tesis UDLA 2026",
    description:
    "Investigación de Machine Learning supervisado y NLP para medir separabilidad y convergencia semántica en perfiles de egreso de Ingeniería en Informática en Chile. Universidad de las Américas. Autores: Brayan Pineda Poblete y Walter Reyes Silva.",
  keywords: [
    "machine learning",
    "clustering",
    "NLP",
    "perfiles de egreso",
    "K-Means",
    "PCA",
    "spaCy",
    "tesis",
    "UDLA",
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
