import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import ChallengeSection from "./components/ChallengeSection";
import MethodologySection from "./components/MethodologySection";
import ResultsSection from "./components/ResultsSection";
import FutureSection from "./components/FutureSection";
import Footer from "./components/Footer";

export default function Home() {
  return (
    <>
      {/* Fixed decorative backgrounds */}
      <div className="mesh-bg" aria-hidden="true" />
      <div className="grid-pattern" aria-hidden="true" />

      {/* Navigation */}
      <Navbar />

      {/* Main content */}
      <main className="relative z-10">
        <HeroSection />
        <ChallengeSection />
        <MethodologySection />
        <ResultsSection />
        <FutureSection />
      </main>

      {/* Footer */}
      <footer className="relative z-10">
        <Footer />
      </footer>
    </>
  );
}
