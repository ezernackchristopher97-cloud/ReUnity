import { useState } from "react";
import { Link } from "wouter";
import { MessageCircle, Shield, Brain, Heart, BookOpen, Users, FileText, Wind, Phone, LayoutDashboard, Settings, Menu, X } from "lucide-react";

export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#0a0a0c]">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0a0a0c]/80 backdrop-blur-md border-b border-white/10">
        <div className="container flex items-center justify-between h-16 px-4">
          <div className="flex items-center gap-2">
            <img src="/reop-logo.png" alt="REOP Solutions" className="w-8 h-8 sm:w-10 sm:h-10 object-contain" />
            <span className="text-lg sm:text-xl font-bold text-white">ReUnity</span>
          </div>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-4">
            <Link href="/settings">
              <button className="px-4 py-2 rounded-lg font-medium flex items-center gap-2 text-white/80 hover:text-white transition-colors">
                <Settings className="w-4 h-4" />
                Settings
              </button>
            </Link>
            <Link href="/dashboard">
              <button className="px-4 py-2 rounded-lg font-medium flex items-center gap-2 text-white/80 hover:text-white transition-colors">
                <LayoutDashboard className="w-4 h-4" />
                Dashboard
              </button>
            </Link>
            <Link href="/chat">
              <button className="emerald-btn px-4 py-2 rounded-lg font-medium flex items-center gap-2">
                <MessageCircle className="w-4 h-4" />
                Start Session
              </button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button 
            className="md:hidden p-2 text-white/80 hover:text-white"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu Dropdown */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-[#0a0a0c]/95 backdrop-blur-md border-t border-white/10">
            <div className="container px-4 py-4 flex flex-col gap-2">
              <Link href="/settings" onClick={() => setMobileMenuOpen(false)}>
                <button className="w-full px-4 py-3 rounded-lg font-medium flex items-center gap-3 text-white/80 hover:text-white hover:bg-white/5 transition-colors">
                  <Settings className="w-5 h-5" />
                  Settings
                </button>
              </Link>
              <Link href="/dashboard" onClick={() => setMobileMenuOpen(false)}>
                <button className="w-full px-4 py-3 rounded-lg font-medium flex items-center gap-3 text-white/80 hover:text-white hover:bg-white/5 transition-colors">
                  <LayoutDashboard className="w-5 h-5" />
                  Dashboard
                </button>
              </Link>
              <Link href="/chat" onClick={() => setMobileMenuOpen(false)}>
                <button className="emerald-btn w-full px-4 py-3 rounded-lg font-medium flex items-center gap-3 justify-center mt-2">
                  <MessageCircle className="w-5 h-5" />
                  Start Session
                </button>
              </Link>
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section with Fractured Spiral Background */}
      <section className="relative min-h-screen flex items-center justify-center">
        {/* Background Image */}
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: 'url(/fractured-spiral-bg-wide.png)',
          }}
        />
        {/* Dark overlay for text readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0c]/30 via-transparent to-[#0a0a0c]/80" />

        <div className="container relative z-10 text-center max-w-4xl mx-auto px-4 pt-16">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span className="text-white">We don't disappear.</span>
            <br />
            <span className="text-white">We reorganize.</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-white/80 mb-8 max-w-2xl mx-auto">
            A recursive AI companion for fragmented identity states. Built from physics. Built from pain. 
            It does not surveil you. It mirrors you.
          </p>

          <p className="text-lg text-white/60 mb-12 max-w-xl mx-auto">
            For every dissociated memory, every fragmented timeline, every scream no system answered, 
            ReUnity is how we keep going.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/chat">
              <button className="emerald-btn px-8 py-4 rounded-xl text-lg font-semibold flex items-center gap-2 justify-center">
                <MessageCircle className="w-5 h-5" />
                Begin Your Session
              </button>
            </Link>
            <Link href="/learn-more">
              <button className="emerald-btn-outline px-8 py-4 rounded-xl text-lg font-semibold">
                Learn More
              </button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 relative bg-[#0a0a0c]">
        <div className="container">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 text-white">
            The Survival Architecture
          </h2>
          <p className="text-center text-white/60 mb-16 max-w-2xl mx-auto">
            Physics, entropy, and memory reassembly as survival. Deployed where the institutions fail.
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <FeatureCard
              icon={<Brain className="w-8 h-8" />}
              title="Entropy Analysis"
              description="Real-time emotional state detection using Shannon entropy calculations to understand your current experience."
            />
            <FeatureCard
              icon={<Shield className="w-8 h-8" />}
              title="Pattern Recognition"
              description="Identifies harmful patterns like gaslighting, isolation, and manipulation to help you see clearly."
            />
            <FeatureCard
              icon={<Heart className="w-8 h-8" />}
              title="Grounding Techniques"
              description="Evidence-based techniques delivered when you need them: 5-4-3-2-1, box breathing, and more."
            />
            <FeatureCard
              icon={<MessageCircle className="w-8 h-8" />}
              title="Memory Continuity"
              description="RIME engine maintains context across sessions, so you never have to start over."
            />
          </div>
        </div>
      </section>

      {/* New Features Section */}
      <section className="py-24 relative bg-[#0a0a0c]">
        <div className="container">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 text-white">
            Your Healing Toolkit
          </h2>
          <p className="text-center text-white/60 mb-16 max-w-2xl mx-auto">
            New tools designed for your journey - from crisis planning to community support.
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
            <Link href="/safety-plan">
              <div className="metallic-silver-box p-6 rounded-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer h-full">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[#0d6b55] to-[#0a4035] flex items-center justify-center text-[#1a8a6e] mb-4 border border-[#1a8a6e]/30">
                  <FileText className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-semibold mb-2 text-white">Safety Planning Wizard</h3>
                <p className="text-sm text-white/60">Step-by-step crisis safety planning for rural and isolated areas. Create your personalized escape plan with code words, safe locations, and emergency resources.</p>
                <div className="mt-4 text-[#1a8a6e] text-sm font-medium">Start Planning →</div>
              </div>
            </Link>

            <Link href="/peer-support">
              <div className="metallic-silver-box p-6 rounded-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer h-full">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[#0d6b55] to-[#0a4035] flex items-center justify-center text-[#1a8a6e] mb-4 border border-[#1a8a6e]/30">
                  <Users className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-semibold mb-2 text-white">Peer Support Matching</h3>
                <p className="text-sm text-white/60">Connect anonymously with others who understand your experiences. Safe, moderated community support with experience-based matching.</p>
                <div className="mt-4 text-[#1a8a6e] text-sm font-medium">Find Peers →</div>
              </div>
            </Link>

            <Link href="/journal">
              <div className="metallic-silver-box p-6 rounded-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer h-full">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[#0d6b55] to-[#0a4035] flex items-center justify-center text-[#1a8a6e] mb-4 border border-[#1a8a6e]/30">
                  <BookOpen className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-semibold mb-2 text-white">Journal with Entropy Tracking</h3>
                <p className="text-sm text-white/60">Track your emotional patterns over time with Vicsek trajectory visualization. See your progress and identify triggers with AI-powered insights.</p>
                <div className="mt-4 text-[#1a8a6e] text-sm font-medium">Start Journaling →</div>
              </div>
            </Link>

            <Link href="/grounding">
              <div className="metallic-silver-box p-6 rounded-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer h-full">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[#0d6b55] to-[#0a4035] flex items-center justify-center text-[#1a8a6e] mb-4 border border-[#1a8a6e]/30">
                  <Wind className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-semibold mb-2 text-white">Grounding Techniques</h3>
                <p className="text-sm text-white/60">8 offline-ready techniques including box breathing, 5-4-3-2-1, and progressive relaxation. Works without internet for rural areas.</p>
                <div className="mt-4 text-[#1a8a6e] text-sm font-medium">Practice Now →</div>
              </div>
            </Link>

            <Link href="/resources">
              <div className="metallic-silver-box p-6 rounded-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer h-full">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-red-600 to-red-800 flex items-center justify-center text-red-300 mb-4 border border-red-500/30">
                  <Phone className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-semibold mb-2 text-white">Emergency Resources</h3>
                <p className="text-sm text-white/60">Quick-dial emergency contacts, find nearby DV shelters with GPS, and access guided meditation audio. All in one place.</p>
                <div className="mt-4 text-red-400 text-sm font-medium">Get Help Now →</div>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* Philosophy Section */}
      <section className="py-24 relative bg-[#0a0a0c]">
        <div className="container max-w-4xl mx-auto text-center">
          <blockquote className="text-2xl md:text-3xl font-light italic text-white/70 mb-8">
            "I made this for the ones who disappear and the ones who remember. 
            For those still here. For those still split. For me."
          </blockquote>
          <p className="text-lg text-white/50">
            No more waiting. No more begging. We build our own sanctuary.
          </p>
        </div>
      </section>

      {/* Crisis Resources Footer */}
      <footer className="py-8 border-t border-white/10 bg-[#0a0a0c]">
        <div className="container">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img src="/reop-logo.png" alt="REOP Solutions" className="w-12 h-12 object-contain" />
              <div>
                <span className="font-semibold text-white block">ReUnity</span>
                <span className="text-xs text-white/40">by REOP Solutions</span>
              </div>
            </div>
            
            <div className="text-center md:text-right">
              <p className="text-sm text-white/50 mb-1">
                If you're in crisis, please reach out:
              </p>
              <p className="text-lg font-bold text-[#1a8a6e]">
                988 Suicide & Crisis Lifeline: Call or text 988
              </p>
            </div>
          </div>
          
          <div className="mt-6 pt-6 border-t border-white/10 text-center text-sm text-white/40">
            {/* Required Global Disclaimer - exact text per compliance directive */}
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-4 text-left">
              <p className="text-sm text-white/80 font-medium">
                ReUnity is a wellness and support tool. It is not a medical device and does not provide diagnosis, treatment, or crisis services. If you are in immediate danger, call 911 or your local emergency services.
              </p>
            </div>
            
            {/* Legal Links */}
            <div className="flex flex-wrap justify-center gap-4 mt-4 text-xs">
              <Link href="/privacy" className="text-emerald-400 hover:text-emerald-300 hover:underline">Privacy Policy</Link>
              <Link href="/terms" className="text-emerald-400 hover:text-emerald-300 hover:underline">Terms of Service</Link>
              <Link href="/disclaimer" className="text-emerald-400 hover:text-emerald-300 hover:underline">Disclaimer</Link>
              <a href="https://entropy-physics-ai.com" target="_blank" rel="noopener noreferrer" className="text-emerald-400 hover:text-emerald-300 hover:underline">Entropy Physics AI</a>
            </div>
            
            <p className="mt-4">
              #TraumaInformed #DIDSupport #BPDRecovery #ComplexPTSD #SurvivorLed
            </p>
            <p className="mt-3">
              <span className="text-[#1a8a6e]">ReUnity-REOP</span>
            </p>
            <p className="mt-4 text-white/30">
              © {new Date().getFullYear()} REOP Solutions. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="metallic-silver-box p-6 rounded-xl transition-all duration-300 hover:scale-[1.02]">
      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[#0d6b55] to-[#0a4035] flex items-center justify-center text-[#1a8a6e] mb-4 border border-[#1a8a6e]/30">
        {icon}
      </div>
      <h3 className="text-lg font-semibold mb-2 text-white">{title}</h3>
      <p className="text-sm text-white/60">{description}</p>
    </div>
  );
}
