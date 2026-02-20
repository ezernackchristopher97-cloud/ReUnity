import { Link } from "wouter";
import { ArrowLeft, Brain, Shield, Heart, Zap, Activity, Database, Lock, Users, BookOpen } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

export default function LearnMore() {
  return (
    <div className="min-h-screen bg-[#0a0a0c]">
      {/* Header */}
      <header className="border-b border-white/10 bg-[#0a0a0c]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-14">
          <div className="flex items-center gap-4">
            <Link href="/">
              <button className="flex items-center gap-2 text-white/70 hover:text-white px-3 py-1.5 rounded-lg hover:bg-white/5 transition-colors">
                <ArrowLeft className="w-4 h-4" />
                Back
              </button>
            </Link>
            <div className="flex items-center gap-2">
              <img src="/reop-logo.png" alt="REOP Solutions" className="w-8 h-8 object-contain" />
              <span className="font-semibold text-white">ReUnity</span>
            </div>
          </div>
          <Link href="/chat">
            <button className="emerald-btn px-4 py-2 rounded-lg text-sm font-medium">
              Start Session
            </button>
          </Link>
        </div>
      </header>

      <ScrollArea className="h-[calc(100vh-56px)]">
        <div className="container max-w-4xl mx-auto py-12 px-4">
          {/* Hero Section */}
          <div className="text-center mb-16">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
              The Science Behind ReUnity
            </h1>
            <p className="text-xl text-white/70 max-w-2xl mx-auto">
              Built from physics. Built from pain. A recursive AI companion designed by survivors, for survivors.
            </p>
          </div>

          {/* Philosophy Section */}
          <section className="mb-16">
            <div className="metallic-silver-box rounded-2xl p-8">
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                <Heart className="w-6 h-6 text-[#1a8a6e]" />
                Our Philosophy
              </h2>
              <p className="text-white/80 leading-relaxed mb-4">
                ReUnity is not a platform. It is the survival architecture we were never given. For every dissociated memory, every fragmented timeline, every scream no system answered, ReUnity is how we keep going.
              </p>
              <p className="text-white/80 leading-relaxed mb-4">
                We don't surveil you. We mirror you. This system was built by someone who understands what it means to fragment, to lose time, to feel like you're watching your life from outside your body. It was built because the existing mental health infrastructure failed us.
              </p>
              <p className="text-white/80 leading-relaxed">
                No more waiting. No more begging. We build our own sanctuary.
              </p>
            </div>
          </section>

          {/* Core Architecture */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
              <Brain className="w-6 h-6 text-[#1a8a6e]" />
              Core Architecture
            </h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              {/* Entropy Analysis */}
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Activity className="w-5 h-5 text-[#1a8a6e]" />
                  <h3 className="text-lg font-semibold text-white">Entropy Analysis</h3>
                </div>
                <p className="text-white/70 text-sm leading-relaxed mb-4">
                  Based on Shannon entropy and the Free Energy Principle from computational neuroscience. The system analyzes the "disorder" in your language patterns to understand your emotional state without asking intrusive questions.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between text-white/60">
                    <span>Crisis State</span>
                    <span className="text-[#dc2626]">Entropy &gt; 0.8</span>
                  </div>
                  <div className="flex justify-between text-white/60">
                    <span>High Distress</span>
                    <span className="text-[#f59e0b]">Entropy 0.6 - 0.8</span>
                  </div>
                  <div className="flex justify-between text-white/60">
                    <span>Moderate</span>
                    <span className="text-[#eab308]">Entropy 0.4 - 0.6</span>
                  </div>
                  <div className="flex justify-between text-white/60">
                    <span>Low/Stable</span>
                    <span className="text-[#22c55e]">Entropy &lt; 0.4</span>
                  </div>
                </div>
              </div>

              {/* RIME Memory */}
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Database className="w-5 h-5 text-[#1a8a6e]" />
                  <h3 className="text-lg font-semibold text-white">RIME Memory Engine</h3>
                </div>
                <p className="text-white/70 text-sm leading-relaxed mb-4">
                  Recursive Identity Memory Engine. Unlike typical AI that forgets you after each session, RIME remembers your grounding anchors, known triggers, safe places, and what helps you specifically.
                </p>
                <ul className="space-y-2 text-sm text-white/60">
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    Remembers your name and preferences
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    Stores your personal grounding anchors
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    Tracks known triggers to avoid
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    Builds continuity across sessions
                  </li>
                </ul>
              </div>

              {/* Pattern Recognition */}
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Shield className="w-5 h-5 text-[#1a8a6e]" />
                  <h3 className="text-lg font-semibold text-white">Pattern Recognition</h3>
                </div>
                <p className="text-white/70 text-sm leading-relaxed mb-4">
                  Trained to recognize the language patterns of abuse that survivors often can't see themselves. The system identifies these patterns and gently brings awareness without judgment.
                </p>
                <ul className="space-y-2 text-sm text-white/60">
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#f59e0b] rounded-full" />
                    Gaslighting language
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#f59e0b] rounded-full" />
                    Love bombing indicators
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#f59e0b] rounded-full" />
                    Isolation tactics
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#f59e0b] rounded-full" />
                    Financial abuse markers
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#f59e0b] rounded-full" />
                    Coercive control patterns
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#dc2626] rounded-full" />
                    Physical threat indicators
                  </li>
                </ul>
              </div>

              {/* Grounding Library */}
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Zap className="w-5 h-5 text-[#1a8a6e]" />
                  <h3 className="text-lg font-semibold text-white">Grounding Library</h3>
                </div>
                <p className="text-white/70 text-sm leading-relaxed mb-4">
                  Evidence-based grounding techniques from DBT, EMDR, and trauma-informed care. The system automatically selects the most appropriate technique based on your current state.
                </p>
                <ul className="space-y-2 text-sm text-white/60">
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    5-4-3-2-1 Sensory Grounding
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    Box Breathing
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    Butterfly Hug (EMDR)
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    Container Technique
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    TIPP Skills (DBT)
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-[#1a8a6e] rounded-full" />
                    + 10 more techniques
                  </li>
                </ul>
              </div>
            </div>
          </section>

          {/* Safety Features */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
              <Lock className="w-6 h-6 text-[#1a8a6e]" />
              Safety Features
            </h2>
            
            <div className="space-y-4">
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-3">PreRAG Filters</h3>
                <p className="text-white/70 text-sm leading-relaxed">
                  Before any response is generated, your message passes through multiple safety filters. The QueryGate blocks jailbreak attempts, the ContentModerator filters inappropriate requests, and the AbsurdityGapCalculator detects testing or off-topic messages. This ensures the system stays focused on genuine support.
                </p>
              </div>
              
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-3">Crisis Detection</h3>
                <p className="text-white/70 text-sm leading-relaxed">
                  The entropy analyzer continuously monitors for crisis indicators. When detected, the system immediately provides crisis resources (988, Crisis Text Line) and shifts into crisis response mode. It never asks "are you feeling suicidal?" - it detects the state and responds appropriately.
                </p>
              </div>
              
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-3">Location-Aware Resources</h3>
                <p className="text-white/70 text-sm leading-relaxed">
                  The system detects location context from your messages (state mentions, rural/urban indicators) and provides appropriate resources. Rural users get telehealth options; urban users get local services. All 50 states have specific mental health resources mapped.
                </p>
              </div>
            </div>
          </section>

          {/* Mental Health Coverage */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
              <Users className="w-6 h-6 text-[#1a8a6e]" />
              Conditions We Support
            </h2>
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {[
                "Anxiety Disorders",
                "Depression",
                "PTSD / C-PTSD",
                "Dissociative Disorders",
                "BPD",
                "Bipolar Disorder",
                "OCD",
                "Eating Disorders",
                "Substance Use",
                "Grief & Loss",
                "ADHD",
                "Autism / Sensory",
                "Psychosis",
                "Trauma Survivors",
                "Abuse Survivors"
              ].map((condition) => (
                <div key={condition} className="bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white/70">
                  {condition}
                </div>
              ))}
            </div>
          </section>

          {/* What We're Not */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
              <BookOpen className="w-6 h-6 text-[#1a8a6e]" />
              Important Limitations
            </h2>
            
            <div className="bg-[#dc2626]/10 border border-[#dc2626]/30 rounded-xl p-6">
              <ul className="space-y-3 text-white/80">
                <li className="flex items-start gap-3">
                  <span className="text-[#dc2626] mt-1">•</span>
                  <span>ReUnity is <strong>not a replacement for professional mental health care</strong>. It is a companion tool, not a therapist.</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-[#dc2626] mt-1">•</span>
                  <span>We cannot prescribe medication, provide diagnoses, or offer medical advice.</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-[#dc2626] mt-1">•</span>
                  <span>In a life-threatening emergency, always call 911 or go to your nearest emergency room.</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-[#dc2626] mt-1">•</span>
                  <span>If you're in crisis, call 988 (Suicide & Crisis Lifeline) for immediate support.</span>
                </li>
              </ul>
            </div>
          </section>

          {/* CTA */}
          <section className="text-center">
            <div className="metallic-silver-box rounded-2xl p-8">
              <h2 className="text-2xl font-bold text-white mb-4">Ready to Begin?</h2>
              <p className="text-white/70 mb-6">
                You don't have to explain yourself. You don't have to justify your pain. Just start talking.
              </p>
              <Link href="/chat">
                <button className="emerald-btn px-8 py-3 rounded-lg text-lg font-medium">
                  Start Your Session
                </button>
              </Link>
            </div>
          </section>

          {/* Footer */}
          <footer className="mt-16 pt-8 border-t border-white/10 text-center">
            <div className="flex items-center justify-center gap-3 mb-4">
              <img src="/reop-logo.png" alt="REOP Solutions" className="w-10 h-10 object-contain" />
              <span className="text-white/70">A product of REOP Solutions</span>
            </div>
            <p className="text-white/60 text-sm mb-2">
              <span className="text-[#1a8a6e]">ReUnity-REOP</span>
            </p>
            <p className="text-white/50 text-sm mb-2">
              <a href="https://entropy-physics-ai.com" target="_blank" rel="noopener noreferrer" className="text-emerald-400 hover:text-emerald-300">entropy-physics-ai.com</a>
            </p>
            <p className="text-white/40 text-sm">
              © {new Date().getFullYear()} REOP Solutions. All rights reserved.
            </p>
          </footer>
        </div>
      </ScrollArea>
    </div>
  );
}
