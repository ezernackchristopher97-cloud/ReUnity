import { Link } from "wouter";
import { ArrowLeft } from "lucide-react";

export default function TermsOfService() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container max-w-4xl py-12">
        <Link href="/" className="inline-flex items-center gap-2 text-emerald-400 hover:text-emerald-300 mb-8">
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        <h1 className="text-4xl font-bold mb-2">Terms of Service</h1>
        <p className="text-muted-foreground mb-8">Last Updated: January 25, 2026</p>

        <div className="prose prose-invert max-w-none space-y-8">
          
          {/* Required Global Disclaimer - exact text per compliance directive */}
          <div className="bg-blue-500/15 border-2 border-blue-500/50 rounded-lg p-6">
            <p className="text-lg text-white/90 font-medium leading-relaxed mb-0">
              ReUnity is a wellness and support tool. It is not a medical device and does not provide diagnosis, treatment, or crisis services. If you are in immediate danger, call 911 or your local emergency services.
            </p>
          </div>

          {/* Important Notice */}
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-amber-400 mt-0">Important Notice</h2>
            <p className="mb-0">
              By using ReUnity, you acknowledge that this application is <strong>NOT a substitute for professional mental health care, therapy, or medical treatment</strong>. If you are experiencing a mental health emergency, please contact emergency services (911) or a crisis hotline immediately.
            </p>
          </div>

          <section>
            <h2 className="text-2xl font-semibold text-white">1. Acceptance of Terms</h2>
            <p>By accessing or using ReUnity ("the App," "the Service"), you agree to be bound by these Terms of Service ("Terms"). If you do not agree to these Terms, you may not use the Service.</p>
            <p>These Terms constitute a legally binding agreement between you and REOP Solutions ("we," "us," "our") regarding your use of ReUnity.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">2. Description of Service</h2>
            <p>ReUnity is an AI-powered mental health support application that provides:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Conversational AI support for emotional well-being</li>
              <li>Grounding techniques and coping strategies</li>
              <li>Crisis resource information and hotline connections</li>
              <li>Journaling tools with emotional pattern tracking</li>
              <li>Safety planning features</li>
              <li>Peer support matching (optional)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">3. Eligibility</h2>
            <p>To use ReUnity, you must:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Be at least 17 years of age</li>
              <li>Have the legal capacity to enter into a binding agreement</li>
              <li>Not be prohibited from using the Service under applicable laws</li>
            </ul>
            <p className="mt-4">If you are between 17 and 18 years old, you represent that you have your parent or guardian's permission to use the Service.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">4. Medical & Mental Health Disclaimer</h2>
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-red-400 mt-0">CRITICAL DISCLAIMER</h3>
              <ul className="list-disc pl-6 space-y-2">
                <li><strong>ReUnity is NOT a medical device, therapy service, or healthcare provider.</strong></li>
                <li><strong>The AI is NOT a licensed therapist, counselor, psychologist, or psychiatrist.</strong></li>
                <li><strong>Nothing in this App constitutes medical advice, diagnosis, or treatment.</strong></li>
                <li><strong>Do NOT use ReUnity as a replacement for professional mental health care.</strong></li>
                <li><strong>If you are in crisis, call 988 (Suicide & Crisis Lifeline), 911, or go to your nearest emergency room.</strong></li>
              </ul>
            </div>
            <p className="mt-4">The information and support provided by ReUnity are for general wellness and educational purposes only. Always seek the advice of qualified health providers with any questions you may have regarding a medical or mental health condition.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">5. User Responsibilities</h2>
            <p>By using ReUnity, you agree to:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Use the Service only for lawful purposes</li>
              <li>Not attempt to harm, harass, or abuse other users (in peer support features)</li>
              <li>Not use the Service to promote violence, self-harm, or illegal activities</li>
              <li>Not attempt to reverse engineer, hack, or compromise the Service</li>
              <li>Not impersonate others or provide false information</li>
              <li>Seek professional help for serious mental health concerns</li>
              <li>Contact emergency services if you or someone else is in immediate danger</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">6. Limitation of Liability</h2>
            <div className="bg-gray-800/50 rounded-lg p-6">
              <p className="font-medium">TO THE MAXIMUM EXTENT PERMITTED BY LAW:</p>
              <ul className="list-disc pl-6 space-y-2 mt-2">
                <li>REOP Solutions and its affiliates shall NOT be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from your use of ReUnity.</li>
                <li>We are NOT responsible for any decisions you make based on information provided by the App.</li>
                <li>We are NOT liable for any harm resulting from reliance on the AI's responses.</li>
                <li>We are NOT responsible for the actions of other users in peer support features.</li>
                <li>Our total liability shall not exceed the amount you paid for the Service (if any) in the 12 months preceding the claim.</li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">7. Disclaimer of Warranties</h2>
            <p>THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Warranties of merchantability or fitness for a particular purpose</li>
              <li>Warranties that the Service will be uninterrupted, error-free, or secure</li>
              <li>Warranties regarding the accuracy, reliability, or completeness of any information provided</li>
              <li>Warranties that the AI responses will be appropriate for your specific situation</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">8. Indemnification</h2>
            <p>You agree to indemnify, defend, and hold harmless REOP Solutions, its officers, directors, employees, and agents from any claims, damages, losses, liabilities, costs, or expenses (including reasonable attorneys' fees) arising from:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Your use of the Service</li>
              <li>Your violation of these Terms</li>
              <li>Your violation of any third-party rights</li>
              <li>Any content you submit through the Service</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">9. Intellectual Property</h2>
            <p>All content, features, and functionality of ReUnity—including but not limited to text, graphics, logos, icons, images, audio, video, software, and the AI model—are owned by REOP Solutions and are protected by copyright, trademark, and other intellectual property laws.</p>
            <p className="mt-4">You may not copy, modify, distribute, sell, or lease any part of the Service without our prior written consent.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">10. Privacy</h2>
            <p>Your use of ReUnity is also governed by our <Link href="/privacy" className="text-emerald-400 hover:underline">Privacy Policy</Link>, which describes how we collect, use, and protect your information. By using the Service, you consent to our privacy practices.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">11. Termination</h2>
            <p>We reserve the right to suspend or terminate your access to ReUnity at any time, with or without cause, and with or without notice. You may also delete your account at any time through the app settings.</p>
            <p className="mt-4">Upon termination, your right to use the Service will immediately cease. Sections 4, 6, 7, 8, and 12 shall survive termination.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">12. Governing Law & Dispute Resolution</h2>
            <p>These Terms shall be governed by and construed in accordance with the laws of the State of Louisiana, United States, without regard to its conflict of law provisions.</p>
            <p className="mt-4">Any disputes arising from these Terms or your use of the Service shall be resolved through binding arbitration in accordance with the American Arbitration Association rules, except that you may assert claims in small claims court if your claims qualify.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">13. Changes to Terms</h2>
            <p>We may modify these Terms at any time. We will notify you of material changes via email or in-app notification at least 30 days before the changes take effect. Your continued use of the Service after changes become effective constitutes acceptance of the modified Terms.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">14. Severability</h2>
            <p>If any provision of these Terms is found to be unenforceable or invalid, that provision shall be limited or eliminated to the minimum extent necessary, and the remaining provisions shall remain in full force and effect.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">15. Contact Information</h2>
            <p>For questions about these Terms of Service:</p>
            <div className="bg-gray-800/50 rounded-lg p-4 mt-4">
              <p><strong>Email:</strong> <a href="mailto:legal@reunity.app" className="text-emerald-400">legal@reunity.app</a></p>
              <p><strong>Developer:</strong> Christopher Ezernack</p>
              <p><strong>REOP Solutions</strong></p>
            </div>
          </section>

          {/* Acknowledgment */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 mt-8">
            <p className="text-center mb-0">
              By using ReUnity, you acknowledge that you have read, understood, and agree to be bound by these Terms of Service and our Privacy Policy.
            </p>
          </div>

        </div>

        <div className="mt-12 pt-8 border-t border-gray-800 text-center text-sm text-muted-foreground">
          <p>© 2026 REOP Solutions. All rights reserved.</p>
          <div className="flex justify-center gap-4 mt-4">
            <Link href="/privacy" className="text-emerald-400 hover:underline">Privacy Policy</Link>
            <Link href="/disclaimer" className="text-emerald-400 hover:underline">Disclaimer</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
