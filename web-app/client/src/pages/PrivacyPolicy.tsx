import { Link } from "wouter";
import { ArrowLeft } from "lucide-react";

export default function PrivacyPolicy() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container max-w-4xl py-12">
        <Link href="/" className="inline-flex items-center gap-2 text-emerald-400 hover:text-emerald-300 mb-8">
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        <h1 className="text-4xl font-bold mb-2">Privacy Policy</h1>
        <p className="text-muted-foreground mb-8">Last Updated: January 25, 2026</p>

        <div className="prose prose-invert max-w-none space-y-8">
          
          {/* Required Global Disclaimer - exact text per compliance directive */}
          <div className="bg-blue-500/15 border-2 border-blue-500/50 rounded-lg p-6">
            <p className="text-lg text-white/90 font-medium leading-relaxed mb-0">
              ReUnity is a wellness and support tool. It is not a medical device and does not provide diagnosis, treatment, or crisis services. If you are in immediate danger, call 911 or your local emergency services.
            </p>
          </div>

          {/* Key Privacy Notice */}
          <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-emerald-400 mt-0">Your Privacy is Our Priority</h2>
            <p className="text-lg mb-0">
              <strong>ReUnity does not store, sell, or share your personal conversations or mental health data.</strong> Your sessions are confidential and are not retained after you close the app. We are committed to providing a safe, private space for your mental health journey.
            </p>
          </div>

          <section>
            <h2 className="text-2xl font-semibold text-white">1. Information We Collect</h2>
            
            <h3 className="text-xl font-medium text-gray-200">1.1 Information You Provide</h3>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Account Information:</strong> Email address and encrypted password for authentication purposes only.</li>
              <li><strong>Conversation Data:</strong> Messages you send during AI chat sessions. <em>This data is processed in real-time and is NOT permanently stored on our servers.</em></li>
              <li><strong>Journal Entries:</strong> If you choose to use the journaling feature, entries are encrypted and stored locally on your device unless you explicitly enable cloud sync.</li>
              <li><strong>Safety Plans:</strong> Safety plan data is encrypted and stored only on your device. You control when and how to export this information.</li>
            </ul>

            <h3 className="text-xl font-medium text-gray-200">1.2 Information We Do NOT Collect</h3>
            <ul className="list-disc pl-6 space-y-2">
              <li>We do NOT collect or store your conversation history after sessions end</li>
              <li>We do NOT track your location</li>
              <li>We do NOT sell any data to third parties</li>
              <li>We do NOT share your information with advertisers</li>
              <li>We do NOT use your data for marketing purposes</li>
              <li>We do NOT create behavioral profiles</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">2. How We Use Your Information</h2>
            <p>The limited information we collect is used solely for:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Authenticating your account access</li>
              <li>Providing real-time AI-powered mental health support during active sessions</li>
              <li>Improving the accuracy and helpfulness of our AI responses (using anonymized, aggregated data only)</li>
              <li>Sending critical safety notifications if you opt-in</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">3. Data Security & Confidentiality</h2>
            <div className="bg-gray-800/50 rounded-lg p-4">
              <p className="font-medium text-emerald-400">Our Confidentiality Commitment:</p>
              <ul className="list-disc pl-6 space-y-2 mt-2">
                <li>All data transmission uses TLS 1.3 encryption</li>
                <li>Passwords are hashed using bcrypt with salt</li>
                <li>Session data is processed in memory and not written to persistent storage</li>
                <li>We employ strict access controls—no employee can access your conversations</li>
                <li>Our AI processes your messages locally without human review</li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">4. Data Retention</h2>
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-4">Data Type</th>
                  <th className="text-left py-2 px-4">Retention Period</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-800">
                  <td className="py-2 px-4">Chat Conversations</td>
                  <td className="py-2 px-4 text-emerald-400">Not stored (session only)</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-2 px-4">Account Email</td>
                  <td className="py-2 px-4">Until account deletion</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-2 px-4">Journal Entries</td>
                  <td className="py-2 px-4">Device-only (your control)</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-2 px-4">Safety Plans</td>
                  <td className="py-2 px-4">Device-only (your control)</td>
                </tr>
              </tbody>
            </table>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">5. Your Rights</h2>
            <p>You have the right to:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Access:</strong> Request a copy of any data we hold about you</li>
              <li><strong>Deletion:</strong> Delete your account and all associated data at any time</li>
              <li><strong>Portability:</strong> Export your journal entries and safety plans</li>
              <li><strong>Correction:</strong> Update your account information</li>
              <li><strong>Withdraw Consent:</strong> Opt out of any optional data collection</li>
            </ul>
            <p className="mt-4">To exercise these rights, contact us at: <a href="mailto:privacy@reunity.app" className="text-emerald-400">privacy@reunity.app</a></p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">6. Third-Party Services</h2>
            <p>ReUnity uses the following third-party services with strict privacy agreements:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Authentication Provider:</strong> Secure login services (no conversation data shared)</li>
              <li><strong>AI Processing:</strong> Language model APIs process messages in real-time without storage</li>
              <li><strong>Crash Reporting:</strong> Anonymous error reports to improve app stability</li>
            </ul>
            <p className="mt-4">We do NOT use analytics that track individual user behavior.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">7. Children's Privacy (COPPA Compliance)</h2>
            <p>ReUnity is intended for users <strong>17 years of age and older</strong>. We do not knowingly collect information from children under 17. In compliance with the Children's Online Privacy Protection Act (COPPA), if we learn that we have collected personal information from a child under 13, we will delete that information immediately.</p>
            <p className="mt-4">If you believe a child has provided us with personal information, please contact us immediately at <a href="mailto:privacy@reunity.app" className="text-emerald-400">privacy@reunity.app</a>.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">8. GDPR Rights (European Users)</h2>
            <p>If you are located in the European Economic Area (EEA), you have additional rights under the General Data Protection Regulation (GDPR):</p>
            <ul className="list-disc pl-6 space-y-2 mt-4">
              <li><strong>Right to Access:</strong> Request a copy of your personal data</li>
              <li><strong>Right to Rectification:</strong> Request correction of inaccurate data</li>
              <li><strong>Right to Erasure:</strong> Request deletion of your data ("right to be forgotten")</li>
              <li><strong>Right to Restrict Processing:</strong> Request limitation of data processing</li>
              <li><strong>Right to Data Portability:</strong> Receive your data in a structured format</li>
              <li><strong>Right to Object:</strong> Object to processing based on legitimate interests</li>
              <li><strong>Right to Withdraw Consent:</strong> Withdraw consent at any time</li>
            </ul>
            <p className="mt-4"><strong>Legal Basis for Processing:</strong> We process your data based on: (a) your consent, (b) performance of our contract with you, and (c) our legitimate interests in providing and improving the Service.</p>
            <p className="mt-4"><strong>Data Controller:</strong> REOP Solutions, Christopher Ezernack, ezernackchristopher97@gmail.com</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">9. California Privacy Rights (CCPA)</h2>
            <p>If you are a California resident, you have additional rights under the California Consumer Privacy Act (CCPA):</p>
            <ul className="list-disc pl-6 space-y-2 mt-4">
              <li><strong>Right to Know:</strong> Request disclosure of personal information collected, used, and shared</li>
              <li><strong>Right to Delete:</strong> Request deletion of your personal information</li>
              <li><strong>Right to Opt-Out:</strong> Opt out of the sale of personal information</li>
              <li><strong>Right to Non-Discrimination:</strong> Not be discriminated against for exercising your rights</li>
            </ul>
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-4 mt-4">
              <p className="font-medium text-emerald-400">We Do Not Sell Your Personal Information</p>
              <p className="mt-2">ReUnity does not sell, rent, or trade your personal information to third parties for monetary or other valuable consideration.</p>
            </div>
            <p className="mt-4">To exercise your CCPA rights, contact us at <a href="mailto:privacy@reunity.app" className="text-emerald-400">privacy@reunity.app</a> or call (toll-free): 1-888-REUNITY</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">10. Apple App Store & Google Play Compliance</h2>
            <p>ReUnity complies with Apple App Store and Google Play Store privacy requirements:</p>
            <div className="bg-gray-800/50 rounded-lg p-4 mt-4">
              <h3 className="font-semibold text-white mb-2">Data Collection Disclosure</h3>
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-2">Data Type</th>
                    <th className="text-left py-2">Collected</th>
                    <th className="text-left py-2">Linked to Identity</th>
                    <th className="text-left py-2">Used for Tracking</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">Email Address</td>
                    <td className="py-2 text-emerald-400">Yes</td>
                    <td className="py-2">Yes (for login)</td>
                    <td className="py-2 text-emerald-400">No</td>
                  </tr>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">Health Data</td>
                    <td className="py-2 text-emerald-400">No*</td>
                    <td className="py-2">No</td>
                    <td className="py-2 text-emerald-400">No</td>
                  </tr>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">Usage Data</td>
                    <td className="py-2">Anonymous only</td>
                    <td className="py-2">No</td>
                    <td className="py-2 text-emerald-400">No</td>
                  </tr>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">Location</td>
                    <td className="py-2 text-emerald-400">No</td>
                    <td className="py-2">No</td>
                    <td className="py-2 text-emerald-400">No</td>
                  </tr>
                </tbody>
              </table>
              <p className="text-xs text-white/50 mt-2">*Conversations are processed in real-time but not stored</p>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">11. Changes to This Policy</h2>
            <p>We may update this Privacy Policy periodically. We will notify you of any material changes via email or in-app notification. Continued use of ReUnity after changes constitutes acceptance of the updated policy.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white">12. Contact Us</h2>
            <p>For privacy-related questions or concerns:</p>
            <div className="bg-gray-800/50 rounded-lg p-4 mt-4">
              <p><strong>Email:</strong> <a href="mailto:privacy@reunity.app" className="text-emerald-400">privacy@reunity.app</a></p>
              <p><strong>Developer:</strong> Christopher Ezernack</p>
              <p><strong>REOP Solutions</strong></p>
            </div>
          </section>

          {/* Final Assurance */}
          <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-6 mt-8">
            <p className="text-center text-lg mb-0">
              <strong>Your mental health journey is private.</strong><br />
              We built ReUnity to be a safe space, not a data collection tool.
            </p>
          </div>

        </div>

        <div className="mt-12 pt-8 border-t border-gray-800 text-center text-sm text-muted-foreground">
          <p>© 2026 REOP Solutions. All rights reserved.</p>
          <div className="flex justify-center gap-4 mt-4">
            <Link href="/terms" className="text-emerald-400 hover:underline">Terms of Service</Link>
            <Link href="/disclaimer" className="text-emerald-400 hover:underline">Disclaimer</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
