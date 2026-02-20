import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { FileText, Mail, Download, Calendar, TrendingUp, Moon, AlertTriangle, Heart, Brain, Check, Loader2 } from 'lucide-react';

interface ReportData {
  dateRange: { start: Date; end: Date };
  moodData: { date: string; score: number; notes: string }[];
  sleepData: { date: string; hours: number; quality: number }[];
  crisisEvents: { date: string; severity: string; triggers: string[] }[];
  entropyScores: { date: string; score: number }[];
  medications: { name: string; adherence: number }[];
  journalEntries: number;
  checkIns: number;
}

export function WellnessReportExport() {
  const [dateRange, setDateRange] = useState({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0],
  });
  const [recipientEmail, setRecipientEmail] = useState('');
  const [recipientName, setRecipientName] = useState('');
  const [includeOptions, setIncludeOptions] = useState({
    mood: true,
    sleep: true,
    crisis: true,
    entropy: true,
    medications: true,
    journal: true,
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [reportGenerated, setReportGenerated] = useState(false);

  // Generate mock report data
  const generateReportData = (): ReportData => {
    const start = new Date(dateRange.start);
    const end = new Date(dateRange.end);
    const days = Math.ceil((end.getTime() - start.getTime()) / (24 * 60 * 60 * 1000));

    const moodData = [];
    const sleepData = [];
    const entropyScores = [];
    const crisisEvents = [];

    for (let i = 0; i < days; i++) {
      const date = new Date(start.getTime() + i * 24 * 60 * 60 * 1000);
      const dateStr = date.toISOString().split('T')[0];

      moodData.push({
        date: dateStr,
        score: Math.floor(Math.random() * 5) + 3, // 3-7
        notes: '',
      });

      sleepData.push({
        date: dateStr,
        hours: 5 + Math.random() * 4, // 5-9 hours
        quality: Math.floor(Math.random() * 40) + 60, // 60-100%
      });

      entropyScores.push({
        date: dateStr,
        score: Math.floor(Math.random() * 40) + 30, // 30-70
      });

      // Random crisis events (10% chance per day)
      if (Math.random() < 0.1) {
        crisisEvents.push({
          date: dateStr,
          severity: ['low', 'moderate', 'high'][Math.floor(Math.random() * 3)],
          triggers: ['Work stress', 'Sleep deprivation', 'Family conflict'].slice(0, Math.floor(Math.random() * 2) + 1),
        });
      }
    }

    return {
      dateRange: { start, end },
      moodData,
      sleepData,
      crisisEvents,
      entropyScores,
      medications: [
        { name: 'Sertraline 50mg', adherence: 95 },
        { name: 'Trazodone 50mg', adherence: 88 },
      ],
      journalEntries: Math.floor(days * 0.7),
      checkIns: Math.floor(days * 0.85),
    };
  };

  const generatePDFContent = (data: ReportData): string => {
    const avgMood = data.moodData.reduce((sum, d) => sum + d.score, 0) / data.moodData.length;
    const avgSleep = data.sleepData.reduce((sum, d) => sum + d.hours, 0) / data.sleepData.length;
    const avgSleepQuality = data.sleepData.reduce((sum, d) => sum + d.quality, 0) / data.sleepData.length;
    const avgEntropy = data.entropyScores.reduce((sum, d) => sum + d.score, 0) / data.entropyScores.length;

    return `
REUNITY WELLNESS REPORT
========================

Patient Report Period: ${data.dateRange.start.toLocaleDateString()} - ${data.dateRange.end.toLocaleDateString()}
Generated: ${new Date().toLocaleDateString()}

EXECUTIVE SUMMARY
-----------------
This report provides a comprehensive overview of the patient's mental wellness metrics
tracked through the ReUnity application during the specified period.

${includeOptions.mood ? `
MOOD TRACKING
-------------
Average Mood Score: ${avgMood.toFixed(1)}/10
Total Mood Entries: ${data.moodData.length}
Trend: ${avgMood >= 6 ? 'Stable/Positive' : avgMood >= 4 ? 'Variable' : 'Needs Attention'}

Mood Distribution:
- High (7-10): ${data.moodData.filter(d => d.score >= 7).length} days
- Moderate (4-6): ${data.moodData.filter(d => d.score >= 4 && d.score < 7).length} days
- Low (1-3): ${data.moodData.filter(d => d.score < 4).length} days
` : ''}

${includeOptions.sleep ? `
SLEEP ANALYSIS
--------------
Average Sleep Duration: ${avgSleep.toFixed(1)} hours/night
Average Sleep Quality: ${avgSleepQuality.toFixed(0)}%
Sleep Goal Achievement: ${avgSleep >= 7 ? 'Meeting recommended 7+ hours' : 'Below recommended duration'}

Sleep Patterns:
- Adequate (7+ hrs): ${data.sleepData.filter(d => d.hours >= 7).length} nights
- Insufficient (<7 hrs): ${data.sleepData.filter(d => d.hours < 7).length} nights
- Poor Quality (<70%): ${data.sleepData.filter(d => d.quality < 70).length} nights
` : ''}

${includeOptions.entropy ? `
ENTROPY SCORE ANALYSIS
----------------------
Average Entropy Score: ${avgEntropy.toFixed(1)}/100
Score Interpretation: ${avgEntropy < 40 ? 'Low (Stable)' : avgEntropy < 60 ? 'Moderate' : 'Elevated (Monitor)'}

The entropy score measures psychological fragmentation and stress levels.
Lower scores indicate greater stability and integration.
` : ''}

${includeOptions.crisis ? `
CRISIS EVENT LOG
----------------
Total Crisis Events: ${data.crisisEvents.length}
Severity Breakdown:
- High: ${data.crisisEvents.filter(e => e.severity === 'high').length}
- Moderate: ${data.crisisEvents.filter(e => e.severity === 'moderate').length}
- Low: ${data.crisisEvents.filter(e => e.severity === 'low').length}

Common Triggers:
${Array.from(new Set(data.crisisEvents.flatMap(e => e.triggers))).map(t => `- ${t}`).join('\n') || '- None identified'}
` : ''}

${includeOptions.medications ? `
MEDICATION ADHERENCE
--------------------
${data.medications.map(m => `${m.name}: ${m.adherence}% adherence`).join('\n')}

Overall Adherence: ${(data.medications.reduce((sum, m) => sum + m.adherence, 0) / data.medications.length).toFixed(0)}%
` : ''}

${includeOptions.journal ? `
ENGAGEMENT METRICS
------------------
Journal Entries: ${data.journalEntries}
Check-ins Completed: ${data.checkIns}
Engagement Rate: ${((data.journalEntries + data.checkIns) / (data.moodData.length * 2) * 100).toFixed(0)}%
` : ''}

RECOMMENDATIONS
---------------
Based on the data collected during this period:

${avgMood < 5 ? '• Consider reviewing current wellness support plan for mood management\n' : ''}
${avgSleep < 7 ? '• Sleep hygiene practices may be beneficial\n' : ''}
${avgEntropy > 60 ? '• Increased monitoring recommended due to elevated entropy scores\n' : ''}
${data.crisisEvents.length > 3 ? '• Crisis prevention strategies should be reviewed\n' : ''}
${data.medications.some(m => m.adherence < 80) ? '• Medication adherence support may be needed\n' : ''}

---
This report was generated by ReUnity, an AI-powered mental wellness companion.
ReUnity is NOT a substitute for professional mental health care.
For professional guidance, please consult with a qualified healthcare provider.

Powered by Entropy Physics AI
https://entropy-physics-ai.com
    `.trim();
  };

  const handleGeneratePDF = async () => {
    setIsGenerating(true);
    
    // Simulate PDF generation
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const data = generateReportData();
    const content = generatePDFContent(data);
    
    // Create downloadable text file (in real app, would use PDF library)
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ReUnity_Wellness_Report_${dateRange.start}_to_${dateRange.end}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    setIsGenerating(false);
    setReportGenerated(true);
  };

  const handleSendEmail = async () => {
    if (!recipientEmail || !recipientName) {
      alert('Please enter recipient name and email');
      return;
    }
    
    setIsSending(true);
    
    // Simulate email sending
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    alert(`Report would be sent to ${recipientName} at ${recipientEmail}\n\nNote: Email functionality requires backend integration.`);
    
    setIsSending(false);
  };

  const toggleOption = (key: keyof typeof includeOptions) => {
    setIncludeOptions({ ...includeOptions, [key]: !includeOptions[key] });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2 bg-blue-500/20 rounded-lg">
          <FileText className="w-6 h-6 text-blue-400" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">Wellness Report Export</h2>
          <p className="text-sm text-zinc-400">Generate reports to share with healthcare providers</p>
        </div>
      </div>

      {/* Date Range Selection */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Calendar className="w-5 h-5 text-blue-400" />
            Report Period
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-zinc-300">Start Date</Label>
              <Input
                type="date"
                value={dateRange.start}
                onChange={e => setDateRange({ ...dateRange, start: e.target.value })}
                className="bg-zinc-800 border-zinc-700"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-zinc-300">End Date</Label>
              <Input
                type="date"
                value={dateRange.end}
                onChange={e => setDateRange({ ...dateRange, end: e.target.value })}
                className="bg-zinc-800 border-zinc-700"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Include Options */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-white">Include in Report</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { key: 'mood' as const, label: 'Mood Tracking', icon: Heart, color: 'text-pink-400' },
              { key: 'sleep' as const, label: 'Sleep Data', icon: Moon, color: 'text-indigo-400' },
              { key: 'crisis' as const, label: 'Crisis Events', icon: AlertTriangle, color: 'text-red-400' },
              { key: 'entropy' as const, label: 'Entropy Scores', icon: Brain, color: 'text-purple-400' },
              { key: 'medications' as const, label: 'Medications', icon: TrendingUp, color: 'text-green-400' },
              { key: 'journal' as const, label: 'Journal Stats', icon: FileText, color: 'text-blue-400' },
            ].map(({ key, label, icon: Icon, color }) => (
              <button
                key={key}
                onClick={() => toggleOption(key)}
                className={`p-3 rounded-lg border transition-all flex items-center gap-2 ${
                  includeOptions[key]
                    ? 'bg-zinc-800 border-zinc-600'
                    : 'bg-zinc-900/30 border-zinc-800 opacity-50'
                }`}
              >
                <Icon className={`w-4 h-4 ${color}`} />
                <span className="text-sm text-zinc-300">{label}</span>
                {includeOptions[key] && <Check className="w-4 h-4 text-green-400 ml-auto" />}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Download Section */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Download className="w-5 h-5 text-green-400" />
            Download Report
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-zinc-400 text-sm mb-4">
            Generate a comprehensive wellness report in PDF format for your records or to share with your healthcare provider.
          </p>
          <Button
            onClick={handleGeneratePDF}
            disabled={isGenerating}
            className="w-full bg-green-600 hover:bg-green-700"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating Report...
              </>
            ) : reportGenerated ? (
              <>
                <Check className="w-4 h-4 mr-2" />
                Download Again
              </>
            ) : (
              <>
                <Download className="w-4 h-4 mr-2" />
                Generate & Download PDF
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Email Section */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Mail className="w-5 h-5 text-blue-400" />
            Send to Healthcare Provider
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-zinc-400 text-sm">
            Send the report directly to your therapist, psychiatrist, or other healthcare provider via email.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-zinc-300">Provider Name</Label>
              <Input
                value={recipientName}
                onChange={e => setRecipientName(e.target.value)}
                placeholder="Dr. Smith"
                className="bg-zinc-800 border-zinc-700"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-zinc-300">Provider Email</Label>
              <Input
                type="email"
                value={recipientEmail}
                onChange={e => setRecipientEmail(e.target.value)}
                placeholder="doctor@clinic.com"
                className="bg-zinc-800 border-zinc-700"
              />
            </div>
          </div>
          <Button
            onClick={handleSendEmail}
            disabled={isSending || !recipientEmail || !recipientName}
            className="w-full bg-blue-600 hover:bg-blue-700"
          >
            {isSending ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <Mail className="w-4 h-4 mr-2" />
                Send Report via Email
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Privacy Notice */}
      <div className="p-4 bg-zinc-900/30 rounded-lg border border-zinc-800">
        <p className="text-xs text-zinc-500">
          <strong className="text-zinc-400">Privacy Notice:</strong> Reports contain sensitive health information. 
          Only share with trusted healthcare providers. ReUnity does not store or access your exported reports.
          All data processing occurs locally on your device.
        </p>
      </div>
    </div>
  );
}

export default WellnessReportExport;
