import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Pill, AlertTriangle, CheckCircle, Info, Plus, X, Search, Brain, Heart, Moon } from "lucide-react";

interface Medication {
  id: string;
  name: string;
  category: string;
  dosage?: string;
}

interface Interaction {
  medications: [string, string];
  severity: "mild" | "moderate" | "severe" | "contraindicated";
  description: string;
  moodEffect: string;
  recommendation: string;
}

// Comprehensive medication database with mental health focus
const MEDICATION_DATABASE: Record<string, { category: string; aliases: string[]; moodEffects: string[] }> = {
  // SSRIs
  "sertraline": { category: "SSRI", aliases: ["zoloft"], moodEffects: ["May take 2-4 weeks for full effect", "Can cause initial anxiety increase"] },
  "fluoxetine": { category: "SSRI", aliases: ["prozac"], moodEffects: ["Activating - may help with fatigue", "Can affect sleep initially"] },
  "escitalopram": { category: "SSRI", aliases: ["lexapro"], moodEffects: ["Generally well-tolerated", "May reduce anxiety quickly"] },
  "paroxetine": { category: "SSRI", aliases: ["paxil"], moodEffects: ["Sedating effect", "Withdrawal can be difficult"] },
  "citalopram": { category: "SSRI", aliases: ["celexa"], moodEffects: ["Neutral energy effect", "Good for anxiety"] },
  
  // SNRIs
  "venlafaxine": { category: "SNRI", aliases: ["effexor"], moodEffects: ["Energizing at higher doses", "Can increase blood pressure"] },
  "duloxetine": { category: "SNRI", aliases: ["cymbalta"], moodEffects: ["Helps with pain and mood", "Can cause nausea initially"] },
  "desvenlafaxine": { category: "SNRI", aliases: ["pristiq"], moodEffects: ["Fewer drug interactions", "Steady energy boost"] },
  
  // Mood Stabilizers
  "lithium": { category: "Mood Stabilizer", aliases: ["lithobid", "eskalith"], moodEffects: ["Gold standard for bipolar", "Requires blood monitoring"] },
  "lamotrigine": { category: "Mood Stabilizer", aliases: ["lamictal"], moodEffects: ["Prevents depressive episodes", "Slow titration required"] },
  "valproate": { category: "Mood Stabilizer", aliases: ["depakote", "valproic acid"], moodEffects: ["Rapid mood stabilization", "Weight gain common"] },
  "carbamazepine": { category: "Mood Stabilizer", aliases: ["tegretol"], moodEffects: ["Effective for mania", "Many drug interactions"] },
  
  // Antipsychotics
  "quetiapine": { category: "Atypical Antipsychotic", aliases: ["seroquel"], moodEffects: ["Sedating - helps sleep", "Used for bipolar depression"] },
  "aripiprazole": { category: "Atypical Antipsychotic", aliases: ["abilify"], moodEffects: ["Activating effect", "Can cause restlessness"] },
  "olanzapine": { category: "Atypical Antipsychotic", aliases: ["zyprexa"], moodEffects: ["Strong sedation", "Significant weight gain risk"] },
  "risperidone": { category: "Atypical Antipsychotic", aliases: ["risperdal"], moodEffects: ["Moderate sedation", "Can affect prolactin"] },
  "lurasidone": { category: "Atypical Antipsychotic", aliases: ["latuda"], moodEffects: ["Must take with food", "Lower metabolic effects"] },
  
  // Benzodiazepines
  "alprazolam": { category: "Benzodiazepine", aliases: ["xanax"], moodEffects: ["Fast anxiety relief", "High dependence risk"] },
  "lorazepam": { category: "Benzodiazepine", aliases: ["ativan"], moodEffects: ["Moderate duration", "Less euphoria than alprazolam"] },
  "clonazepam": { category: "Benzodiazepine", aliases: ["klonopin"], moodEffects: ["Long-acting", "Good for panic disorder"] },
  "diazepam": { category: "Benzodiazepine", aliases: ["valium"], moodEffects: ["Muscle relaxant properties", "Very long half-life"] },
  
  // Sleep Medications
  "trazodone": { category: "Sleep Aid/Antidepressant", aliases: ["desyrel"], moodEffects: ["Low-dose for sleep", "Higher doses for depression"] },
  "mirtazapine": { category: "Antidepressant", aliases: ["remeron"], moodEffects: ["Strong sedation", "Increases appetite"] },
  "zolpidem": { category: "Sleep Aid", aliases: ["ambien"], moodEffects: ["Short-term use only", "Can cause memory issues"] },
  
  // ADHD Medications
  "methylphenidate": { category: "Stimulant", aliases: ["ritalin", "concerta"], moodEffects: ["Can worsen anxiety", "May affect appetite"] },
  "amphetamine": { category: "Stimulant", aliases: ["adderall", "vyvanse"], moodEffects: ["Mood elevation possible", "Crash when wearing off"] },
  "atomoxetine": { category: "Non-Stimulant ADHD", aliases: ["strattera"], moodEffects: ["Gradual onset", "Can help with anxiety"] },
  
  // Other
  "bupropion": { category: "Atypical Antidepressant", aliases: ["wellbutrin"], moodEffects: ["Energizing", "Can worsen anxiety"] },
  "buspirone": { category: "Anxiolytic", aliases: ["buspar"], moodEffects: ["No sedation", "Takes 2-4 weeks to work"] },
  "gabapentin": { category: "Anticonvulsant", aliases: ["neurontin"], moodEffects: ["Calming effect", "Helps with anxiety"] },
  "pregabalin": { category: "Anticonvulsant", aliases: ["lyrica"], moodEffects: ["Fast anxiety relief", "Can cause euphoria"] },
  "hydroxyzine": { category: "Antihistamine", aliases: ["vistaril", "atarax"], moodEffects: ["Non-addictive anxiety relief", "Sedating"] },
  "propranolol": { category: "Beta Blocker", aliases: ["inderal"], moodEffects: ["Physical anxiety symptoms", "No mental sedation"] },
};

// Known drug interactions
const INTERACTIONS: Interaction[] = [
  // Serotonin Syndrome risks
  { medications: ["sertraline", "tramadol"], severity: "severe", description: "Risk of serotonin syndrome", moodEffect: "Can cause dangerous mood/behavior changes, confusion, rapid heartbeat", recommendation: "Avoid combination or use with extreme caution under close supervision" },
  { medications: ["fluoxetine", "tramadol"], severity: "severe", description: "Risk of serotonin syndrome", moodEffect: "Can cause dangerous mood/behavior changes, confusion, rapid heartbeat", recommendation: "Avoid combination or use with extreme caution under close supervision" },
  { medications: ["sertraline", "lithium"], severity: "moderate", description: "Increased serotonin effects", moodEffect: "May enhance mood stabilization but increases serotonin syndrome risk", recommendation: "Monitor for tremor, confusion, agitation" },
  
  // MAOI interactions (if added)
  { medications: ["fluoxetine", "bupropion"], severity: "moderate", description: "Increased seizure risk", moodEffect: "Both can lower seizure threshold", recommendation: "Use lower doses, monitor closely" },
  
  // Benzodiazepine combinations
  { medications: ["alprazolam", "quetiapine"], severity: "moderate", description: "Enhanced sedation", moodEffect: "Excessive drowsiness, impaired cognition", recommendation: "Start with lower doses, avoid driving" },
  { medications: ["lorazepam", "olanzapine"], severity: "severe", description: "Respiratory depression risk", moodEffect: "Extreme sedation, breathing difficulties", recommendation: "Avoid IM combination, monitor vital signs" },
  
  // Lithium interactions
  { medications: ["lithium", "ibuprofen"], severity: "moderate", description: "Increased lithium levels", moodEffect: "Lithium toxicity can cause confusion, tremor, nausea", recommendation: "Use acetaminophen instead, monitor lithium levels" },
  { medications: ["lithium", "naproxen"], severity: "moderate", description: "Increased lithium levels", moodEffect: "Lithium toxicity can cause confusion, tremor, nausea", recommendation: "Use acetaminophen instead, monitor lithium levels" },
  
  // Stimulant interactions
  { medications: ["methylphenidate", "sertraline"], severity: "mild", description: "May increase stimulant effects", moodEffect: "Increased anxiety, jitteriness possible", recommendation: "Monitor for increased anxiety or agitation" },
  { medications: ["amphetamine", "bupropion"], severity: "moderate", description: "Increased seizure and cardiovascular risk", moodEffect: "Overstimulation, anxiety, insomnia", recommendation: "Use lower doses, monitor blood pressure" },
  
  // Antipsychotic combinations
  { medications: ["quetiapine", "carbamazepine"], severity: "moderate", description: "Reduced quetiapine levels", moodEffect: "May reduce antipsychotic effectiveness", recommendation: "May need higher quetiapine dose" },
  { medications: ["aripiprazole", "fluoxetine"], severity: "mild", description: "Increased aripiprazole levels", moodEffect: "May increase restlessness or sedation", recommendation: "May need lower aripiprazole dose" },
  
  // Sleep medication interactions
  { medications: ["zolpidem", "alprazolam"], severity: "severe", description: "Dangerous CNS depression", moodEffect: "Extreme sedation, memory blackouts, falls risk", recommendation: "Avoid combination" },
  { medications: ["trazodone", "alprazolam"], severity: "moderate", description: "Enhanced sedation", moodEffect: "Excessive drowsiness, next-day impairment", recommendation: "Use lower doses of both" },
  
  // Mood stabilizer interactions
  { medications: ["lamotrigine", "valproate"], severity: "moderate", description: "Increased lamotrigine levels", moodEffect: "Risk of serious rash, mood changes", recommendation: "Reduce lamotrigine dose by 50%" },
  { medications: ["carbamazepine", "lamotrigine"], severity: "moderate", description: "Reduced lamotrigine levels", moodEffect: "May reduce mood stabilization", recommendation: "May need higher lamotrigine dose" },
  
  // Beta blocker interactions
  { medications: ["propranolol", "venlafaxine"], severity: "mild", description: "May increase propranolol effects", moodEffect: "Increased fatigue, low blood pressure", recommendation: "Monitor blood pressure and heart rate" },
];

export function MedicationInteractionChecker({ compact = false }: { compact?: boolean }) {
  const [medications, setMedications] = useState<Medication[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [showSearch, setShowSearch] = useState(false);

  const searchMedications = (term: string) => {
    setSearchTerm(term);
    if (term.length < 2) {
      setSearchResults([]);
      return;
    }
    
    const lowerTerm = term.toLowerCase();
    const results: string[] = [];
    
    for (const [name, data] of Object.entries(MEDICATION_DATABASE)) {
      if (name.includes(lowerTerm) || data.aliases.some(a => a.includes(lowerTerm))) {
        results.push(name);
      }
    }
    
    setSearchResults(results.slice(0, 8));
  };

  const addMedication = (name: string) => {
    const medData = MEDICATION_DATABASE[name];
    if (!medData || medications.some(m => m.name === name)) return;
    
    const newMed: Medication = {
      id: Date.now().toString(),
      name,
      category: medData.category,
    };
    
    const newMedications = [...medications, newMed];
    setMedications(newMedications);
    setSearchTerm("");
    setSearchResults([]);
    setShowSearch(false);
    
    // Check for interactions
    checkInteractions(newMedications);
  };

  const removeMedication = (id: string) => {
    const newMedications = medications.filter(m => m.id !== id);
    setMedications(newMedications);
    checkInteractions(newMedications);
  };

  const checkInteractions = (meds: Medication[]) => {
    const foundInteractions: Interaction[] = [];
    
    for (let i = 0; i < meds.length; i++) {
      for (let j = i + 1; j < meds.length; j++) {
        const med1 = meds[i].name;
        const med2 = meds[j].name;
        
        for (const interaction of INTERACTIONS) {
          if (
            (interaction.medications[0] === med1 && interaction.medications[1] === med2) ||
            (interaction.medications[0] === med2 && interaction.medications[1] === med1)
          ) {
            foundInteractions.push(interaction);
          }
        }
      }
    }
    
    setInteractions(foundInteractions);
  };

  const getSeverityColor = (severity: Interaction["severity"]) => {
    switch (severity) {
      case "mild": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "moderate": return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      case "severe": return "bg-red-500/20 text-red-400 border-red-500/30";
      case "contraindicated": return "bg-red-600/20 text-red-300 border-red-600/30";
    }
  };

  const getSeverityIcon = (severity: Interaction["severity"]) => {
    switch (severity) {
      case "mild": return <Info className="h-4 w-4" />;
      case "moderate": return <AlertTriangle className="h-4 w-4" />;
      case "severe": return <AlertTriangle className="h-4 w-4" />;
      case "contraindicated": return <X className="h-4 w-4" />;
    }
  };

  if (compact) {
    return (
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Pill className="h-4 w-4 text-purple-400" />
            Medication Checker
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-xs text-slate-400">
              {medications.length} medications tracked
            </div>
            {interactions.length > 0 && (
              <Badge variant="outline" className="text-xs bg-orange-500/20 text-orange-400 border-orange-500/30">
                {interactions.length} interaction{interactions.length !== 1 ? "s" : ""}
              </Badge>
            )}
          </div>
          <Button 
            variant="outline" 
            size="sm" 
            className="w-full mt-2 text-xs"
            onClick={() => setShowSearch(true)}
          >
            <Plus className="h-3 w-3 mr-1" />
            Add Medication
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Pill className="h-5 w-5 text-purple-400" />
            Medication Interaction Checker
          </CardTitle>
          <CardDescription>
            Track your medications and check for interactions that may affect your mood and mental health
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search */}
          <div className="relative">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <Input
                  placeholder="Search medications..."
                  value={searchTerm}
                  onChange={(e) => searchMedications(e.target.value)}
                  className="pl-10 bg-slate-900/50 border-slate-600"
                />
              </div>
            </div>
            
            {searchResults.length > 0 && (
              <div className="absolute z-10 w-full mt-1 bg-slate-800 border border-slate-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                {searchResults.map((name) => {
                  const data = MEDICATION_DATABASE[name];
                  return (
                    <button
                      key={name}
                      onClick={() => addMedication(name)}
                      className="w-full px-4 py-2 text-left hover:bg-slate-700 flex items-center justify-between"
                    >
                      <div>
                        <div className="font-medium capitalize">{name}</div>
                        <div className="text-xs text-slate-400">{data.category}</div>
                      </div>
                      <Plus className="h-4 w-4 text-emerald-400" />
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Current Medications */}
          {medications.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-slate-300">Your Medications</h4>
              <div className="flex flex-wrap gap-2">
                {medications.map((med) => (
                  <Badge
                    key={med.id}
                    variant="outline"
                    className="bg-slate-700/50 border-slate-600 py-1 px-3 flex items-center gap-2"
                  >
                    <span className="capitalize">{med.name}</span>
                    <span className="text-xs text-slate-400">({med.category})</span>
                    <button
                      onClick={() => removeMedication(med.id)}
                      className="ml-1 hover:text-red-400"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Mood Effects */}
          {medications.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-slate-300 flex items-center gap-2">
                <Brain className="h-4 w-4 text-purple-400" />
                Mood & Mental Health Effects
              </h4>
              <div className="space-y-2">
                {medications.map((med) => {
                  const data = MEDICATION_DATABASE[med.name];
                  return (
                    <div key={med.id} className="bg-slate-900/50 rounded-lg p-3">
                      <div className="font-medium capitalize text-sm">{med.name}</div>
                      <ul className="mt-1 space-y-1">
                        {data.moodEffects.map((effect, i) => (
                          <li key={i} className="text-xs text-slate-400 flex items-start gap-2">
                            <span className="text-purple-400 mt-0.5">•</span>
                            {effect}
                          </li>
                        ))}
                      </ul>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Interactions */}
          {interactions.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-red-400 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Potential Interactions Detected
              </h4>
              <div className="space-y-3">
                {interactions.map((interaction, i) => (
                  <div
                    key={i}
                    className={`rounded-lg p-4 border ${getSeverityColor(interaction.severity)}`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      {getSeverityIcon(interaction.severity)}
                      <span className="font-medium capitalize">
                        {interaction.medications[0]} + {interaction.medications[1]}
                      </span>
                      <Badge variant="outline" className={getSeverityColor(interaction.severity)}>
                        {interaction.severity}
                      </Badge>
                    </div>
                    <p className="text-sm mb-2">{interaction.description}</p>
                    <div className="space-y-2 text-xs">
                      <div className="flex items-start gap-2">
                        <Brain className="h-3 w-3 mt-0.5 text-purple-400" />
                        <span><strong>Mood Effect:</strong> {interaction.moodEffect}</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <CheckCircle className="h-3 w-3 mt-0.5 text-emerald-400" />
                        <span><strong>Recommendation:</strong> {interaction.recommendation}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {medications.length === 0 && (
            <div className="text-center py-8 text-slate-400">
              <Pill className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>Add your medications to check for interactions</p>
              <p className="text-xs mt-1">We'll show you how they may affect your mood and mental health</p>
            </div>
          )}

          {medications.length > 0 && interactions.length === 0 && (
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-4 flex items-center gap-3">
              <CheckCircle className="h-5 w-5 text-emerald-400" />
              <div>
                <div className="font-medium text-emerald-400">No Interactions Detected</div>
                <div className="text-xs text-slate-400">Your current medications appear safe to take together</div>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="bg-slate-900/50 rounded-lg p-3 text-xs text-slate-400">
            <strong className="text-slate-300">Important:</strong> This tool provides general information only and is not a substitute for professional medical advice. Always consult your doctor or pharmacist about medication interactions.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
