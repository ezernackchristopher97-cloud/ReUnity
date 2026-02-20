import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { 
  Shield, 
  ChevronLeft, 
  ChevronRight, 
  Phone, 
  MapPin, 
  FileText, 
  DollarSign, 
  Smartphone, 
  Users, 
  Heart, 
  LogOut,
  AlertTriangle,
  Lock,
  CheckCircle2,
  Download,
  Loader2
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";
import BiometricLock from "@/components/BiometricLock";

// Safety plan step data
const steps = [
  {
    id: "safe_contacts",
    title: "Safe Contacts & Code Words",
    icon: Phone,
    description: "Identify people you can trust and create secret signals to communicate danger."
  },
  {
    id: "warning_signs",
    title: "Recognizing Warning Signs",
    icon: AlertTriangle,
    description: "Understanding patterns helps you anticipate danger and act before situations escalate."
  },
  {
    id: "safe_locations",
    title: "Safe Places to Go",
    icon: MapPin,
    description: "Know where you can go at any time, day or night, if you need to leave quickly."
  },
  {
    id: "emergency_bag",
    title: "Emergency Bag",
    icon: FileText,
    description: "Prepare a bag with essentials that you can grab quickly."
  },
  {
    id: "documents",
    title: "Important Documents",
    icon: FileText,
    description: "Having your documents makes starting over much easier."
  },
  {
    id: "financial_safety",
    title: "Financial Safety",
    icon: DollarSign,
    description: "Taking steps now can help you have resources when you leave."
  },
  {
    id: "technology_safety",
    title: "Technology Safety",
    icon: Smartphone,
    description: "Abusers often use technology to monitor and control."
  },
  {
    id: "children",
    title: "Children's Safety",
    icon: Users,
    description: "If you have children, their safety is part of your plan."
  },
  {
    id: "pets",
    title: "Pet Safety",
    icon: Heart,
    description: "There are options to keep your pets safe too."
  },
  {
    id: "exit_strategy",
    title: "Your Exit Strategy",
    icon: LogOut,
    description: "Your plan for when you're ready to leave."
  }
];

// Emergency bag checklist items
const emergencyBagItems = [
  "Cash (small bills)",
  "Change of clothes",
  "Toiletries",
  "Phone charger",
  "Medications",
  "Copies of important documents",
  "Keys (car, house, work)",
  "Children's items (if applicable)",
  "Pet supplies (if applicable)",
  "Comfort item",
  "Snacks and water",
  "Flashlight",
  "First aid kit"
];

// Document checklist items
const documentItems = [
  "Driver's license / ID",
  "Social Security card",
  "Birth certificate",
  "Passport",
  "Children's birth certificates",
  "Children's Social Security cards",
  "Marriage certificate",
  "Protective orders / legal documents",
  "Medical records",
  "Insurance cards",
  "Bank statements",
  "Pay stubs",
  "Lease / mortgage documents",
  "Car title / registration",
  "Photos documenting abuse",
  "Immigration documents (if applicable)"
];

export default function SafetyPlan() {
  const { user, isLoading: authLoading } = useAuth();
  const [isUnlocked, setIsUnlocked] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [planData, setPlanData] = useState<Record<string, any>>({
    safeContacts: [{ name: "", relationship: "", phone: "", codeWord: "", knowsSituation: false }],
    warningSigns: [""],
    dangerousTimes: [""],
    safeLocations: [{ name: "", address: "", distance: "", hasKey: false }],
    emergencyBagLocation: "",
    emergencyBagItems: [],
    documentsSecured: [],
    documentLocation: "",
    hasHiddenMoney: false,
    hiddenMoneyLocation: "",
    hasSecretAccount: false,
    financialSteps: [],
    phoneMonitored: false,
    locationTracked: false,
    socialMediaMonitored: false,
    techSafetySteps: [],
    hasChildren: false,
    childrenNames: [""],
    schoolInfo: "",
    childSafetySteps: [],
    hasPets: false,
    petInfo: "",
    petPlan: "",
    bestTimeToLeave: "",
    transportationPlan: "",
    firstDestination: "",
    backupPlan: "",
    whoToCall: ""
  });
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [showSavedMessage, setShowSavedMessage] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  // tRPC queries and mutations
  const { data: savedPlan, isLoading: planLoading } = trpc.safetyPlan.get.useQuery(
    undefined,
    { enabled: !!user }
  );

  const savePlanMutation = trpc.safetyPlan.save.useMutation({
    onSuccess: () => {
      toast.success("Your safety plan has been saved securely.");
      setShowSavedMessage(true);
      setTimeout(() => setShowSavedMessage(false), 3000);
    },
    onError: (error) => {
      toast.error(error.message);
    }
  });

  const exportPlanMutation = trpc.safetyPlan.export.useMutation({
    onSuccess: (data) => {
      // Create a blob and download
      const blob = new Blob([data.html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = data.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success("Safety plan exported! You can print this and keep it in a safe place.");
      setIsExporting(false);
    },
    onError: (error) => {
      toast.error(error.message);
      setIsExporting(false);
    }
  });

  // Load saved plan on mount
  useEffect(() => {
    if (savedPlan?.encryptedData) {
      try {
        const parsed = JSON.parse(savedPlan.encryptedData);
        setPlanData(parsed);
        if (savedPlan.lastStepId) {
          const stepIndex = steps.findIndex(s => s.id === savedPlan.lastStepId);
          if (stepIndex >= 0) setCurrentStep(stepIndex);
        }
      } catch (e) {
        console.error("Failed to parse saved plan", e);
      }
    }
  }, [savedPlan]);

  const progress = (completedSteps.length / steps.length) * 100;
  const currentStepData = steps[currentStep];

  const handleNext = () => {
    if (!completedSteps.includes(currentStepData.id)) {
      setCompletedSteps([...completedSteps, currentStepData.id]);
    }
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSave = () => {
    savePlanMutation.mutate({
      planData: JSON.stringify(planData),
      isComplete: completedSteps.length === steps.length,
      currentStep: currentStepData.id
    });
  };

  const handleExport = () => {
    if (completedSteps.length === 0) {
      toast.error("Please complete at least one step before exporting.");
      return;
    }
    setIsExporting(true);
    exportPlanMutation.mutate();
  };

  const updatePlanData = (key: string, value: any) => {
    setPlanData({ ...planData, [key]: value });
  };

  const addContact = () => {
    setPlanData({
      ...planData,
      safeContacts: [...planData.safeContacts, { name: "", relationship: "", phone: "", codeWord: "", knowsSituation: false }]
    });
  };

  const updateContact = (index: number, field: string, value: any) => {
    const newContacts = [...planData.safeContacts];
    newContacts[index] = { ...newContacts[index], [field]: value };
    setPlanData({ ...planData, safeContacts: newContacts });
  };

  const addLocation = () => {
    setPlanData({
      ...planData,
      safeLocations: [...planData.safeLocations, { name: "", address: "", distance: "", hasKey: false }]
    });
  };

  const updateLocation = (index: number, field: string, value: any) => {
    const newLocations = [...planData.safeLocations];
    newLocations[index] = { ...newLocations[index], [field]: value };
    setPlanData({ ...planData, safeLocations: newLocations });
  };

  const toggleChecklistItem = (listKey: string, item: string) => {
    const currentList = planData[listKey] || [];
    if (currentList.includes(item)) {
      updatePlanData(listKey, currentList.filter((i: string) => i !== item));
    } else {
      updatePlanData(listKey, [...currentList, item]);
    }
  };

  const renderStepContent = () => {
    switch (currentStepData.id) {
      case "safe_contacts":
        return (
          <div className="space-y-6">
            <Alert className="bg-teal-500/10 border-teal-500/30">
              <Shield className="h-4 w-4 text-teal-400" />
              <AlertTitle className="text-teal-300">Rural Safety Tip</AlertTitle>
              <AlertDescription className="text-teal-200/80">
                In rural areas, neighbors may be far away. Consider contacts who can reach you quickly, 
                and establish a check-in schedule if phone service is unreliable.
              </AlertDescription>
            </Alert>

            {planData.safeContacts.map((contact: any, index: number) => (
              <Card key={index} className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-lg text-white">Contact {index + 1}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-slate-300">Name</Label>
                      <Input
                        value={contact.name}
                        onChange={(e) => updateContact(index, "name", e.target.value)}
                        placeholder="Who do you trust?"
                        className="bg-slate-900/50 border-slate-600 text-white"
                      />
                    </div>
                    <div>
                      <Label className="text-slate-300">Relationship</Label>
                      <Input
                        value={contact.relationship}
                        onChange={(e) => updateContact(index, "relationship", e.target.value)}
                        placeholder="Friend, sister, etc."
                        className="bg-slate-900/50 border-slate-600 text-white"
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-slate-300">Phone Number</Label>
                      <Input
                        value={contact.phone}
                        onChange={(e) => updateContact(index, "phone", e.target.value)}
                        placeholder="Their phone number"
                        className="bg-slate-900/50 border-slate-600 text-white"
                      />
                    </div>
                    <div>
                      <Label className="text-slate-300">Code Word</Label>
                      <Input
                        value={contact.codeWord}
                        onChange={(e) => updateContact(index, "codeWord", e.target.value)}
                        placeholder="e.g., 'I need milk'"
                        className="bg-slate-900/50 border-slate-600 text-white"
                      />
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      checked={contact.knowsSituation}
                      onCheckedChange={(checked) => updateContact(index, "knowsSituation", checked)}
                    />
                    <Label className="text-slate-300">This person knows about my situation</Label>
                  </div>
                </CardContent>
              </Card>
            ))}

            <Button onClick={addContact} variant="outline" className="w-full border-slate-600 text-slate-300">
              + Add Another Contact
            </Button>
          </div>
        );

      case "warning_signs":
        return (
          <div className="space-y-6">
            <Alert className="bg-amber-500/10 border-amber-500/30">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <AlertTitle className="text-amber-300">Trust Your Instincts</AlertTitle>
              <AlertDescription className="text-amber-200/80">
                If something feels wrong, it probably is. The most dangerous time is often when leaving - plan carefully.
              </AlertDescription>
            </Alert>

            <div className="space-y-4">
              <Label className="text-slate-300">What signs tell you violence may be coming?</Label>
              <Textarea
                value={planData.warningSigns.join("\n")}
                onChange={(e) => updatePlanData("warningSigns", e.target.value.split("\n"))}
                placeholder="e.g., drinking, certain tone of voice, specific topics..."
                className="bg-slate-900/50 border-slate-600 text-white min-h-[100px]"
              />
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">Are there times that are more dangerous?</Label>
              <Textarea
                value={planData.dangerousTimes.join("\n")}
                onChange={(e) => updatePlanData("dangerousTimes", e.target.value.split("\n"))}
                placeholder="e.g., after work, weekends, holidays, payday..."
                className="bg-slate-900/50 border-slate-600 text-white min-h-[100px]"
              />
            </div>
          </div>
        );

      case "safe_locations":
        return (
          <div className="space-y-6">
            <Alert className="bg-teal-500/10 border-teal-500/30">
              <MapPin className="h-4 w-4 text-teal-400" />
              <AlertTitle className="text-teal-300">Rural Safety Tip</AlertTitle>
              <AlertDescription className="text-teal-200/80">
                In isolated areas, distance is a major factor. Know multiple routes, and identify safe stops 
                along the way. Gas stations, hospitals, and police stations are open 24/7.
              </AlertDescription>
            </Alert>

            {planData.safeLocations.map((location: any, index: number) => (
              <Card key={index} className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-lg text-white">Safe Location {index + 1}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-slate-300">Place Name</Label>
                      <Input
                        value={location.name}
                        onChange={(e) => updateLocation(index, "name", e.target.value)}
                        placeholder="Friend's house, shelter, etc."
                        className="bg-slate-900/50 border-slate-600 text-white"
                      />
                    </div>
                    <div>
                      <Label className="text-slate-300">Distance</Label>
                      <Input
                        value={location.distance}
                        onChange={(e) => updateLocation(index, "distance", e.target.value)}
                        placeholder="e.g., 5 miles, 2 hours"
                        className="bg-slate-900/50 border-slate-600 text-white"
                      />
                    </div>
                  </div>
                  <div>
                    <Label className="text-slate-300">Address</Label>
                    <Input
                      value={location.address}
                      onChange={(e) => updateLocation(index, "address", e.target.value)}
                      placeholder="Full address"
                      className="bg-slate-900/50 border-slate-600 text-white"
                    />
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      checked={location.hasKey}
                      onCheckedChange={(checked) => updateLocation(index, "hasKey", checked)}
                    />
                    <Label className="text-slate-300">I have a key or can get in anytime</Label>
                  </div>
                </CardContent>
              </Card>
            ))}

            <Button onClick={addLocation} variant="outline" className="w-full border-slate-600 text-slate-300">
              + Add Another Location
            </Button>
          </div>
        );

      case "emergency_bag":
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <Label className="text-slate-300">Where will you keep your emergency bag?</Label>
              <Input
                value={planData.emergencyBagLocation}
                onChange={(e) => updatePlanData("emergencyBagLocation", e.target.value)}
                placeholder="e.g., car trunk, friend's house, work locker"
                className="bg-slate-900/50 border-slate-600 text-white"
              />
              <p className="text-sm text-slate-400">Choose somewhere your abuser won't find it</p>
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">Items to include:</Label>
              <div className="grid grid-cols-2 gap-2">
                {emergencyBagItems.map((item) => (
                  <div key={item} className="flex items-center space-x-2">
                    <Checkbox
                      checked={planData.emergencyBagItems.includes(item)}
                      onCheckedChange={() => toggleChecklistItem("emergencyBagItems", item)}
                    />
                    <Label className="text-slate-300 text-sm">{item}</Label>
                  </div>
                ))}
              </div>
            </div>

            <Alert className="bg-teal-500/10 border-teal-500/30">
              <Shield className="h-4 w-4 text-teal-400" />
              <AlertTitle className="text-teal-300">Rural Safety Tip</AlertTitle>
              <AlertDescription className="text-teal-200/80">
                Include extra gas money - distances are longer. Pack for weather and include a paper map 
                in case phone service is unavailable.
              </AlertDescription>
            </Alert>
          </div>
        );

      case "documents":
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <Label className="text-slate-300">Which documents do you have access to?</Label>
              <div className="grid grid-cols-2 gap-2">
                {documentItems.map((item) => (
                  <div key={item} className="flex items-center space-x-2">
                    <Checkbox
                      checked={planData.documentsSecured.includes(item)}
                      onCheckedChange={() => toggleChecklistItem("documentsSecured", item)}
                    />
                    <Label className="text-slate-300 text-sm">{item}</Label>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">Where are these documents stored safely?</Label>
              <Input
                value={planData.documentLocation}
                onChange={(e) => updatePlanData("documentLocation", e.target.value)}
                placeholder="e.g., safe deposit box, friend's house"
                className="bg-slate-900/50 border-slate-600 text-white"
              />
            </div>

            <Alert className="bg-amber-500/10 border-amber-500/30">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <AlertTitle className="text-amber-300">Important</AlertTitle>
              <AlertDescription className="text-amber-200/80">
                If you can't get originals, copies are still helpful. Take photos of documents and 
                email them to a secure account your abuser doesn't know about.
              </AlertDescription>
            </Alert>
          </div>
        );

      case "financial_safety":
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Checkbox
                  checked={planData.hasHiddenMoney}
                  onCheckedChange={(checked) => updatePlanData("hasHiddenMoney", checked)}
                />
                <Label className="text-slate-300">I have been able to set aside some money</Label>
              </div>

              {planData.hasHiddenMoney && (
                <Input
                  value={planData.hiddenMoneyLocation}
                  onChange={(e) => updatePlanData("hiddenMoneyLocation", e.target.value)}
                  placeholder="Where is it hidden?"
                  className="bg-slate-900/50 border-slate-600 text-white"
                />
              )}

              <div className="flex items-center space-x-2">
                <Checkbox
                  checked={planData.hasSecretAccount}
                  onCheckedChange={(checked) => updatePlanData("hasSecretAccount", checked)}
                />
                <Label className="text-slate-300">I have access to a bank account my abuser doesn't know about</Label>
              </div>
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">Financial steps to consider:</Label>
              <div className="space-y-2">
                {[
                  "Open a secret bank account (use a friend's address)",
                  "Set aside small amounts of cash",
                  "Get a P.O. Box for mail",
                  "Know your credit score",
                  "Document shared assets",
                  "Research emergency financial assistance"
                ].map((step) => (
                  <div key={step} className="flex items-center space-x-2">
                    <Checkbox
                      checked={planData.financialSteps.includes(step)}
                      onCheckedChange={() => toggleChecklistItem("financialSteps", step)}
                    />
                    <Label className="text-slate-300 text-sm">{step}</Label>
                  </div>
                ))}
              </div>
            </div>

            <Alert className="bg-teal-500/10 border-teal-500/30">
              <Shield className="h-4 w-4 text-teal-400" />
              <AlertTitle className="text-teal-300">Rural Safety Tip</AlertTitle>
              <AlertDescription className="text-teal-200/80">
                In small towns, be careful which bank you use - word travels. Consider a bank in a nearby 
                town where you're not known.
              </AlertDescription>
            </Alert>
          </div>
        );

      case "technology_safety":
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Checkbox
                  checked={planData.phoneMonitored}
                  onCheckedChange={(checked) => updatePlanData("phoneMonitored", checked)}
                />
                <Label className="text-slate-300">I think my phone is being monitored</Label>
              </div>
              <p className="text-sm text-slate-400 ml-6">
                Signs: they know things you only said on the phone, apps you didn't install, battery draining quickly
              </p>

              <div className="flex items-center space-x-2">
                <Checkbox
                  checked={planData.locationTracked}
                  onCheckedChange={(checked) => updatePlanData("locationTracked", checked)}
                />
                <Label className="text-slate-300">My location is being tracked</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  checked={planData.socialMediaMonitored}
                  onCheckedChange={(checked) => updatePlanData("socialMediaMonitored", checked)}
                />
                <Label className="text-slate-300">My social media or email is monitored</Label>
              </div>
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">Technology safety steps:</Label>
              <div className="space-y-2">
                {[
                  "Use a safer device (library computer, friend's phone)",
                  "Create new email account they don't know about",
                  "Use private/incognito browsing",
                  "Clear browser history after searching for help",
                  "Check phone for tracking apps",
                  "Check car for GPS trackers",
                  "Turn off location sharing",
                  "Get a prepaid phone they don't know about"
                ].map((step) => (
                  <div key={step} className="flex items-center space-x-2">
                    <Checkbox
                      checked={planData.techSafetySteps.includes(step)}
                      onCheckedChange={() => toggleChecklistItem("techSafetySteps", step)}
                    />
                    <Label className="text-slate-300 text-sm">{step}</Label>
                  </div>
                ))}
              </div>
            </div>

            <Alert className="bg-amber-500/10 border-amber-500/30">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <AlertTitle className="text-amber-300">Warning</AlertTitle>
              <AlertDescription className="text-amber-200/80">
                If your phone is monitored, use a different device to search for help. Libraries have 
                free computers you can use privately.
              </AlertDescription>
            </Alert>
          </div>
        );

      case "children":
        return (
          <div className="space-y-6">
            <div className="flex items-center space-x-2">
              <Checkbox
                checked={planData.hasChildren}
                onCheckedChange={(checked) => updatePlanData("hasChildren", checked)}
              />
              <Label className="text-slate-300">I have children</Label>
            </div>

            {planData.hasChildren && (
              <>
                <div className="space-y-4">
                  <Label className="text-slate-300">Children's names and ages:</Label>
                  <Textarea
                    value={planData.childrenNames.join("\n")}
                    onChange={(e) => updatePlanData("childrenNames", e.target.value.split("\n"))}
                    placeholder="Name, age (one per line)"
                    className="bg-slate-900/50 border-slate-600 text-white"
                  />
                </div>

                <div className="space-y-4">
                  <Label className="text-slate-300">School/daycare information:</Label>
                  <Input
                    value={planData.schoolInfo}
                    onChange={(e) => updatePlanData("schoolInfo", e.target.value)}
                    placeholder="School name, address, contact"
                    className="bg-slate-900/50 border-slate-600 text-white"
                  />
                </div>

                <div className="space-y-4">
                  <Label className="text-slate-300">Steps for children's safety:</Label>
                  <div className="space-y-2">
                    {[
                      "Teach children how to call 911",
                      "Create a code word children understand",
                      "Identify a safe room in the house",
                      "Tell children it's not their fault",
                      "Pack comfort items for children",
                      "Inform school/daycare of situation",
                      "Get copies of children's documents"
                    ].map((step) => (
                      <div key={step} className="flex items-center space-x-2">
                        <Checkbox
                          checked={planData.childSafetySteps.includes(step)}
                          onCheckedChange={() => toggleChecklistItem("childSafetySteps", step)}
                        />
                        <Label className="text-slate-300 text-sm">{step}</Label>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        );

      case "pets":
        return (
          <div className="space-y-6">
            <div className="flex items-center space-x-2">
              <Checkbox
                checked={planData.hasPets}
                onCheckedChange={(checked) => updatePlanData("hasPets", checked)}
              />
              <Label className="text-slate-300">I have pets</Label>
            </div>

            {planData.hasPets && (
              <>
                <div className="space-y-4">
                  <Label className="text-slate-300">Pet information:</Label>
                  <Input
                    value={planData.petInfo}
                    onChange={(e) => updatePlanData("petInfo", e.target.value)}
                    placeholder="e.g., Dog, Max"
                    className="bg-slate-900/50 border-slate-600 text-white"
                  />
                </div>

                <div className="space-y-4">
                  <Label className="text-slate-300">Plan for your pet:</Label>
                  <Textarea
                    value={planData.petPlan}
                    onChange={(e) => updatePlanData("petPlan", e.target.value)}
                    placeholder="How will you keep your pet safe?"
                    className="bg-slate-900/50 border-slate-600 text-white"
                  />
                </div>

                <Alert className="bg-teal-500/10 border-teal-500/30">
                  <Heart className="h-4 w-4 text-teal-400" />
                  <AlertTitle className="text-teal-300">Pet Safety Resources</AlertTitle>
                  <AlertDescription className="text-teal-200/80">
                    Some shelters now accept pets or have partnerships with foster programs. 
                    Visit <a href="https://redrover.org/relief/safe-place/" className="underline">Safe Place for Pets</a> for help.
                  </AlertDescription>
                </Alert>
              </>
            )}
          </div>
        );

      case "exit_strategy":
        return (
          <div className="space-y-6">
            <Alert className="bg-red-500/10 border-red-500/30">
              <AlertTriangle className="h-4 w-4 text-red-400" />
              <AlertTitle className="text-red-300">Critical Safety Information</AlertTitle>
              <AlertDescription className="text-red-200/80">
                The most dangerous time is when leaving. Don't tell your abuser you're leaving. 
                If possible, leave when they're not home. Trust your instincts about timing.
              </AlertDescription>
            </Alert>

            <div className="space-y-4">
              <Label className="text-slate-300">When is the safest time for you to leave?</Label>
              <Input
                value={planData.bestTimeToLeave}
                onChange={(e) => updatePlanData("bestTimeToLeave", e.target.value)}
                placeholder="e.g., when they're at work, after they fall asleep"
                className="bg-slate-900/50 border-slate-600 text-white"
              />
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">How will you leave?</Label>
              <Input
                value={planData.transportationPlan}
                onChange={(e) => updatePlanData("transportationPlan", e.target.value)}
                placeholder="Drive, someone picks you up, taxi, etc."
                className="bg-slate-900/50 border-slate-600 text-white"
              />
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">Where will you go first?</Label>
              <Input
                value={planData.firstDestination}
                onChange={(e) => updatePlanData("firstDestination", e.target.value)}
                placeholder="Your first safe destination"
                className="bg-slate-900/50 border-slate-600 text-white"
              />
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">What is your backup plan?</Label>
              <Input
                value={planData.backupPlan}
                onChange={(e) => updatePlanData("backupPlan", e.target.value)}
                placeholder="Alternative plan if the first doesn't work"
                className="bg-slate-900/50 border-slate-600 text-white"
              />
            </div>

            <div className="space-y-4">
              <Label className="text-slate-300">Who will you call when you're safe?</Label>
              <Input
                value={planData.whoToCall}
                onChange={(e) => updatePlanData("whoToCall", e.target.value)}
                placeholder="Name and number"
                className="bg-slate-900/50 border-slate-600 text-white"
              />
            </div>

            <Alert className="bg-teal-500/10 border-teal-500/30">
              <Shield className="h-4 w-4 text-teal-400" />
              <AlertTitle className="text-teal-300">Rural Safety Tip</AlertTitle>
              <AlertDescription className="text-teal-200/80">
                Plan your route carefully - know where gas stations and safe stops are. Have a backup 
                route in case roads are blocked or watched. Consider weather and road conditions.
              </AlertDescription>
            </Alert>
          </div>
        );

      default:
        return null;
    }
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  // Show biometric lock if not unlocked
  if (!isUnlocked) {
    return (
      <BiometricLock
        onUnlock={() => setIsUnlocked(true)}
        title="Safety Plan Protected"
        description="Your safety plan contains sensitive escape information. Verify your identity to access it."
      />
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <Shield className="h-6 w-6 text-emerald-400" />
            <span className="text-xl font-bold text-white">Safety Plan</span>
          </Link>
          <div className="flex items-center gap-4">
            <Lock className="h-4 w-4 text-slate-400" />
            <span className="text-sm text-slate-400">Your plan is encrypted</span>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Progress */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-400">Progress</span>
            <span className="text-sm text-slate-400">{Math.round(progress)}% complete</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Step Navigation */}
        <div className="flex overflow-x-auto gap-2 mb-8 pb-2">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isComplete = completedSteps.includes(step.id);
            const isCurrent = index === currentStep;
            
            return (
              <button
                key={step.id}
                onClick={() => setCurrentStep(index)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
                  isCurrent
                    ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                    : isComplete
                    ? "bg-slate-700/50 text-slate-300"
                    : "bg-slate-800/50 text-slate-500"
                }`}
              >
                {isComplete ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                ) : (
                  <Icon className="h-4 w-4" />
                )}
                <span className="text-sm">{step.title}</span>
              </button>
            );
          })}
        </div>

        {/* Current Step Content */}
        <Card className="bg-slate-800/50 border-slate-700 mb-8">
          <CardHeader>
            <CardTitle className="text-2xl text-white flex items-center gap-3">
              {(() => {
                const Icon = currentStepData.icon;
                return <Icon className="h-6 w-6 text-emerald-400" />;
              })()}
              {currentStepData.title}
            </CardTitle>
            <CardDescription className="text-slate-400">
              {currentStepData.description}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {renderStepContent()}
          </CardContent>
        </Card>

        {/* Navigation Buttons */}
        <div className="flex items-center justify-between">
          <Button
            onClick={handlePrevious}
            disabled={currentStep === 0}
            variant="outline"
            className="border-slate-600 text-slate-300"
          >
            <ChevronLeft className="h-4 w-4 mr-2" />
            Previous
          </Button>

          <div className="flex gap-2">
            <Button
              onClick={handleSave}
              variant="outline"
              className="border-emerald-500/30 text-emerald-400"
              disabled={savePlanMutation.isPending}
            >
              {savePlanMutation.isPending ? (
                <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Saving...</>
              ) : (
                "Save Progress"
              )}
            </Button>
            <Button
              onClick={handleExport}
              variant="outline"
              className="border-amber-500/30 text-amber-400"
              disabled={isExporting || completedSteps.length === 0}
            >
              {isExporting ? (
                <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Exporting...</>
              ) : (
                <><Download className="h-4 w-4 mr-2" /> Export PDF</>
              )}
            </Button>
          </div>

          <Button
            onClick={handleNext}
            className="bg-emerald-500 hover:bg-emerald-600 text-white"
          >
            {currentStep === steps.length - 1 ? "Complete Plan" : "Next"}
            <ChevronRight className="h-4 w-4 ml-2" />
          </Button>
        </div>

        {/* Save Confirmation */}
        {showSavedMessage && (
          <div className="fixed bottom-4 right-4 bg-emerald-500 text-white px-4 py-2 rounded-lg shadow-lg">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4" />
              <span>Progress saved securely</span>
            </div>
          </div>
        )}

        {/* Emergency Resources */}
        <Card className="bg-red-500/10 border-red-500/30 mt-8">
          <CardHeader>
            <CardTitle className="text-red-300 flex items-center gap-2">
              <Phone className="h-5 w-5" />
              Emergency Resources
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-red-200">
              <strong>National DV Hotline:</strong> 1-800-799-7233 (24/7)
            </p>
            <p className="text-red-200">
              <strong>Text:</strong> START to 88788
            </p>
            <p className="text-red-200">
              <strong>Emergency:</strong> 911
            </p>
            <p className="text-sm text-red-200/70 mt-4">
              Remember: You deserve to be safe. This is not your fault.
            </p>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
