import { useState } from "react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Heart, 
  Phone, 
  MapPin, 
  Headphones,
  ArrowLeft,
  Shield,
  Smartphone
} from "lucide-react";
import { EmergencyContacts } from "@/components/EmergencyContacts";
import { ShelterFinder } from "@/components/ShelterFinder";
import { GuidedMeditation } from "@/components/GuidedMeditation";
import { TrustedDevicePairing } from "@/components/TrustedDevicePairing";
import { CheckInSystem } from "@/components/CheckInSystem";

export default function Resources() {
  const [activeTab, setActiveTab] = useState("contacts");

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <Heart className="h-6 w-6 text-emerald-400" />
            <span className="text-xl font-bold text-white">Emergency Resources</span>
          </Link>
          <Link href="/">
            <Button variant="ghost" size="sm" className="text-slate-400 hover:text-white">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
          </Link>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Hero Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-emerald-500/20 mb-4">
            <Shield className="w-8 h-8 text-emerald-400" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">
            You Are Not Alone
          </h1>
          <p className="text-slate-400 max-w-xl mx-auto">
            Access emergency contacts, find nearby shelters, or use guided meditations 
            to help you through difficult moments. Help is always available.
          </p>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid grid-cols-4 bg-slate-800/50 p-1 rounded-lg">
            <TabsTrigger 
              value="contacts" 
              className="flex items-center gap-2 data-[state=active]:bg-emerald-600"
            >
              <Phone className="w-4 h-4" />
              <span className="hidden sm:inline">Contacts</span>
            </TabsTrigger>
            <TabsTrigger 
              value="shelters"
              className="flex items-center gap-2 data-[state=active]:bg-emerald-600"
            >
              <MapPin className="w-4 h-4" />
              <span className="hidden sm:inline">Shelters</span>
            </TabsTrigger>
            <TabsTrigger 
              value="meditation"
              className="flex items-center gap-2 data-[state=active]:bg-emerald-600"
            >
              <Headphones className="w-4 h-4" />
              <span className="hidden sm:inline">Meditation</span>
            </TabsTrigger>
            <TabsTrigger 
              value="safety"
              className="flex items-center gap-2 data-[state=active]:bg-emerald-600"
            >
              <Smartphone className="w-4 h-4" />
              <span className="hidden sm:inline">Safety</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="contacts" className="mt-6">
            <EmergencyContacts />
          </TabsContent>

          <TabsContent value="shelters" className="mt-6">
            <ShelterFinder />
          </TabsContent>

          <TabsContent value="meditation" className="mt-6">
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h2 className="text-xl font-semibold text-white mb-2">
                  Guided Meditation
                </h2>
                <p className="text-slate-400">
                  Choose a meditation technique and let the guided audio help you find calm.
                </p>
              </div>
              <GuidedMeditation />
            </div>
          </TabsContent>

          <TabsContent value="safety" className="mt-6">
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h2 className="text-xl font-semibold text-white mb-2">
                  Safety Systems
                </h2>
                <p className="text-slate-400">
                  Set up wellness check-ins and pair with trusted family members for emergency alerts.
                </p>
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                <CheckInSystem />
                <TrustedDevicePairing />
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Bottom Safety Note */}
        <div className="mt-12 text-center text-sm text-slate-500">
          <p>
            If you are in immediate danger, please call <strong className="text-red-400">911</strong>.
          </p>
          <p className="mt-1">
            National Domestic Violence Hotline: <strong className="text-emerald-400">1-800-799-7233</strong>
          </p>
        </div>
      </main>
    </div>
  );
}
