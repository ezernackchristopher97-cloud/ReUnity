import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { 
  Phone, 
  Plus, 
  Trash2, 
  Edit2, 
  Save, 
  X, 
  Shield, 
  Heart,
  AlertTriangle,
  User,
  MessageSquare
} from "lucide-react";

interface EmergencyContact {
  id: string;
  name: string;
  phone: string;
  relationship: string;
  codeWord?: string;
  isPrimary: boolean;
  notifyOnHighRisk?: boolean;
}

interface EmergencyContactsProps {
  onHighRiskAlert?: boolean;
  riskLevel?: 'low' | 'moderate' | 'elevated' | 'high';
}

const CONTACTS_KEY = "reunity_emergency_contacts";

// Default crisis hotlines
const crisisHotlines = [
  { name: "988 Suicide & Crisis Lifeline", phone: "988", description: "24/7 crisis support" },
  { name: "National DV Hotline", phone: "1-800-799-7233", description: "Domestic violence help" },
  { name: "Crisis Text Line", phone: "741741", description: "Text HOME to connect" },
  { name: "911 Emergency", phone: "911", description: "Immediate danger" },
];

export function EmergencyContacts({ onHighRiskAlert = false, riskLevel = 'low' }: EmergencyContactsProps = {}) {
  const [contacts, setContacts] = useState<EmergencyContact[]>([]);
  const [isAdding, setIsAdding] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showHotlines, setShowHotlines] = useState(true);
  const [showHighRiskDialog, setShowHighRiskDialog] = useState(false);
  
  // Show high risk dialog when triggered
  useEffect(() => {
    if (onHighRiskAlert && riskLevel === 'high') {
      setShowHighRiskDialog(true);
    }
  }, [onHighRiskAlert, riskLevel]);

  // Form state
  const [formData, setFormData] = useState({
    name: "",
    phone: "",
    relationship: "",
    codeWord: "",
    isPrimary: false
  });

  // Load contacts from localStorage
  useEffect(() => {
    const stored = localStorage.getItem(CONTACTS_KEY);
    if (stored) {
      try {
        setContacts(JSON.parse(stored));
      } catch (e) {
        console.error("Failed to parse contacts:", e);
      }
    }
  }, []);

  // Save contacts to localStorage
  const saveContacts = (newContacts: EmergencyContact[]) => {
    setContacts(newContacts);
    localStorage.setItem(CONTACTS_KEY, JSON.stringify(newContacts));
  };

  // Add new contact
  const addContact = () => {
    if (!formData.name || !formData.phone) return;
    
    const newContact: EmergencyContact = {
      id: Date.now().toString(),
      name: formData.name,
      phone: formData.phone,
      relationship: formData.relationship,
      codeWord: formData.codeWord,
      isPrimary: formData.isPrimary || contacts.length === 0
    };
    
    // If this is primary, unset others
    let updatedContacts = contacts;
    if (newContact.isPrimary) {
      updatedContacts = contacts.map(c => ({ ...c, isPrimary: false }));
    }
    
    saveContacts([...updatedContacts, newContact]);
    resetForm();
  };

  // Update contact
  const updateContact = () => {
    if (!editingId || !formData.name || !formData.phone) return;
    
    let updatedContacts = contacts.map(c => {
      if (c.id === editingId) {
        return {
          ...c,
          name: formData.name,
          phone: formData.phone,
          relationship: formData.relationship,
          codeWord: formData.codeWord,
          isPrimary: formData.isPrimary
        };
      }
      // If editing contact is now primary, unset others
      if (formData.isPrimary && c.id !== editingId) {
        return { ...c, isPrimary: false };
      }
      return c;
    });
    
    saveContacts(updatedContacts);
    resetForm();
  };

  // Delete contact
  const deleteContact = (id: string) => {
    const updatedContacts = contacts.filter(c => c.id !== id);
    // If deleted was primary, make first remaining contact primary
    if (updatedContacts.length > 0 && !updatedContacts.some(c => c.isPrimary)) {
      updatedContacts[0].isPrimary = true;
    }
    saveContacts(updatedContacts);
  };

  // Start editing
  const startEdit = (contact: EmergencyContact) => {
    setFormData({
      name: contact.name,
      phone: contact.phone,
      relationship: contact.relationship,
      codeWord: contact.codeWord || "",
      isPrimary: contact.isPrimary
    });
    setEditingId(contact.id);
    setIsAdding(false);
  };

  // Reset form
  const resetForm = () => {
    setFormData({ name: "", phone: "", relationship: "", codeWord: "", isPrimary: false });
    setIsAdding(false);
    setEditingId(null);
  };

  // Make call
  const makeCall = (phone: string) => {
    window.location.href = `tel:${phone.replace(/[^0-9+]/g, "")}`;
  };

  // Send text
  const sendText = (phone: string, codeWord?: string) => {
    const message = codeWord ? encodeURIComponent(codeWord) : "";
    window.location.href = `sms:${phone.replace(/[^0-9+]/g, "")}${message ? `?body=${message}` : ""}`;
  };

  return (
    <div className="space-y-6">
      {/* Crisis Hotlines Section */}
      <Card className="bg-red-900/20 border-red-800/50">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-400" />
              <CardTitle className="text-lg text-red-300">Crisis Hotlines</CardTitle>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowHotlines(!showHotlines)}
              className="text-red-400"
            >
              {showHotlines ? "Hide" : "Show"}
            </Button>
          </div>
          <CardDescription className="text-red-300/70">
            Professional help available 24/7
          </CardDescription>
        </CardHeader>
        
        {showHotlines && (
          <CardContent className="grid gap-2">
            {crisisHotlines.map((hotline, idx) => (
              <div 
                key={idx}
                className="flex items-center justify-between bg-red-950/30 rounded-lg p-3"
              >
                <div>
                  <p className="font-medium text-red-200">{hotline.name}</p>
                  <p className="text-sm text-red-300/70">{hotline.description}</p>
                </div>
                <Button
                  onClick={() => makeCall(hotline.phone)}
                  className="bg-red-600 hover:bg-red-500"
                >
                  <Phone className="w-4 h-4 mr-2" />
                  {hotline.phone}
                </Button>
              </div>
            ))}
          </CardContent>
        )}
      </Card>

      {/* Personal Emergency Contacts */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-emerald-400" />
              <CardTitle className="text-lg text-white">My Emergency Contacts</CardTitle>
            </div>
            {!isAdding && !editingId && (
              <Button
                onClick={() => setIsAdding(true)}
                size="sm"
                className="bg-emerald-600 hover:bg-emerald-500"
              >
                <Plus className="w-4 h-4 mr-1" />
                Add Contact
              </Button>
            )}
          </div>
          <CardDescription className="text-slate-400">
            Trusted people you can reach quickly in an emergency
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Add/Edit Form */}
          {(isAdding || editingId) && (
            <div className="bg-slate-900/50 rounded-lg p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Name</Label>
                  <Input
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="Contact name"
                    className="bg-slate-800 border-slate-600"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Phone</Label>
                  <Input
                    value={formData.phone}
                    onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                    placeholder="Phone number"
                    type="tel"
                    className="bg-slate-800 border-slate-600"
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Relationship</Label>
                  <Input
                    value={formData.relationship}
                    onChange={(e) => setFormData({ ...formData, relationship: e.target.value })}
                    placeholder="e.g., Friend, Sister"
                    className="bg-slate-800 border-slate-600"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Code Word (optional)</Label>
                  <Input
                    value={formData.codeWord}
                    onChange={(e) => setFormData({ ...formData, codeWord: e.target.value })}
                    placeholder="Secret phrase for texts"
                    className="bg-slate-800 border-slate-600"
                  />
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="isPrimary"
                  checked={formData.isPrimary}
                  onChange={(e) => setFormData({ ...formData, isPrimary: e.target.checked })}
                  className="rounded border-slate-600"
                />
                <Label htmlFor="isPrimary" className="text-slate-300 cursor-pointer">
                  Primary contact (shown first)
                </Label>
              </div>
              
              <div className="flex gap-2">
                <Button
                  onClick={editingId ? updateContact : addContact}
                  className="bg-emerald-600 hover:bg-emerald-500"
                >
                  <Save className="w-4 h-4 mr-1" />
                  {editingId ? "Update" : "Save"}
                </Button>
                <Button variant="outline" onClick={resetForm} className="border-slate-600">
                  <X className="w-4 h-4 mr-1" />
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {/* Contact List */}
          {contacts.length === 0 && !isAdding ? (
            <div className="text-center py-8 text-slate-400">
              <User className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No emergency contacts added yet</p>
              <p className="text-sm">Add trusted people you can reach quickly</p>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Sort to show primary first */}
              {[...contacts].sort((a, b) => (b.isPrimary ? 1 : 0) - (a.isPrimary ? 1 : 0)).map((contact) => (
                <div
                  key={contact.id}
                  className={`rounded-lg p-4 ${
                    contact.isPrimary 
                      ? "bg-emerald-900/30 border border-emerald-700/50" 
                      : "bg-slate-900/50"
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <h3 className="font-medium text-white">{contact.name}</h3>
                        {contact.isPrimary && (
                          <span className="text-xs bg-emerald-600 text-white px-2 py-0.5 rounded">
                            Primary
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-slate-400">{contact.relationship}</p>
                      <p className="text-sm text-slate-500">{contact.phone}</p>
                      {contact.codeWord && (
                        <p className="text-xs text-emerald-400 mt-1">
                          Code word set ✓
                        </p>
                      )}
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {/* Quick Actions */}
                      <Button
                        onClick={() => makeCall(contact.phone)}
                        size="sm"
                        className="bg-emerald-600 hover:bg-emerald-500"
                      >
                        <Phone className="w-4 h-4" />
                      </Button>
                      <Button
                        onClick={() => sendText(contact.phone, contact.codeWord)}
                        size="sm"
                        variant="outline"
                        className="border-emerald-600 text-emerald-400 hover:bg-emerald-600/20"
                      >
                        <MessageSquare className="w-4 h-4" />
                      </Button>
                      
                      {/* Edit/Delete */}
                      <Button
                        onClick={() => startEdit(contact)}
                        size="sm"
                        variant="ghost"
                        className="text-slate-400 hover:text-white"
                      >
                        <Edit2 className="w-4 h-4" />
                      </Button>
                      <Button
                        onClick={() => deleteContact(contact.id)}
                        size="sm"
                        variant="ghost"
                        className="text-red-400 hover:text-red-300"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick Dial Instructions */}
      <Card className="bg-slate-800/30 border-slate-700/50">
        <CardContent className="py-4">
          <div className="flex items-start gap-3">
            <Heart className="w-5 h-5 text-emerald-400 mt-0.5" />
            <div className="text-sm text-slate-400">
              <p className="font-medium text-slate-300 mb-1">Quick Dial Tips</p>
              <ul className="space-y-1">
                <li>• Tap the phone icon to call instantly</li>
                <li>• Tap the message icon to send a text (with code word if set)</li>
                <li>• Code words let you signal for help without typing</li>
                <li>• Your primary contact appears first for fastest access</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// High Risk Alert Dialog Component
export function HighRiskAlertDialog({ 
  isOpen, 
  onClose, 
  contacts 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  contacts: EmergencyContact[];
}) {
  const primaryContact = contacts.find(c => c.isPrimary);
  const highRiskContacts = contacts.filter(c => c.notifyOnHighRisk !== false);

  const makeCall = (phone: string) => {
    window.location.href = `tel:${phone.replace(/[^0-9+]/g, "")}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-red-500/50 rounded-xl p-6 max-w-md mx-4 shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center animate-pulse">
            <AlertTriangle className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">High Risk Detected</h3>
            <p className="text-sm text-zinc-400">Your wellness data suggests you may need support</p>
          </div>
        </div>
        
        <p className="text-zinc-300 mb-4">
          Would you like to reach out to someone? One tap to call your emergency contact or crisis line.
        </p>
        
        <div className="space-y-2">
          {primaryContact && (
            <Button
              className="w-full bg-emerald-600 hover:bg-emerald-700"
              onClick={() => {
                makeCall(primaryContact.phone);
                onClose();
              }}
            >
              <Phone className="w-4 h-4 mr-2" />
              Call {primaryContact.name} ({primaryContact.relationship})
            </Button>
          )}
          
          <Button
            variant="outline"
            className="w-full border-red-500/50 text-red-400 hover:bg-red-500/20"
            onClick={() => {
              makeCall('988');
              onClose();
            }}
          >
            <Phone className="w-4 h-4 mr-2" />
            Call 988 Crisis Lifeline
          </Button>
          
          {highRiskContacts.length > 1 && (
            <div className="pt-2 border-t border-zinc-800">
              <p className="text-xs text-zinc-500 mb-2">Other emergency contacts:</p>
              {highRiskContacts.filter(c => !c.isPrimary).slice(0, 2).map(contact => (
                <Button
                  key={contact.id}
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-zinc-400 hover:text-white"
                  onClick={() => {
                    makeCall(contact.phone);
                    onClose();
                  }}
                >
                  <Phone className="w-3 h-3 mr-2" />
                  {contact.name}
                </Button>
              ))}
            </div>
          )}
          
          <Button
            variant="ghost"
            className="w-full text-zinc-500"
            onClick={onClose}
          >
            I'm okay for now
          </Button>
        </div>
        
        <p className="text-xs text-zinc-600 mt-4 text-center">
          <Shield className="w-3 h-3 inline mr-1" />
          Your privacy is protected. No data is shared without consent.
        </p>
      </div>
    </div>
  );
}

// Hook for triggering high risk alerts from other components
export function useEmergencyAlert() {
  const [showAlert, setShowAlert] = useState(false);
  
  const triggerHighRiskAlert = () => {
    setShowAlert(true);
  };
  
  return { showAlert, triggerHighRiskAlert, setShowAlert };
}

export default EmergencyContacts;
