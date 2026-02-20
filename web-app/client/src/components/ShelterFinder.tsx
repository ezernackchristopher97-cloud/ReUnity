import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { 
  MapPin, 
  Navigation, 
  Phone, 
  ExternalLink, 
  Search,
  Loader2,
  Shield,
  Home,
  Heart,
  AlertTriangle,
  Clock,
  Users,
  Baby
} from "lucide-react";

interface Shelter {
  id: string;
  name: string;
  address: string;
  city: string;
  state: string;
  zip: string;
  phone: string;
  website?: string;
  distance?: number;
  services: string[];
  hours: string;
  accepts: string[];
  lat: number;
  lng: number;
}

// Comprehensive DV shelter database (real organizations)
const shelterDatabase: Shelter[] = [
  // Louisiana
  {
    id: "la1",
    name: "Faith House of Acadiana",
    address: "P.O. Box 93665",
    city: "Lafayette",
    state: "LA",
    zip: "70509",
    phone: "1-800-256-0047",
    website: "https://faithhouseacadiana.com",
    services: ["Emergency Shelter", "Counseling", "Legal Advocacy", "Children's Services"],
    hours: "24/7 Hotline",
    accepts: ["Women", "Children"],
    lat: 30.2241,
    lng: -92.0198
  },
  {
    id: "la2",
    name: "New Orleans Family Justice Center",
    address: "701 Loyola Ave",
    city: "New Orleans",
    state: "LA",
    zip: "70113",
    phone: "504-592-4005",
    website: "https://nofjc.org",
    services: ["Crisis Intervention", "Legal Services", "Counseling", "Case Management"],
    hours: "Mon-Fri 8:30am-4:30pm",
    accepts: ["All Genders", "Children", "LGBTQ+"],
    lat: 29.9511,
    lng: -90.0715
  },
  {
    id: "la3",
    name: "Chez Hope",
    address: "Confidential Location",
    city: "Lake Charles",
    state: "LA",
    zip: "70601",
    phone: "337-436-4552",
    services: ["Emergency Shelter", "Transitional Housing", "Counseling"],
    hours: "24/7",
    accepts: ["Women", "Children"],
    lat: 30.2266,
    lng: -93.2174
  },
  // Texas
  {
    id: "tx1",
    name: "The Family Place",
    address: "Confidential Location",
    city: "Dallas",
    state: "TX",
    zip: "75201",
    phone: "214-941-1991",
    website: "https://familyplace.org",
    services: ["Emergency Shelter", "Counseling", "Legal Services", "Children's Programs"],
    hours: "24/7 Hotline",
    accepts: ["Women", "Children", "Men", "LGBTQ+"],
    lat: 32.7767,
    lng: -96.7970
  },
  {
    id: "tx2",
    name: "Houston Area Women's Center",
    address: "Confidential Location",
    city: "Houston",
    state: "TX",
    zip: "77002",
    phone: "713-528-2121",
    website: "https://hawc.org",
    services: ["Emergency Shelter", "Counseling", "Legal Aid", "Children's Services"],
    hours: "24/7",
    accepts: ["Women", "Children"],
    lat: 29.7604,
    lng: -95.3698
  },
  {
    id: "tx3",
    name: "SafePlace Austin",
    address: "Confidential Location",
    city: "Austin",
    state: "TX",
    zip: "78701",
    phone: "512-267-7233",
    website: "https://safeaustin.org",
    services: ["Emergency Shelter", "Counseling", "Legal Services", "Prevention Education"],
    hours: "24/7 Hotline",
    accepts: ["All Genders", "Children", "LGBTQ+", "Pets"],
    lat: 30.2672,
    lng: -97.7431
  },
  // California
  {
    id: "ca1",
    name: "Peace Over Violence",
    address: "Confidential Location",
    city: "Los Angeles",
    state: "CA",
    zip: "90010",
    phone: "213-626-3393",
    website: "https://peaceoverviolence.org",
    services: ["Crisis Line", "Counseling", "Legal Services", "Support Groups"],
    hours: "24/7 Hotline",
    accepts: ["All Genders", "LGBTQ+"],
    lat: 34.0522,
    lng: -118.2437
  },
  {
    id: "ca2",
    name: "La Casa de las Madres",
    address: "Confidential Location",
    city: "San Francisco",
    state: "CA",
    zip: "94102",
    phone: "877-503-1850",
    website: "https://lacasa.org",
    services: ["Emergency Shelter", "Transitional Housing", "Counseling", "Children's Programs"],
    hours: "24/7",
    accepts: ["Women", "Children", "Trans Women"],
    lat: 37.7749,
    lng: -122.4194
  },
  // New York
  {
    id: "ny1",
    name: "Safe Horizon",
    address: "Multiple Locations",
    city: "New York",
    state: "NY",
    zip: "10038",
    phone: "1-800-621-4673",
    website: "https://safehorizon.org",
    services: ["Emergency Shelter", "Counseling", "Legal Services", "Court Advocacy"],
    hours: "24/7 Hotline",
    accepts: ["All Genders", "Children", "LGBTQ+"],
    lat: 40.7128,
    lng: -74.0060
  },
  // Florida
  {
    id: "fl1",
    name: "Hubbard House",
    address: "Confidential Location",
    city: "Jacksonville",
    state: "FL",
    zip: "32202",
    phone: "904-354-3114",
    website: "https://hubbardhouse.org",
    services: ["Emergency Shelter", "Counseling", "Legal Advocacy", "Children's Services"],
    hours: "24/7",
    accepts: ["Women", "Children"],
    lat: 30.3322,
    lng: -81.6557
  },
  {
    id: "fl2",
    name: "The Lodge - Domestic Violence Resource Center",
    address: "Confidential Location",
    city: "Miami",
    state: "FL",
    zip: "33130",
    phone: "305-693-1170",
    website: "https://thelodgemiami.org",
    services: ["Emergency Shelter", "Counseling", "Case Management"],
    hours: "24/7",
    accepts: ["Women", "Children"],
    lat: 25.7617,
    lng: -80.1918
  },
  // Rural areas - Mississippi
  {
    id: "ms1",
    name: "Gulf Coast Women's Center for Nonviolence",
    address: "Confidential Location",
    city: "Biloxi",
    state: "MS",
    zip: "39530",
    phone: "228-435-1968",
    services: ["Emergency Shelter", "Counseling", "Legal Services"],
    hours: "24/7 Hotline",
    accepts: ["Women", "Children"],
    lat: 30.3960,
    lng: -88.8853
  },
  // Arkansas
  {
    id: "ar1",
    name: "Women & Children First",
    address: "Confidential Location",
    city: "Little Rock",
    state: "AR",
    zip: "72201",
    phone: "501-376-3219",
    website: "https://wcfarkansas.org",
    services: ["Emergency Shelter", "Transitional Housing", "Counseling"],
    hours: "24/7",
    accepts: ["Women", "Children"],
    lat: 34.7465,
    lng: -92.2896
  },
  // Oklahoma
  {
    id: "ok1",
    name: "YWCA Oklahoma City",
    address: "Confidential Location",
    city: "Oklahoma City",
    state: "OK",
    zip: "73102",
    phone: "405-917-9922",
    website: "https://ywcaokc.org",
    services: ["Emergency Shelter", "Counseling", "Legal Services", "Children's Programs"],
    hours: "24/7 Hotline",
    accepts: ["Women", "Children"],
    lat: 35.4676,
    lng: -97.5164
  },
  // National Resource
  {
    id: "nat1",
    name: "National Domestic Violence Hotline",
    address: "National Resource",
    city: "Nationwide",
    state: "US",
    zip: "00000",
    phone: "1-800-799-7233",
    website: "https://thehotline.org",
    services: ["24/7 Hotline", "Safety Planning", "Shelter Referrals", "Crisis Support"],
    hours: "24/7",
    accepts: ["Everyone"],
    lat: 0,
    lng: 0
  }
];

// Calculate distance between two coordinates (Haversine formula)
function calculateDistance(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 3959; // Earth's radius in miles
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
    Math.sin(dLng/2) * Math.sin(dLng/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

export function ShelterFinder() {
  const [userLocation, setUserLocation] = useState<{lat: number, lng: number} | null>(null);
  const [isLocating, setIsLocating] = useState(false);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [shelters, setShelters] = useState<Shelter[]>([]);
  const [selectedFilter, setSelectedFilter] = useState<string>("all");

  // Get user's location
  const getUserLocation = () => {
    setIsLocating(true);
    setLocationError(null);
    
    if (!navigator.geolocation) {
      setLocationError("Geolocation is not supported by your browser");
      setIsLocating(false);
      return;
    }
    
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        setUserLocation({ lat: latitude, lng: longitude });
        setIsLocating(false);
        
        // Calculate distances and sort
        const withDistances = shelterDatabase
          .filter(s => s.lat !== 0) // Exclude national resources from distance calc
          .map(shelter => ({
            ...shelter,
            distance: calculateDistance(latitude, longitude, shelter.lat, shelter.lng)
          }))
          .sort((a, b) => (a.distance || 0) - (b.distance || 0));
        
        // Add national resource at the end
        const national = shelterDatabase.find(s => s.id === "nat1");
        if (national) {
          withDistances.push({ ...national, distance: 99999 }); // Large number to sort last
        }
        
        setShelters(withDistances);
      },
      (error) => {
        setIsLocating(false);
        switch (error.code) {
          case error.PERMISSION_DENIED:
            setLocationError("Location access denied. Please enable location services or search by city/state.");
            break;
          case error.POSITION_UNAVAILABLE:
            setLocationError("Location unavailable. Please try again or search by city/state.");
            break;
          case error.TIMEOUT:
            setLocationError("Location request timed out. Please try again.");
            break;
          default:
            setLocationError("Unable to get location. Please search by city/state.");
        }
        // Show all shelters without distance
        setShelters(shelterDatabase);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  };

  // Search by text
  const searchShelters = () => {
    if (!searchQuery.trim()) {
      setShelters(shelterDatabase);
      return;
    }
    
    const query = searchQuery.toLowerCase();
    const filtered = shelterDatabase.filter(shelter => 
      shelter.city.toLowerCase().includes(query) ||
      shelter.state.toLowerCase().includes(query) ||
      shelter.name.toLowerCase().includes(query) ||
      shelter.zip.includes(query)
    );
    setShelters(filtered);
  };

  // Filter by service type
  const filteredShelters = shelters.filter(shelter => {
    if (selectedFilter === "all") return true;
    if (selectedFilter === "24/7") return shelter.hours.includes("24/7");
    if (selectedFilter === "children") return shelter.accepts.some(a => a.toLowerCase().includes("children"));
    if (selectedFilter === "lgbtq") return shelter.accepts.some(a => a.toLowerCase().includes("lgbtq"));
    if (selectedFilter === "pets") return shelter.accepts.some(a => a.toLowerCase().includes("pets"));
    return true;
  });

  // Make call
  const makeCall = (phone: string) => {
    window.location.href = `tel:${phone.replace(/[^0-9+\-]/g, "")}`;
  };

  // Open directions
  const openDirections = (shelter: Shelter) => {
    const address = encodeURIComponent(`${shelter.address}, ${shelter.city}, ${shelter.state} ${shelter.zip}`);
    window.open(`https://www.google.com/maps/dir/?api=1&destination=${address}`, "_blank");
  };

  return (
    <div className="space-y-6">
      {/* Safety Warning */}
      <Card className="bg-amber-900/20 border-amber-700/50">
        <CardContent className="py-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-medium text-amber-300 mb-1">Safety First</p>
              <p className="text-amber-200/80">
                If you're in immediate danger, call 911. If your device may be monitored, 
                consider using a public computer or a friend's phone to search for resources.
                You can also call the National DV Hotline at 1-800-799-7233.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Location & Search */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-emerald-400" />
            <CardTitle className="text-lg text-white">Find Nearby Shelters</CardTitle>
          </div>
          <CardDescription className="text-slate-400">
            Locate domestic violence shelters and resources near you
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Location Button */}
          <Button
            onClick={getUserLocation}
            disabled={isLocating}
            className="w-full bg-emerald-600 hover:bg-emerald-500"
          >
            {isLocating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Finding your location...
              </>
            ) : (
              <>
                <Navigation className="w-4 h-4 mr-2" />
                Use My Location
              </>
            )}
          </Button>
          
          {locationError && (
            <p className="text-sm text-amber-400">{locationError}</p>
          )}
          
          {userLocation && (
            <p className="text-sm text-emerald-400">
              ✓ Location found - showing nearest shelters
            </p>
          )}
          
          {/* Text Search */}
          <div className="flex gap-2">
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by city, state, or zip code"
              className="bg-slate-900 border-slate-600"
              onKeyDown={(e) => e.key === "Enter" && searchShelters()}
            />
            <Button onClick={searchShelters} variant="outline" className="border-slate-600">
              <Search className="w-4 h-4" />
            </Button>
          </div>
          
          {/* Filters */}
          <div className="flex flex-wrap gap-2">
            {[
              { id: "all", label: "All", icon: Home },
              { id: "24/7", label: "24/7", icon: Clock },
              { id: "children", label: "Children", icon: Baby },
              { id: "lgbtq", label: "LGBTQ+", icon: Heart },
              { id: "pets", label: "Pets OK", icon: Heart },
            ].map(filter => (
              <Button
                key={filter.id}
                variant={selectedFilter === filter.id ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedFilter(filter.id)}
                className={selectedFilter === filter.id 
                  ? "bg-emerald-600 hover:bg-emerald-500" 
                  : "border-slate-600 text-slate-300"
                }
              >
                <filter.icon className="w-3 h-3 mr-1" />
                {filter.label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Shelter Results */}
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-white flex items-center gap-2">
          <Shield className="w-5 h-5 text-emerald-400" />
          {filteredShelters.length} Resources Found
        </h3>
        
        {filteredShelters.map((shelter) => (
          <Card key={shelter.id} className="bg-slate-800/50 border-slate-700">
            <CardContent className="py-4">
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-start gap-2">
                    <h4 className="font-medium text-white">{shelter.name}</h4>
                    {shelter.distance !== undefined && (
                      <span className="text-xs bg-emerald-600/30 text-emerald-300 px-2 py-0.5 rounded">
                        {shelter.distance.toFixed(1)} mi
                      </span>
                    )}
                  </div>
                  
                  <p className="text-sm text-slate-400 mt-1">
                    {shelter.city}, {shelter.state} {shelter.zip}
                  </p>
                  
                  <div className="flex flex-wrap gap-1 mt-2">
                    {shelter.services.slice(0, 3).map((service, idx) => (
                      <span 
                        key={idx}
                        className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded"
                      >
                        {service}
                      </span>
                    ))}
                    {shelter.services.length > 3 && (
                      <span className="text-xs text-slate-500">
                        +{shelter.services.length - 3} more
                      </span>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {shelter.hours}
                    </span>
                    <span className="flex items-center gap-1">
                      <Users className="w-3 h-3" />
                      {shelter.accepts.join(", ")}
                    </span>
                  </div>
                </div>
                
                <div className="flex flex-row md:flex-col gap-2">
                  <Button
                    onClick={() => makeCall(shelter.phone)}
                    className="bg-emerald-600 hover:bg-emerald-500 flex-1 md:flex-none"
                  >
                    <Phone className="w-4 h-4 mr-2" />
                    Call
                  </Button>
                  
                  {shelter.lat !== 0 && (
                    <Button
                      onClick={() => openDirections(shelter)}
                      variant="outline"
                      className="border-slate-600 flex-1 md:flex-none"
                    >
                      <Navigation className="w-4 h-4 mr-2" />
                      Directions
                    </Button>
                  )}
                  
                  {shelter.website && (
                    <Button
                      onClick={() => window.open(shelter.website, "_blank")}
                      variant="ghost"
                      size="sm"
                      className="text-slate-400 hover:text-white"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
        
        {filteredShelters.length === 0 && (
          <Card className="bg-slate-800/30 border-slate-700">
            <CardContent className="py-8 text-center">
              <MapPin className="w-12 h-12 mx-auto mb-3 text-slate-600" />
              <p className="text-slate-400">No shelters found matching your search</p>
              <p className="text-sm text-slate-500 mt-1">
                Try a different location or call the National DV Hotline: 1-800-799-7233
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Additional Resources */}
      <Card className="bg-slate-800/30 border-slate-700/50">
        <CardContent className="py-4">
          <div className="flex items-start gap-3">
            <Heart className="w-5 h-5 text-emerald-400 mt-0.5" />
            <div className="text-sm text-slate-400">
              <p className="font-medium text-slate-300 mb-1">Need More Help?</p>
              <ul className="space-y-1">
                <li>• <strong>National DV Hotline:</strong> 1-800-799-7233 (24/7)</li>
                <li>• <strong>Text:</strong> START to 88788</li>
                <li>• <strong>Chat:</strong> thehotline.org</li>
                <li>• Shelter addresses are often confidential for safety</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ShelterFinder;
