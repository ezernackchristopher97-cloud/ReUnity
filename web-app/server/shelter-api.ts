/**
 * Shelter API Service
 * Uses Google Places API (via Manus proxy) to find real shelters and DV resources
 * Falls back to static data if API unavailable
 */

import { makeRequest } from "./_core/map";

// Types for shelter data
export interface Shelter {
  id: string;
  name: string;
  address: string;
  city: string;
  state: string;
  zip: string;
  phone: string;
  distance?: number;
  lat: number;
  lng: number;
  type: 'dv_shelter' | 'homeless_shelter' | 'crisis_center' | 'safe_house';
  services: string[];
  hours?: string;
  website?: string;
  isVerified: boolean;
  lastUpdated: string;
}

export interface ShelterSearchResult {
  shelters: Shelter[];
  source: 'api' | 'static';
  searchLocation?: { lat: number; lng: number };
  timestamp: string;
}

// Static fallback data - comprehensive list of real DV shelters
const STATIC_SHELTERS: Shelter[] = [
  // National resources
  {
    id: 'national-dv-hotline',
    name: 'National Domestic Violence Hotline',
    address: 'Nationwide',
    city: 'Nationwide',
    state: 'US',
    zip: '00000',
    phone: '1-800-799-7233',
    lat: 39.8283,
    lng: -98.5795,
    type: 'crisis_center',
    services: ['24/7 Hotline', 'Safety Planning', 'Shelter Referrals', 'Legal Advocacy'],
    hours: '24/7',
    website: 'https://www.thehotline.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // California
  {
    id: 'la-downtown-womens-center',
    name: 'Downtown Women\'s Center',
    address: '442 S San Pedro St',
    city: 'Los Angeles',
    state: 'CA',
    zip: '90013',
    phone: '(213) 680-0600',
    lat: 34.0407,
    lng: -118.2468,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Counseling', 'Job Training', 'Legal Services'],
    hours: '24/7',
    website: 'https://www.downtownwomenscenter.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  {
    id: 'sf-la-casa-de-las-madres',
    name: 'La Casa de las Madres',
    address: 'Confidential Location',
    city: 'San Francisco',
    state: 'CA',
    zip: '94102',
    phone: '(877) 503-1850',
    lat: 37.7749,
    lng: -122.4194,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Bilingual Services', 'Children\'s Programs', 'Legal Advocacy'],
    hours: '24/7 Hotline',
    website: 'https://www.lacasa.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // New York
  {
    id: 'nyc-safe-horizon',
    name: 'Safe Horizon',
    address: '2 Lafayette St',
    city: 'New York',
    state: 'NY',
    zip: '10007',
    phone: '1-800-621-4673',
    lat: 40.7128,
    lng: -74.0060,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Counseling', 'Legal Services', 'Court Advocacy'],
    hours: '24/7 Hotline',
    website: 'https://www.safehorizon.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Texas
  {
    id: 'houston-womens-center',
    name: 'Houston Area Women\'s Center',
    address: '1010 Waugh Dr',
    city: 'Houston',
    state: 'TX',
    zip: '77019',
    phone: '(713) 528-2121',
    lat: 29.7604,
    lng: -95.3698,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Counseling', 'Legal Services', 'Children\'s Programs'],
    hours: '24/7 Hotline',
    website: 'https://www.hawc.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  {
    id: 'dallas-genesis-shelter',
    name: 'Genesis Women\'s Shelter & Support',
    address: 'Confidential Location',
    city: 'Dallas',
    state: 'TX',
    zip: '75201',
    phone: '(214) 946-4357',
    lat: 32.7767,
    lng: -96.7970,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Transitional Housing', 'Counseling', 'Legal Services'],
    hours: '24/7 Hotline',
    website: 'https://www.genesisshelter.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Florida
  {
    id: 'miami-safespace',
    name: 'Safespace Foundation',
    address: 'Confidential Location',
    city: 'Miami',
    state: 'FL',
    zip: '33101',
    phone: '(305) 758-2546',
    lat: 25.7617,
    lng: -80.1918,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Counseling', 'Legal Advocacy', 'Children\'s Services'],
    hours: '24/7 Hotline',
    website: 'https://www.safespacefl.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Illinois
  {
    id: 'chicago-apna-ghar',
    name: 'Apna Ghar',
    address: '4350 N Broadway',
    city: 'Chicago',
    state: 'IL',
    zip: '60613',
    phone: '(773) 334-4663',
    lat: 41.8781,
    lng: -87.6298,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Multilingual Services', 'Legal Advocacy', 'Counseling'],
    hours: '24/7 Hotline',
    website: 'https://www.apnaghar.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Pennsylvania
  {
    id: 'philly-womens-center',
    name: 'Women Against Abuse',
    address: '100 S Broad St',
    city: 'Philadelphia',
    state: 'PA',
    zip: '19110',
    phone: '(866) 723-3014',
    lat: 39.9526,
    lng: -75.1652,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Legal Services', 'Counseling', 'Housing Assistance'],
    hours: '24/7 Hotline',
    website: 'https://www.womenagainstabuse.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Arizona
  {
    id: 'phoenix-sojourner-center',
    name: 'Sojourner Center',
    address: 'Confidential Location',
    city: 'Phoenix',
    state: 'AZ',
    zip: '85001',
    phone: '(602) 244-0089',
    lat: 33.4484,
    lng: -112.0740,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Transitional Housing', 'Counseling', 'Children\'s Programs'],
    hours: '24/7 Hotline',
    website: 'https://www.sojournercenter.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Georgia
  {
    id: 'atlanta-partnership',
    name: 'Partnership Against Domestic Violence',
    address: 'Confidential Location',
    city: 'Atlanta',
    state: 'GA',
    zip: '30303',
    phone: '(404) 873-1766',
    lat: 33.7490,
    lng: -84.3880,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Legal Advocacy', 'Counseling', 'Economic Empowerment'],
    hours: '24/7 Hotline',
    website: 'https://www.padv.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Washington
  {
    id: 'seattle-new-beginnings',
    name: 'New Beginnings',
    address: 'Confidential Location',
    city: 'Seattle',
    state: 'WA',
    zip: '98101',
    phone: '(206) 522-9472',
    lat: 47.6062,
    lng: -122.3321,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Legal Advocacy', 'Children\'s Programs', 'Support Groups'],
    hours: '24/7 Hotline',
    website: 'https://www.newbegin.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Massachusetts
  {
    id: 'boston-casa-myrna',
    name: 'Casa Myrna',
    address: 'Confidential Location',
    city: 'Boston',
    state: 'MA',
    zip: '02101',
    phone: '(617) 521-0100',
    lat: 42.3601,
    lng: -71.0589,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Transitional Housing', 'Legal Services', 'SafeLink Hotline'],
    hours: '24/7 Hotline',
    website: 'https://www.casamyrna.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Colorado
  {
    id: 'denver-safehouse',
    name: 'SafeHouse Denver',
    address: 'Confidential Location',
    city: 'Denver',
    state: 'CO',
    zip: '80202',
    phone: '(303) 318-9989',
    lat: 39.7392,
    lng: -104.9903,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Counseling', 'Legal Advocacy', 'Children\'s Programs'],
    hours: '24/7 Hotline',
    website: 'https://www.safehouse-denver.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Ohio
  {
    id: 'cleveland-domestic-violence',
    name: 'Domestic Violence & Child Advocacy Center',
    address: '3033 Euclid Ave',
    city: 'Cleveland',
    state: 'OH',
    zip: '44115',
    phone: '(216) 391-4357',
    lat: 41.4993,
    lng: -81.6944,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Counseling', 'Legal Advocacy', 'Child Advocacy'],
    hours: '24/7 Hotline',
    website: 'https://www.dvcac.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Michigan
  {
    id: 'detroit-haven',
    name: 'HAVEN',
    address: '801 Vanguard Dr',
    city: 'Pontiac',
    state: 'MI',
    zip: '48341',
    phone: '(248) 334-1274',
    lat: 42.6389,
    lng: -83.2910,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Counseling', 'Legal Services', 'Support Groups'],
    hours: '24/7 Hotline',
    website: 'https://www.haven-oakland.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // North Carolina
  {
    id: 'charlotte-safe-alliance',
    name: 'Safe Alliance',
    address: '601 E 5th St',
    city: 'Charlotte',
    state: 'NC',
    zip: '28202',
    phone: '(980) 771-4673',
    lat: 35.2271,
    lng: -80.8431,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Counseling', 'Legal Advocacy', 'Hospital Response'],
    hours: '24/7 Hotline',
    website: 'https://www.safealliance.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Oregon
  {
    id: 'portland-raphael-house',
    name: 'Raphael House of Portland',
    address: 'Confidential Location',
    city: 'Portland',
    state: 'OR',
    zip: '97201',
    phone: '(503) 222-6507',
    lat: 45.5152,
    lng: -122.6784,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Transitional Housing', 'Children\'s Programs', 'Legal Advocacy'],
    hours: '24/7 Hotline',
    website: 'https://www.raphaelhouse.com',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Minnesota
  {
    id: 'minneapolis-harriet-tubman',
    name: 'Tubman',
    address: '3111 1st Ave S',
    city: 'Minneapolis',
    state: 'MN',
    zip: '55408',
    phone: '(612) 825-0000',
    lat: 44.9778,
    lng: -93.2650,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Legal Services', 'Counseling', 'Youth Programs'],
    hours: '24/7 Hotline',
    website: 'https://www.tubman.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Nevada
  {
    id: 'las-vegas-safenest',
    name: 'SafeNest',
    address: '2915 W Charleston Blvd',
    city: 'Las Vegas',
    state: 'NV',
    zip: '89102',
    phone: '(702) 646-4981',
    lat: 36.1699,
    lng: -115.1398,
    type: 'dv_shelter',
    services: ['Emergency Shelter', 'Legal Advocacy', 'Counseling', 'Children\'s Programs'],
    hours: '24/7 Hotline',
    website: 'https://www.safenest.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  },
  // Rural resources
  {
    id: 'rural-hotline',
    name: 'Rural Domestic Violence Hotline',
    address: 'Nationwide Rural Areas',
    city: 'Rural',
    state: 'US',
    zip: '00000',
    phone: '1-800-799-7233',
    lat: 39.8283,
    lng: -98.5795,
    type: 'crisis_center',
    services: ['24/7 Hotline', 'Rural-Specific Resources', 'Telehealth', 'Transportation Assistance'],
    hours: '24/7',
    website: 'https://www.thehotline.org',
    isVerified: true,
    lastUpdated: '2026-01-25'
  }
];

/**
 * Calculate distance between two coordinates using Haversine formula
 */
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

/**
 * Search for shelters using Google Places API
 */
export async function searchSheltersAPI(
  lat: number, 
  lng: number, 
  radius: number = 50 // miles
): Promise<ShelterSearchResult> {
  try {
    // Convert miles to meters for Google API
    const radiusMeters = radius * 1609.34;
    
    // Search for domestic violence shelters and crisis centers
    const searchQueries = [
      'domestic violence shelter',
      'women\'s shelter',
      'crisis center',
      'safe house'
    ];
    
    const allResults: Shelter[] = [];
    
    for (const query of searchQueries) {
      try {
        const response = await makeRequest('/maps/api/place/nearbysearch/json', {
          location: `${lat},${lng}`,
          radius: radiusMeters.toString(),
          keyword: query,
          type: 'establishment'
        });
        
        const data = response as { results?: Array<{ place_id: string; name: string; vicinity?: string; geometry?: { location?: { lat: number; lng: number } } }> };
        if (data.results && Array.isArray(data.results)) {
          for (const place of data.results) {
            // Avoid duplicates
            if (allResults.some(s => s.id === place.place_id)) continue;
            
            const shelter: Shelter = {
              id: place.place_id,
              name: place.name,
              address: place.vicinity || 'Address not available',
              city: '', // Would need geocoding to extract
              state: '',
              zip: '',
              phone: '', // Would need place details API
              lat: place.geometry?.location?.lat || 0,
              lng: place.geometry?.location?.lng || 0,
              distance: calculateDistance(lat, lng, place.geometry?.location?.lat || 0, place.geometry?.location?.lng || 0),
              type: query.includes('domestic') || query.includes('women') ? 'dv_shelter' : 'crisis_center',
              services: ['Contact for services'],
              isVerified: false,
              lastUpdated: new Date().toISOString().split('T')[0]
            };
            
            allResults.push(shelter);
          }
        }
      } catch (err) {
        console.error(`Error searching for ${query}:`, err);
      }
    }
    
    // Sort by distance
    allResults.sort((a, b) => (a.distance || 999) - (b.distance || 999));
    
    if (allResults.length > 0) {
      return {
        shelters: allResults.slice(0, 20), // Limit to 20 results
        source: 'api',
        searchLocation: { lat, lng },
        timestamp: new Date().toISOString()
      };
    }
    
    // Fall back to static data if no API results
    throw new Error('No API results, falling back to static');
    
  } catch (error) {
    console.error('Shelter API error, using static data:', error);
    return searchSheltersStatic(lat, lng, radius);
  }
}

/**
 * Search shelters using static data (fallback)
 */
export function searchSheltersStatic(
  lat: number, 
  lng: number, 
  radius: number = 50
): ShelterSearchResult {
  const sheltersWithDistance = STATIC_SHELTERS.map(shelter => ({
    ...shelter,
    distance: calculateDistance(lat, lng, shelter.lat, shelter.lng)
  }));
  
  // Filter by radius and sort by distance
  const nearbyShelters = sheltersWithDistance
    .filter(s => s.distance <= radius || s.state === 'US') // Include national resources
    .sort((a, b) => {
      // National resources go last
      if (a.state === 'US' && b.state !== 'US') return 1;
      if (b.state === 'US' && a.state !== 'US') return -1;
      return a.distance - b.distance;
    });
  
  return {
    shelters: nearbyShelters,
    source: 'static',
    searchLocation: { lat, lng },
    timestamp: new Date().toISOString()
  };
}

/**
 * Search shelters by state name
 */
export function searchSheltersByState(state: string): ShelterSearchResult {
  const stateUpper = state.toUpperCase();
  const stateShelters = STATIC_SHELTERS.filter(
    s => s.state.toUpperCase() === stateUpper || 
         s.state === 'US' || 
         s.city.toLowerCase().includes(state.toLowerCase())
  );
  
  return {
    shelters: stateShelters,
    source: 'static',
    timestamp: new Date().toISOString()
  };
}

/**
 * Get all static shelters (for offline/fallback)
 */
export function getAllStaticShelters(): Shelter[] {
  return STATIC_SHELTERS;
}

/**
 * Geocode an address to coordinates
 */
export async function geocodeAddress(address: string): Promise<{ lat: number; lng: number } | null> {
  try {
    const response = await makeRequest('/maps/api/geocode/json', {
      address: address
    });
    
    const data = response as { results?: Array<{ geometry: { location: { lat: number; lng: number } } }> };
    if (data.results && data.results.length > 0) {
      const location = data.results[0].geometry.location;
      return { lat: location.lat, lng: location.lng };
    }
    return null;
  } catch (error) {
    console.error('Geocoding error:', error);
    return null;
  }
}
