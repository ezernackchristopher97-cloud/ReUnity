import { useState, useEffect, useCallback } from "react";
import { X, Calculator, Cloud, Sun, CloudRain, Wind, Settings, Key, Trash2 } from "lucide-react";

// Decoy Calculator App
function DecoyCalculator({ onExit }: { onExit: () => void }) {
  const [display, setDisplay] = useState("0");
  const [previousValue, setPreviousValue] = useState<number | null>(null);
  const [operation, setOperation] = useState<string | null>(null);
  const [waitingForOperand, setWaitingForOperand] = useState(false);

  const inputDigit = (digit: string) => {
    if (waitingForOperand) {
      setDisplay(digit);
      setWaitingForOperand(false);
    } else {
      setDisplay(display === "0" ? digit : display + digit);
    }
  };

  const inputDecimal = () => {
    if (waitingForOperand) {
      setDisplay("0.");
      setWaitingForOperand(false);
    } else if (!display.includes(".")) {
      setDisplay(display + ".");
    }
  };

  const clear = () => {
    setDisplay("0");
    setPreviousValue(null);
    setOperation(null);
    setWaitingForOperand(false);
  };

  const performOperation = (nextOperation: string) => {
    const inputValue = parseFloat(display);

    if (previousValue === null) {
      setPreviousValue(inputValue);
    } else if (operation) {
      const currentValue = previousValue || 0;
      let newValue = currentValue;

      switch (operation) {
        case "+":
          newValue = currentValue + inputValue;
          break;
        case "-":
          newValue = currentValue - inputValue;
          break;
        case "×":
          newValue = currentValue * inputValue;
          break;
        case "÷":
          newValue = inputValue !== 0 ? currentValue / inputValue : 0;
          break;
      }

      setDisplay(String(newValue));
      setPreviousValue(newValue);
    }

    setWaitingForOperand(true);
    setOperation(nextOperation);
  };

  const calculate = () => {
    if (!operation || previousValue === null) return;

    const inputValue = parseFloat(display);
    let newValue = previousValue;

    switch (operation) {
      case "+":
        newValue = previousValue + inputValue;
        break;
      case "-":
        newValue = previousValue - inputValue;
        break;
      case "×":
        newValue = previousValue * inputValue;
        break;
      case "÷":
        newValue = inputValue !== 0 ? previousValue / inputValue : 0;
        break;
    }

    setDisplay(String(newValue));
    setPreviousValue(null);
    setOperation(null);
    setWaitingForOperand(true);
  };

  // Secret exit: Press 0 five times rapidly
  const [zeroCount, setZeroCount] = useState(0);
  const [lastZeroTime, setLastZeroTime] = useState(0);

  const handleDigit = (digit: string) => {
    if (digit === "0") {
      const now = Date.now();
      if (now - lastZeroTime < 500) {
        const newCount = zeroCount + 1;
        setZeroCount(newCount);
        if (newCount >= 5) {
          onExit();
          return;
        }
      } else {
        setZeroCount(1);
      }
      setLastZeroTime(now);
    } else {
      setZeroCount(0);
    }
    inputDigit(digit);
  };

  const Button = ({ value, onClick, className = "" }: { value: string; onClick: () => void; className?: string }) => (
    <button
      onClick={onClick}
      className={`w-16 h-16 rounded-full text-2xl font-medium transition-all active:scale-95 ${className}`}
    >
      {value}
    </button>
  );

  return (
    <div className="fixed inset-0 z-[9999] bg-black flex flex-col items-center justify-center">
      <div className="w-full max-w-sm px-6">
        {/* Display */}
        <div className="text-right text-white text-6xl font-light mb-8 overflow-hidden">
          {display.length > 9 ? parseFloat(display).toExponential(4) : display}
        </div>

        {/* Buttons */}
        <div className="grid grid-cols-4 gap-3">
          <Button value="AC" onClick={clear} className="bg-gray-400 text-black" />
          <Button value="+/-" onClick={() => setDisplay(String(-parseFloat(display)))} className="bg-gray-400 text-black" />
          <Button value="%" onClick={() => setDisplay(String(parseFloat(display) / 100))} className="bg-gray-400 text-black" />
          <Button value="÷" onClick={() => performOperation("÷")} className="bg-orange-500 text-white" />

          <Button value="7" onClick={() => handleDigit("7")} className="bg-gray-700 text-white" />
          <Button value="8" onClick={() => handleDigit("8")} className="bg-gray-700 text-white" />
          <Button value="9" onClick={() => handleDigit("9")} className="bg-gray-700 text-white" />
          <Button value="×" onClick={() => performOperation("×")} className="bg-orange-500 text-white" />

          <Button value="4" onClick={() => handleDigit("4")} className="bg-gray-700 text-white" />
          <Button value="5" onClick={() => handleDigit("5")} className="bg-gray-700 text-white" />
          <Button value="6" onClick={() => handleDigit("6")} className="bg-gray-700 text-white" />
          <Button value="-" onClick={() => performOperation("-")} className="bg-orange-500 text-white" />

          <Button value="1" onClick={() => handleDigit("1")} className="bg-gray-700 text-white" />
          <Button value="2" onClick={() => handleDigit("2")} className="bg-gray-700 text-white" />
          <Button value="3" onClick={() => handleDigit("3")} className="bg-gray-700 text-white" />
          <Button value="+" onClick={() => performOperation("+")} className="bg-orange-500 text-white" />

          <button
            onClick={() => handleDigit("0")}
            className="col-span-2 h-16 rounded-full text-2xl font-medium bg-gray-700 text-white text-left pl-7 transition-all active:scale-95"
          >
            0
          </button>
          <Button value="." onClick={inputDecimal} className="bg-gray-700 text-white" />
          <Button value="=" onClick={calculate} className="bg-orange-500 text-white" />
        </div>

        {/* Hidden hint */}
        <p className="text-gray-900 text-xs text-center mt-8">
          Press 0 five times quickly to return
        </p>
      </div>
    </div>
  );
}

// Decoy Weather App
function DecoyWeather({ onExit }: { onExit: () => void }) {
  const [tapCount, setTapCount] = useState(0);
  const [lastTapTime, setLastTapTime] = useState(0);

  // Secret exit: Tap the temperature 5 times rapidly
  const handleTempTap = () => {
    const now = Date.now();
    if (now - lastTapTime < 500) {
      const newCount = tapCount + 1;
      setTapCount(newCount);
      if (newCount >= 5) {
        onExit();
        return;
      }
    } else {
      setTapCount(1);
    }
    setLastTapTime(now);
  };

  const currentHour = new Date().getHours();
  const isNight = currentHour < 6 || currentHour > 18;

  return (
    <div className={`fixed inset-0 z-[9999] flex flex-col ${isNight ? 'bg-gradient-to-b from-indigo-900 to-slate-900' : 'bg-gradient-to-b from-blue-400 to-blue-600'}`}>
      {/* Header */}
      <div className="pt-12 px-6 text-center text-white">
        <p className="text-lg opacity-80">Current Location</p>
        <h1 className="text-3xl font-light">My Location</h1>
      </div>

      {/* Main Temperature */}
      <div className="flex-1 flex flex-col items-center justify-center -mt-20">
        <div className="mb-4">
          {isNight ? (
            <Cloud className="w-24 h-24 text-white/80" />
          ) : (
            <Sun className="w-24 h-24 text-yellow-300" />
          )}
        </div>
        <button 
          onClick={handleTempTap}
          className="text-white text-9xl font-thin"
        >
          72°
        </button>
        <p className="text-white/80 text-xl mt-2">Partly Cloudy</p>
        <p className="text-white/60 text-lg">H:78° L:65°</p>
      </div>

      {/* Hourly Forecast */}
      <div className="bg-white/10 backdrop-blur-sm mx-4 mb-4 rounded-2xl p-4">
        <div className="flex justify-between text-white text-center">
          {["Now", "1PM", "2PM", "3PM", "4PM", "5PM"].map((time, i) => (
            <div key={time} className="flex flex-col items-center gap-2">
              <span className="text-sm opacity-80">{time}</span>
              {i % 2 === 0 ? (
                <Sun className="w-6 h-6 text-yellow-300" />
              ) : (
                <Cloud className="w-6 h-6" />
              )}
              <span className="text-lg">{72 + i}°</span>
            </div>
          ))}
        </div>
      </div>

      {/* 5-Day Forecast */}
      <div className="bg-white/10 backdrop-blur-sm mx-4 mb-8 rounded-2xl p-4">
        {["Today", "Tomorrow", "Wednesday", "Thursday", "Friday"].map((day, i) => (
          <div key={day} className="flex items-center justify-between text-white py-2 border-b border-white/10 last:border-0">
            <span className="w-24">{day}</span>
            <div className="flex items-center gap-2">
              {i % 3 === 0 ? <CloudRain className="w-5 h-5" /> : i % 2 === 0 ? <Sun className="w-5 h-5 text-yellow-300" /> : <Cloud className="w-5 h-5" />}
              {i % 3 === 0 && <span className="text-xs text-blue-300">60%</span>}
            </div>
            <div className="flex gap-4">
              <span className="opacity-60">{65 + i}°</span>
              <span>{78 - i}°</span>
            </div>
          </div>
        ))}
      </div>

      {/* Hidden hint */}
      <p className="text-white/5 text-xs text-center mb-4">
        Tap temperature 5 times to return
      </p>
    </div>
  );
}

// Safe word storage key
const SAFE_WORD_KEY = 'reunity_safe_word';

// Main Panic Button Component
export function PanicButton() {
  const [decoyMode, setDecoyMode] = useState<"calculator" | "weather" | null>(null);
  const [showMenu, setShowMenu] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  // Safe word state
  const [safeWord, setSafeWord] = useState('');
  const [newSafeWord, setNewSafeWord] = useState('');
  const [typedBuffer, setTypedBuffer] = useState('');

  // Keyboard shortcut: Triple press Escape
  const [escapeCount, setEscapeCount] = useState(0);
  const [lastEscapeTime, setLastEscapeTime] = useState(0);

  // Shake detection for mobile
  const [lastShakeTime, setLastShakeTime] = useState(0);
  const [shakeCount, setShakeCount] = useState(0);
  const SHAKE_THRESHOLD = 15; // Acceleration threshold
  const SHAKE_TIMEOUT = 1000; // Time window for shake detection
  const SHAKES_REQUIRED = 3; // Number of shakes needed

  // Load safe word from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(SAFE_WORD_KEY);
    if (stored) {
      setSafeWord(stored.toLowerCase());
    }
  }, []);

  // Global keypress listener for safe word detection
  useEffect(() => {
    if (!safeWord || decoyMode) return;

    const handleKeyPress = (e: KeyboardEvent) => {
      // Only track alphanumeric keys
      if (e.key.length === 1 && /[a-zA-Z0-9]/.test(e.key)) {
        setTypedBuffer(prev => {
          const newBuffer = (prev + e.key.toLowerCase()).slice(-50); // Keep last 50 chars
          
          // Check if safe word was typed
          if (newBuffer.includes(safeWord)) {
            // Trigger panic mode!
            setDecoyMode('calculator');
            setTypedBuffer('');
            
            // Vibrate if available
            if (navigator.vibrate) {
              navigator.vibrate([50, 50, 50]);
            }
          }
          
          return newBuffer;
        });
      }
    };

    window.addEventListener('keypress', handleKeyPress);
    return () => window.removeEventListener('keypress', handleKeyPress);
  }, [safeWord, decoyMode]);

  // Save safe word
  const saveSafeWord = () => {
    const word = newSafeWord.toLowerCase().trim();
    if (word.length >= 3) {
      localStorage.setItem(SAFE_WORD_KEY, word);
      setSafeWord(word);
      setNewSafeWord('');
      setShowSettings(false);
    }
  };

  // Clear safe word
  const clearSafeWord = () => {
    localStorage.removeItem(SAFE_WORD_KEY);
    setSafeWord('');
    setNewSafeWord('');
  };

  // Handle device motion for shake detection
  useEffect(() => {
    let lastX = 0, lastY = 0, lastZ = 0;
    let lastUpdate = 0;

    const handleMotion = (event: DeviceMotionEvent) => {
      if (decoyMode) return; // Don't trigger if already in decoy mode

      const acceleration = event.accelerationIncludingGravity;
      if (!acceleration) return;

      const currentTime = Date.now();
      const timeDiff = currentTime - lastUpdate;

      if (timeDiff > 100) {
        lastUpdate = currentTime;

        const x = acceleration.x || 0;
        const y = acceleration.y || 0;
        const z = acceleration.z || 0;

        const deltaX = Math.abs(x - lastX);
        const deltaY = Math.abs(y - lastY);
        const deltaZ = Math.abs(z - lastZ);

        // Check if movement exceeds threshold
        if (deltaX + deltaY + deltaZ > SHAKE_THRESHOLD) {
          const now = Date.now();
          
          if (now - lastShakeTime < SHAKE_TIMEOUT) {
            const newCount = shakeCount + 1;
            setShakeCount(newCount);
            
            if (newCount >= SHAKES_REQUIRED) {
              // Trigger panic mode!
              setDecoyMode("calculator");
              setShakeCount(0);
              
              // Vibrate if available to confirm
              if (navigator.vibrate) {
                navigator.vibrate([50, 50, 50]);
              }
            }
          } else {
            setShakeCount(1);
          }
          setLastShakeTime(now);
        }

        lastX = x;
        lastY = y;
        lastZ = z;
      }
    };

    // Request permission for iOS 13+
    const requestMotionPermission = async () => {
      if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
        try {
          const permission = await (DeviceMotionEvent as any).requestPermission();
          if (permission === 'granted') {
            window.addEventListener('devicemotion', handleMotion);
          }
        } catch (e) {
          console.log('Motion permission denied');
        }
      } else {
        // Non-iOS or older iOS
        window.addEventListener('devicemotion', handleMotion);
      }
    };

    requestMotionPermission();

    return () => {
      window.removeEventListener('devicemotion', handleMotion);
    };
  }, [decoyMode, lastShakeTime, shakeCount]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === "Escape") {
      const now = Date.now();
      if (now - lastEscapeTime < 500) {
        const newCount = escapeCount + 1;
        setEscapeCount(newCount);
        if (newCount >= 3 && !decoyMode) {
          // Show calculator by default on triple escape
          setDecoyMode("calculator");
        }
      } else {
        setEscapeCount(1);
      }
      setLastEscapeTime(now);
    }
  }, [escapeCount, lastEscapeTime, decoyMode]);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const exitDecoy = () => {
    setDecoyMode(null);
    setShowMenu(false);
  };

  if (decoyMode === "calculator") {
    return <DecoyCalculator onExit={exitDecoy} />;
  }

  if (decoyMode === "weather") {
    return <DecoyWeather onExit={exitDecoy} />;
  }

  return (
    <>
      {/* Floating Panic Button - disguised as settings gear */}
      <button
        onClick={() => setShowMenu(!showMenu)}
        className="fixed bottom-20 right-4 z-50 w-12 h-12 bg-gray-800/50 hover:bg-gray-700/50 rounded-full flex items-center justify-center transition-all group"
        title="Quick Settings"
      >
        <div className="w-6 h-6 relative">
          {/* Gear icon that looks like settings */}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-full h-full text-gray-400 group-hover:text-white transition-colors">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
          </svg>
        </div>
      </button>

      {/* Quick Menu */}
      {showMenu && !showSettings && (
        <div className="fixed bottom-36 right-4 z-50 bg-gray-900 border border-gray-700 rounded-xl shadow-2xl overflow-hidden w-56">
          <div className="p-2 border-b border-gray-700">
            <p className="text-xs text-gray-500 px-2">Quick Switch</p>
          </div>
          <button
            onClick={() => setDecoyMode("calculator")}
            className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-800 transition-colors text-left"
          >
            <Calculator className="w-5 h-5 text-gray-400" />
            <span className="text-white">Calculator</span>
          </button>
          <button
            onClick={() => setDecoyMode("weather")}
            className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-800 transition-colors text-left"
          >
            <Cloud className="w-5 h-5 text-blue-400" />
            <span className="text-white">Weather</span>
          </button>
          <div className="border-t border-gray-700">
            <button
              onClick={() => setShowSettings(true)}
              className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-800 transition-colors text-left"
            >
              <Key className="w-5 h-5 text-emerald-400" />
              <div className="flex flex-col">
                <span className="text-white text-sm">Safe Word</span>
                {safeWord ? (
                  <span className="text-xs text-emerald-400">Active: ****</span>
                ) : (
                  <span className="text-xs text-gray-500">Not set</span>
                )}
              </div>
            </button>
          </div>
          <div className="p-2 border-t border-gray-700">
            <p className="text-xs text-gray-600 px-2">ESC 3x, shake, or type safe word</p>
          </div>
        </div>
      )}

      {/* Safe Word Settings */}
      {showSettings && (
        <div className="fixed bottom-36 right-4 z-50 bg-gray-900 border border-gray-700 rounded-xl shadow-2xl overflow-hidden w-72">
          <div className="p-3 border-b border-gray-700 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Key className="w-4 h-4 text-emerald-400" />
              <span className="text-white font-medium">Safe Word Settings</span>
            </div>
            <button
              onClick={() => setShowSettings(false)}
              className="text-gray-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          
          <div className="p-4">
            <p className="text-xs text-gray-400 mb-3">
              Type this word anywhere in the app to instantly trigger the decoy screen.
            </p>
            
            {safeWord ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between bg-gray-800 rounded-lg px-3 py-2">
                  <span className="text-emerald-400">Safe word is set</span>
                  <button
                    onClick={clearSafeWord}
                    className="text-red-400 hover:text-red-300 flex items-center gap-1 text-sm"
                  >
                    <Trash2 className="w-3 h-3" />
                    Remove
                  </button>
                </div>
                <p className="text-xs text-gray-500">
                  For security, your safe word is hidden. Remove it to set a new one.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <input
                  type="text"
                  value={newSafeWord}
                  onChange={(e) => setNewSafeWord(e.target.value)}
                  placeholder="Enter safe word (min 3 chars)"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                  autoComplete="off"
                />
                <button
                  onClick={saveSafeWord}
                  disabled={newSafeWord.trim().length < 3}
                  className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg py-2 text-sm font-medium transition-colors"
                >
                  Save Safe Word
                </button>
                <p className="text-xs text-gray-500">
                  Choose something you can type naturally but won't type by accident.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default PanicButton;
