import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import Chat from "./pages/Chat";
import LearnMore from "./pages/LearnMore";
import Login from "./pages/Login";
import Register from "./pages/Register";
import ForgotPassword from "./pages/ForgotPassword";
import SafetyPlan from "./pages/SafetyPlan";
import PeerSupport from "./pages/PeerSupport";
import Journal from "./pages/Journal";
import PrivacyPolicy from "./pages/PrivacyPolicy";
import TermsOfService from "./pages/TermsOfService";
import Disclaimer from "./pages/Disclaimer";
import OfflineGrounding from "./pages/OfflineGrounding";
import Resources from "./pages/Resources";
import TherapistPortal from "./pages/TherapistPortal";
import Dashboard from "./pages/Dashboard";
import Settings from "./pages/Settings";
import { VideoCallPage } from "./components/VideoCall";
import ConsentDialog from "./components/ConsentDialog";
import PanicButton from "./components/PanicButton";
import { LanguageProvider } from "./contexts/LanguageContext";
import { LanguageSelector } from "./components/LanguageSelector";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/chat" component={Chat} />
      <Route path="/learn-more" component={LearnMore} />
      <Route path="/login" component={Login} />
      <Route path="/register" component={Register} />
      <Route path="/forgot-password" component={ForgotPassword} />
      <Route path="/safety-plan" component={SafetyPlan} />
      <Route path="/peer-support" component={PeerSupport} />
      <Route path="/journal" component={Journal} />
      <Route path="/privacy" component={PrivacyPolicy} />
      <Route path="/terms" component={TermsOfService} />
      <Route path="/disclaimer" component={Disclaimer} />
      <Route path="/grounding" component={OfflineGrounding} />
      <Route path="/resources" component={Resources} />
      <Route path="/therapist" component={TherapistPortal} />
      <Route path="/dashboard" component={Dashboard} />
      <Route path="/settings" component={Settings} />
      <Route path="/video-call" component={VideoCallPage} />
      <Route path="/404" component={NotFound} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <LanguageProvider>
          <TooltipProvider>
            <Toaster />
            <ConsentDialog />
            <PanicButton />
            <Router />
          </TooltipProvider>
        </LanguageProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
