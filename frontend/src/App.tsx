import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import Navbar from "./components/Navbar";
import Assistant from "./pages/Assistant";
import Documentation from "./pages/Documentation";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-b from-gray-950 to-gray-900 text-gray-100">
        <Navbar />
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<Assistant />} />
            <Route path="/docs" element={<Documentation />} />
          </Routes>
        </AnimatePresence>
      </div>
    </Router>
  );
}

export default App;
