import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Evaluate from './pages/Evaluate'
import Results from './pages/Results'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      {/* Wide enough for the evaluate page's side-by-side upload + results
          layout; every page sets its own narrower max-width inside. */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/evaluate" element={<Evaluate />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
