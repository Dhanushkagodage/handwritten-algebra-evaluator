import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav className="bg-white border-b border-gray-100 sticky top-0 z-10">
      <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center">
            <span className="text-white text-xs font-bold">AE</span>
          </div>
          <span className="font-semibold text-gray-900 text-sm">AlgebraEval</span>
        </Link>
        <div className="flex gap-1">
          {[
            { to: '/', label: 'Home' },
            { to: '/evaluate', label: 'Evaluate' },
          ].map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                pathname === to
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-500 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
