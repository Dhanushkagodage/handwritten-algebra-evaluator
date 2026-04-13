import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <Link to="/" className="text-xl font-bold text-blue-700">
          AlgebraEval
        </Link>
        <div className="flex gap-6 text-sm font-medium">
          <Link to="/" className="text-gray-600 hover:text-blue-700 transition-colors">
            Home
          </Link>
          <Link to="/evaluate" className="text-gray-600 hover:text-blue-700 transition-colors">
            Evaluate
          </Link>
        </div>
      </div>
    </nav>
  )
}
