import { useEffect, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Link } from "react-router-dom"
import { Menu, Github, X } from "lucide-react"

const navConfig = [
  {
    label: "Assistant",
    href: "/",
  },
  {
    label: "Documentation",
    href: "/docs",
  },
  {
    label: "GitHub",
    href: "https://github.com/Akash8585/claso",
  },
]

interface NavbarProps {
  className?: string
}

export default function Navbar({ className = "" }: NavbarProps) {
  const [scrollY, setScrollY] = useState(0)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const handleScroll = useCallback(() => {
    setScrollY(window.scrollY)
  }, [])

  useEffect(() => {
    window.addEventListener("scroll", handleScroll, { passive: true })
    return () => window.removeEventListener("scroll", handleScroll)
  }, [handleScroll])

  // Close mobile menu when clicking outside or on escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsMobileMenuOpen(false)
      }
    }

    if (isMobileMenuOpen) {
      document.addEventListener("keydown", handleKeyDown)
      document.body.style.overflow = "hidden"
    } else {
      document.body.style.overflow = "unset"
    }

    return () => {
      document.removeEventListener("keydown", handleKeyDown)
      document.body.style.overflow = "unset"
    }
  }, [isMobileMenuOpen])

  // Calculate navbar scale and opacity based on scroll
  const navScale = scrollY > 200 ? 0.98 : 1
  const navOpacity = scrollY > 50 ? 0.95 : 0.85

  const closeMobileMenu = () => setIsMobileMenuOpen(false)

  return (
    <>
      <motion.nav
        className={`fixed top-4 left-4 right-4 z-50 max-w-4xl mx-auto px-4 py-3 bg-gray-950/85 backdrop-blur-xl border border-gray-800/50 rounded-2xl shadow-2xl ${className}`}
        initial={{ opacity: 0, y: -20 }}
        animate={{
          opacity: 1,
          y: 0,
          scale: navScale,
          backgroundColor: `rgba(3, 7, 18, ${navOpacity})`,
        }}
        transition={{ duration: 0.3, type: "spring", bounce: 0.05 }}
      >
        <div className="flex justify-between items-center">
          {/* Brand */}
          <Link to="/" className="flex items-center">
            <div className="font-bold text-2xl bg-gradient-to-r from-blue-400 via-purple-500 to-blue-600 text-transparent bg-clip-text">
              Claso
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-2 lg:gap-4">
            {navConfig.map((item) => (
              item.href.startsWith('http') ? (
                <a
                  key={item.label}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-medium text-gray-300 hover:text-white transition-colors duration-200 relative group px-2 py-1"
                >
                  {item.label}
                  <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-blue-400 to-purple-500 transition-all duration-200 group-hover:w-full rounded-full"></span>
                </a>
              ) : (
                <Link
                  key={item.label}
                  to={item.href}
                  className="text-sm font-medium text-gray-300 hover:text-white transition-colors duration-200 relative group px-2 py-1"
                >
                  {item.label}
                  <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-blue-400 to-purple-500 transition-all duration-200 group-hover:w-full rounded-full"></span>
                </Link>
              )
            ))}
          </div>

          {/* Right Side */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Social Links - Desktop */}
            <div className="hidden md:flex items-center gap-1">
              <a
                href="https://github.com/Akash8585/claso"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 text-gray-400 hover:text-white transition-colors duration-200 hover:bg-gray-800/50 rounded-lg"
                aria-label="GitHub Repository"
              >
                <Github className="w-4 h-4" />
              </a>

            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 text-gray-400 hover:text-white transition-colors duration-200 hover:bg-gray-800/50 rounded-lg"
              aria-label={isMobileMenuOpen ? "Close menu" : "Open menu"}
              aria-expanded={isMobileMenuOpen}
            >
              {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="fixed top-20 left-4 right-4 z-40 max-w-sm mx-auto bg-gray-950/95 backdrop-blur-xl border border-gray-800/50 rounded-2xl shadow-2xl p-6 md:hidden"
          >
            <div className="flex flex-col gap-4">
              {navConfig.map((item) => (
                item.href.startsWith('http') ? (
                  <a
                    key={item.label}
                    href={item.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={closeMobileMenu}
                    className="text-gray-300 hover:text-white transition-colors duration-200 py-2 border-b border-gray-800/50 last:border-b-0"
                  >
                    {item.label}
                  </a>
                ) : (
                  <Link
                    key={item.label}
                    to={item.href}
                    onClick={closeMobileMenu}
                    className="text-gray-300 hover:text-white transition-colors duration-200 py-2 border-b border-gray-800/50 last:border-b-0"
                  >
                    {item.label}
                  </Link>
                )
              ))}

              {/* Mobile Social Links */}
              <div className="flex items-center gap-4 pt-4 border-t border-gray-800/50">
                <a
                  href="https://github.com/Akash8585/claso"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors duration-200"
                  aria-label="GitHub Repository"
                >
                  <Github className="w-4 h-4" />
                  <span className="text-sm">GitHub</span>
                </a>

              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Mobile Menu Backdrop */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-30 md:hidden"
            onClick={closeMobileMenu}
          />
        )}
      </AnimatePresence>
    </>
  )
}
