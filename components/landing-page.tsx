'use client'

import React from "react"

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Loader2 } from 'lucide-react'

export function LandingPage() {
  const [isLogin, setIsLogin] = useState(true)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const router = useRouter()
  const supabase = createClient()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setMessage(null)

    try {
      if (isLogin) {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        })
        if (error) throw error
        router.push('/dashboard')
        router.refresh()
      } else {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: `${window.location.origin}/dashboard`,
          },
        })
        if (error) throw error
        setMessage('Check your email to confirm your account')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div 
      className="min-h-screen w-full bg-cover bg-center bg-no-repeat relative"
      style={{
        backgroundImage: 'url(/images/arbor-bg.jpg)',
      }}
    >
      {/* Logo - Top Left */}
      <div className="absolute top-8 left-8 z-20 flex items-center gap-2">
        {/* Geometric Logo */}
        <div className="relative w-8 h-8 flex items-center justify-center">
          {/* Tree/Growth geometric shape - interconnected circles and lines */}
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            {/* Root circle */}
            <circle cx="16" cy="26" r="2" fill="white" />
            {/* Trunk line */}
            <line x1="16" y1="24" x2="16" y2="14" stroke="white" strokeWidth="1.5" />
            {/* Main branch circles */}
            <circle cx="8" cy="12" r="2" fill="white" />
            <circle cx="24" cy="12" r="2" fill="white" />
            <circle cx="16" cy="8" r="2.5" fill="white" />
            {/* Branch lines */}
            <line x1="16" y1="14" x2="8" y2="12" stroke="white" strokeWidth="1.5" />
            <line x1="16" y1="14" x2="24" y2="12" stroke="white" strokeWidth="1.5" />
            <line x1="16" y1="14" x2="16" y2="10.5" stroke="white" strokeWidth="1.5" />
            {/* Top leaves */}
            <circle cx="4" cy="6" r="1.5" fill="white" opacity="0.8" />
            <circle cx="12" cy="4" r="1.5" fill="white" opacity="0.8" />
            <circle cx="20" cy="4" r="1.5" fill="white" opacity="0.8" />
            <circle cx="28" cy="6" r="1.5" fill="white" opacity="0.8" />
            {/* Connecting lines to leaves */}
            <line x1="8" y1="12" x2="4" y2="6" stroke="white" strokeWidth="1" opacity="0.6" />
            <line x1="8" y1="12" x2="12" y2="4" stroke="white" strokeWidth="1" opacity="0.6" />
            <line x1="24" y1="12" x2="20" y2="4" stroke="white" strokeWidth="1" opacity="0.6" />
            <line x1="24" y1="12" x2="28" y2="6" stroke="white" strokeWidth="1" opacity="0.6" />
          </svg>
        </div>
        
        {/* Logo text */}
        <h1 className="text-2xl font-bold text-white tracking-tight drop-shadow-lg font-sans">
          ARBOR
        </h1>
      </div>

      {/* Glass Card - Center */}
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="w-full max-w-md">
          <div 
            className="rounded-2xl p-8 shadow-2xl border border-[rgba(255,255,255,1)] opacity-85"
            style={{
              background: 'rgba(255, 255, 255, 0.95)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
            }}
          >
            {/* Card Header */}
            <div className="text-center mb-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                {isLogin ? 'Welcome back' : 'Create account'}
              </h2>
              <p className="text-gray-600 text-sm">
                {isLogin 
                  ? 'Sign in to access your portfolio' 
                  : 'Start your journey to smarter investing'}
              </p>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="space-y-2">
                <Label htmlFor="email" className="text-gray-700 text-sm font-medium">
                  Email
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="bg-white border-gray-300 text-gray-900 placeholder:text-gray-400 focus:border-blue-400 focus:ring-blue-200"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-gray-700 text-sm font-medium">
                  Password
                </Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  className="bg-white border-gray-300 text-gray-900 placeholder:text-gray-400 focus:border-blue-400 focus:ring-blue-200"
                />
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-red-50 border border-red-200">
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              )}

              {message && (
                <div className="p-3 rounded-lg bg-green-50 border border-green-200">
                  <p className="text-green-700 text-sm">{message}</p>
                </div>
              )}

              <Button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 text-white hover:bg-blue-700 font-medium py-5"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {isLogin ? 'Signing in...' : 'Creating account...'}
                  </>
                ) : (
                  isLogin ? 'Sign in' : 'Create account'
                )}
              </Button>
            </form>

            {/* Toggle */}
            <div className="mt-6 text-center">
              <p className="text-gray-600 text-sm">
                {isLogin ? "Don't have an account?" : 'Already have an account?'}
                <button
                  type="button"
                  onClick={() => {
                    setIsLogin(!isLogin)
                    setError(null)
                    setMessage(null)
                  }}
                  className="ml-2 text-blue-600 font-medium hover:underline"
                >
                  {isLogin ? 'Sign up' : 'Sign in'}
                </button>
              </p>
            </div>
          </div>

          {/* Tagline below card */}
          <p className="text-center text-white text-sm mt-8 drop-shadow-lg font-medium">
            Grow your portfolio with AI-powered insights
          </p>
        </div>
      </div>
    </div>
  )
}
