// Simple in-memory cache with TTL
interface CacheEntry<T> {
  data: T
  expiresAt: number
}

const cache = new Map<string, CacheEntry<unknown>>()

export function getFromCache<T>(key: string): T | null {
  const entry = cache.get(key) as CacheEntry<T> | undefined
  if (!entry) return null
  
  if (Date.now() > entry.expiresAt) {
    cache.delete(key)
    return null
  }
  
  return entry.data
}

export function setInCache<T>(key: string, data: T, ttlSeconds: number): void {
  cache.set(key, {
    data,
    expiresAt: Date.now() + ttlSeconds * 1000,
  })
}

export function clearCache(): void {
  cache.clear()
}
