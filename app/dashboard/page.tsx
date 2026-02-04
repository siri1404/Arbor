import { redirect } from 'next/navigation'
import { createClient } from '@/lib/supabase/server'
import { DashboardWrapper } from '@/components/dashboard-wrapper'

export default async function DashboardPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect('/')
  }

  return <DashboardWrapper />
}
