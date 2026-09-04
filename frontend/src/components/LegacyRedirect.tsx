import { Navigate, useLocation, useParams } from 'react-router-dom'

export default function LegacyRedirect({ to }: { to: string | ((params: Readonly<Record<string, string | undefined>>) => string) }) {
  const location = useLocation()
  const params = useParams()
  const target = typeof to === 'function' ? to(params) : to
  return <Navigate replace to={`${target}${location.search}${location.hash}`} />
}
