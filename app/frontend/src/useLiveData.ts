import { useEffect, useRef, useState } from 'react'

/** Live WebSocket subscription with auto-reconnect. Returns the latest
 * event and a connected flag; consumers refetch on new events. */
export function useLiveData(onEvent?: (e: any) => void) {
  const [lastEvent, setLastEvent] = useState<any>(null)
  const [connected, setConnected] = useState(false)
  const cbRef = useRef(onEvent)
  cbRef.current = onEvent

  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    let ws: WebSocket | null = null
    let closed = false
    let timer: any

    const connect = () => {
      ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) timer = setTimeout(connect, 2000)
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (m) => {
        try {
          const e = JSON.parse(m.data)
          setLastEvent(e)
          cbRef.current?.(e)
        } catch { /* ignore */ }
      }
    }
    connect()
    return () => {
      closed = true
      clearTimeout(timer)
      ws?.close()
    }
  }, [])

  return { lastEvent, connected }
}