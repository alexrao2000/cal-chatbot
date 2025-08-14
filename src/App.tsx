import { useState } from 'react'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('')
  const [answer, setAnswer] = useState('')
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function ask() {
    if (!prompt.trim()) return
    setLoading(true)
    setError(null)
    setAnswer('')
    try {
      const res = await fetch('http://localhost:5050/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'web', message: prompt, user_email: email || undefined }),
      })
      const contentType = res.headers.get('content-type') || ''
      const raw = await res.text()
      let data = null
      if (raw && contentType.includes('application/json')) {
        try { data = JSON.parse(raw) } catch { /* ignore */ }
      }
      if (!res.ok) {
        const message = (data && data.error) ? data.error : (raw || `HTTP ${res.status}`)
        throw new Error(message)
      }
      setAnswer((data && data.text) ? data.text : raw)
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('Something went wrong')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: '40px auto', padding: 16 }}>
      <h2>Ask GPT</h2>
      <div style={{ marginBottom: 12 }}>
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="(Optional) Your email for booking lookups"
          style={{ width: '100%', marginBottom: 8 }}
        />
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={5}
          placeholder="Type your question here..."
          style={{ width: '100%' }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault()
              ask()
            }
          }}
        />
      </div>
      <button onClick={ask} disabled={loading || !prompt.trim()}>
        {loading ? 'Asking...' : 'Ask'}
      </button>
      {error && (
        <div style={{ color: '#b91c1c', marginTop: 12 }}>Error: {error}</div>
      )}
      <div style={{ marginTop: 24 }}>
        <h3>Answer</h3>
        <pre style={{ whiteSpace: 'pre-wrap' }}>{answer}</pre>
      </div>
    </div>
  )
}

export default App
