import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function ask() {
    if (!prompt.trim()) return
    setLoading(true)
    setError(null)
    setAnswer('')
    try {
      const endpoint = '/chat'
      const body = { session_id: 'web', message: prompt }
      
      const res = await fetch(`http://localhost:5050${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
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
      setPrompt("")
    }
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1 className="title">📅 Cal.com AI Assistant</h1>
        <p className="subtitle">Schedule meetings effortlessly with natural language</p>
      </div>
      
      <div className="chat-container">
        <div className="input-section">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            className="chat-input"
            placeholder="Try: 'Book a 30 minute meeting tomorrow at 2 PM' or 'Show my upcoming events'"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault()
                ask()
              }
            }}
          />
          <div className="input-footer">
            <span className="keyboard-hint">Press Ctrl+Enter to send</span>
            <button 
              onClick={ask} 
              disabled={loading || !prompt.trim()}
              className={`send-button ${loading ? 'loading' : ''}`}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Thinking...
                </>
              ) : (
                <>
                  <span>→</span>
                  Send
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {answer && (
          <div className="response-section">
            <div className="response-header">
              <span className="response-icon">🤖</span>
              <span>Assistant</span>
            </div>
            <div className="response-content">
              <ReactMarkdown>{answer}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
